"""
Voice Activity Detection (VAD) processor using Silero VAD model.

This module provides a SpeechDetector class that encapsulates the Silero VAD model
and state machine for detecting speech segments in audio streams.
"""

from collections import deque
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from silero_vad import load_silero_vad

# Default sample rate for VAD processing
TARGET_SAMPLE_RATE = 16000

# VAD default configuration
DEFAULT_MIN_SPEECH_SECONDS = 3.0
DEFAULT_SOFT_LIMIT_SECONDS = 60.0
DEFAULT_HARD_LIMIT_SECONDS = 120.0
DEFAULT_SPEECH_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_DURATION_MS = 2000
DEFAULT_LOOK_BACK_SECONDS = 0.5

# Whisper backend limits.
# 60s applies when return_timestamps=True (default): HF's pipeline triggers
# Whisper's sequential long-form algorithm and stitches 30s windows internally.
# When --no-sub-timestamps is set, return_timestamps=False and the pipeline
# can't decode beyond the native 30s window without chunk_length_s, so the
# backend caps at 30s in that mode.
WHISPER_HARD_LIMIT_SECONDS = 60
WHISPER_HARD_LIMIT_NO_SUB_TIMESTAMPS_SECONDS = 30
WHISPER_SOFT_LIMIT_SECONDS = 6.0

# NVIDIA-hosted Whisper (Riva gRPC) backend limits.
# Capped at 30s to stay comfortably under NVCF's 60s default request timeout
# and to match Whisper's native 30s decoding window (no server-side chunking risk).
NVIDIA_WHISPER_HARD_LIMIT_SECONDS = 30
NVIDIA_WHISPER_SOFT_LIMIT_SECONDS = 6.0

# Moonshine backend limits
MOONSHINE_STREAMING_HARD_LIMIT_SECONDS = 60
MOONSHINE_STREAMING_SOFT_LIMIT_SECONDS = 6.0
MOONSHINE_NON_STREAMING_HARD_LIMIT_SECONDS = 9
MOONSHINE_NON_STREAMING_SOFT_LIMIT_SECONDS = 6.0

# Qwen backend limits
QWEN_ASR_HARD_LIMIT_SECONDS = 60
QWEN_ASR_SOFT_LIMIT_SECONDS = 30

# GLM-ASR backend limits
GLM_ASR_HARD_LIMIT_SECONDS = 60
GLM_ASR_SOFT_LIMIT_SECONDS = 30

# Nemotron ASR backend limits.
# The backend decodes long segments in Whisper-style seek windows that always
# end at a sentence boundary, so VAD only needs to drop long silences. The hard
# limit just bounds buffered audio (~115MB at 30 min) and output latency.
NEMOTRON_ASR_HARD_LIMIT_SECONDS = 1800
NEMOTRON_ASR_SOFT_LIMIT_SECONDS = 1500


class AudioSegment:
    """Audio segment with timing information.

    Args:
        start: Start timestamp in seconds from file beginning
        audio: Audio samples as float32 array
        duration_seconds: Optional duration in seconds
    """
    def __init__(
        self,
        start: float,
        audio: npt.NDArray[np.float32],
        duration_seconds: float | None = None,
    ):
        self.start = start
        self.audio = audio
        self.duration_seconds = duration_seconds

    def __repr__(self):
        return (
            f"AudioSegment(start={self.start}, duration={self.duration_seconds})"
        )


def get_window_size_samples(sample_rate: int = 16000) -> int:
    """Get VAD window size in samples based on sample rate."""
    return 512 if sample_rate == 16000 else 256


class SpeechDetector:
    """
    Detects speech segments in audio using Silero VAD (WhisperX-style cutting).

    A segment ends naturally after ``min_silence_duration_ms`` of silence. Short
    pauses are merged into the segment, so it keeps growing across sentences.
    When a segment reaches ``hard_limit_seconds`` without such a silence, it is
    cut at the best pause between ``soft_limit_seconds`` and the hard limit:
    the longest run of non-speech windows (cut in its middle), or, when there
    is no pause at all, the window with the lowest speech probability. Audio
    after the cut carries over into the next segment, so nothing is lost.
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
        soft_limit_seconds: float | None = DEFAULT_SOFT_LIMIT_SECONDS,
        hard_limit_seconds: float = DEFAULT_HARD_LIMIT_SECONDS,
        speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
        min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
        look_back_seconds: float = DEFAULT_LOOK_BACK_SECONDS,
        on_segment_complete: Callable[[AudioSegment], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.min_speech_seconds = min_speech_seconds
        self.soft_limit_seconds = soft_limit_seconds
        self.hard_limit_seconds = hard_limit_seconds
        self.speech_threshold = speech_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.on_segment_complete = on_segment_complete

        if self.soft_limit_seconds is not None and self.soft_limit_seconds > self.hard_limit_seconds:
            raise ValueError(
                "soft_limit_seconds cannot exceed hard_limit_seconds."
            )

        # Load Silero VAD model
        self.vad_model = load_silero_vad()

        # Window configuration
        self._window_size_samples = get_window_size_samples(self.sample_rate)
        self._window_seconds = self._window_size_samples / self.sample_rate
        self.look_back_seconds = max(0.0, look_back_seconds)

        # Segment length caps, in whole VAD windows
        self._max_windows = max(1, int(self.hard_limit_seconds * self.sample_rate) // self._window_size_samples)
        self._min_cut_windows: int | None = (
            None if self.soft_limit_seconds is None
            else int(self.soft_limit_seconds * self.sample_rate) // self._window_size_samples
        )

        # State machine variables
        self._prev_has_speech = False
        self._windows: list[npt.NDArray[np.float32]] = []
        self._probs: list[float] = []
        self._trailing_silence_windows = 0
        self._has_speech_begin_timestamp: float | None = None
        self._look_back_buffer: deque[tuple[npt.NDArray[np.float32], float]] = deque()
        self._pending_non_speech_seconds = 0.0

    def process_window(
        self,
        audio_window: npt.NDArray[np.float32],
        timestamp: float,
    ) -> bool:
        """Process a single audio window through VAD."""
        if len(audio_window) != self._window_size_samples:
            raise ValueError(
                f"Audio length {len(audio_window)} does not match window size {self._window_size_samples}"
            )
        prob = float(self._speech_probability(audio_window))
        has_speech = prob > self.speech_threshold

        if not self._prev_has_speech:
            if has_speech:
                self._start_segment(audio_window, prob, timestamp)
            else:
                self._handle_non_speech(audio_window, prob)
                self._prev_has_speech = False
                return False
        else:
            if self.current_segment_duration < self.min_speech_seconds and not has_speech:
                has_speech = True
            self._windows.append(np.copy(audio_window))
            self._probs.append(prob)
            if has_speech:
                self._trailing_silence_windows = 0
            else:
                self._trailing_silence_windows += 1
                silence_ms = self._trailing_silence_windows * self._window_seconds * 1000
                if silence_ms >= self.min_silence_duration_ms:
                    self._end_segment()
                    self._prev_has_speech = False
                    return False

        if len(self._windows) >= self._max_windows:
            self._split_at_best_pause()

        self._prev_has_speech = bool(self._windows)
        return self._prev_has_speech

    def _speech_probability(self, audio: npt.NDArray[np.float32]) -> float:
        """Silero speech probability for one window."""
        audio_tensor = torch.from_numpy(audio)
        return self.vad_model(audio_tensor, self.sample_rate).item()

    def _start_segment(
        self,
        audio_window: npt.NDArray[np.float32],
        prob: float,
        timestamp: float,
    ) -> None:
        """Handle transition from no speech to speech, prepending look-back audio."""
        # Cap look-back so the new segment never starts out over the hard limit.
        while len(self._look_back_buffer) > self._max_windows - 1:
            self._look_back_buffer.popleft()
        prepended_duration = len(self._look_back_buffer) * self._window_seconds
        for prev_window, prev_prob in self._look_back_buffer:
            self._windows.append(prev_window)
            self._probs.append(prev_prob)
        self._look_back_buffer.clear()
        if prepended_duration > 0:
            self._pending_non_speech_seconds = 0.0
        self._has_speech_begin_timestamp = max(0.0, timestamp - prepended_duration)
        self._windows.append(np.copy(audio_window))
        self._probs.append(prob)
        self._trailing_silence_windows = 0

    def _handle_non_speech(self, audio_window: npt.NDArray[np.float32], prob: float) -> None:
        """Handle continued non-speech windows."""
        self._pending_non_speech_seconds = self._window_seconds

        if self.look_back_seconds <= 0.0:
            self._look_back_buffer.clear()
            return

        self._look_back_buffer.append((np.copy(audio_window), prob))
        while self._look_back_buffer and len(self._look_back_buffer) * self._window_seconds - 1e-9 > self.look_back_seconds:
            self._look_back_buffer.popleft()

    def _best_cut_index(self) -> int:
        """Pick where to cut a full segment: number of windows to emit.

        Searches windows from the soft limit (or the segment midpoint when no
        soft limit is set) to the end. Prefers the middle of the longest
        non-speech run; with no non-speech window, cuts after the lowest
        speech-probability window. Ties go to the later position.
        """
        n = len(self._windows)
        lo = self._min_cut_windows if self._min_cut_windows is not None else n // 2
        lo = min(max(lo, 1), n - 1) if n > 1 else 0

        best: tuple[int, int] | None = None
        i = lo
        while i < n:
            if self._probs[i] <= self.speech_threshold:
                j = i
                while j < n and self._probs[j] <= self.speech_threshold:
                    j += 1
                if best is None or j - i >= best[1] - best[0]:
                    best = (i, j)
                i = j
            else:
                i += 1
        if best is not None:
            return max(1, (best[0] + best[1]) // 2)

        tail = np.asarray(self._probs[lo:], dtype=np.float64)
        last_min = len(tail) - 1 - int(np.argmin(tail[::-1]))
        return lo + last_min + 1

    def _split_at_best_pause(self) -> None:
        """Emit the segment up to the best pause; carry the rest into a new segment."""
        cut = self._best_cut_index()
        rest_windows, rest_probs = self._windows[cut:], self._probs[cut:]
        begin = self._has_speech_begin_timestamp if self._has_speech_begin_timestamp is not None else 0.0

        self._windows, self._probs = self._windows[:cut], self._probs[:cut]
        self._emit_segment()

        self._trailing_silence_windows = 0
        if not any(p > self.speech_threshold for p in rest_probs):
            # Silence-only remainder: treat it as non-speech instead of an active segment.
            self._has_speech_begin_timestamp = None
            for window, prob in zip(rest_windows, rest_probs):
                self._handle_non_speech(window, prob)
            # _handle_non_speech accounts one window per call; count the whole remainder.
            self._pending_non_speech_seconds = len(rest_windows) * self._window_seconds
            return

        self._windows, self._probs = rest_windows, rest_probs
        self._has_speech_begin_timestamp = begin + cut * self._window_seconds
        for p in reversed(rest_probs):
            if p > self.speech_threshold:
                break
            self._trailing_silence_windows += 1

    def _end_segment(self) -> None:
        """Handle transition from speech to no speech."""
        self._emit_segment()
        self._has_speech_begin_timestamp = None
        self._trailing_silence_windows = 0
        self._look_back_buffer.clear()

    def _emit_segment(self) -> None:
        """Create AudioSegment and invoke callback."""
        if not self._windows:
            return

        audio_array = np.concatenate(self._windows).astype(np.float32, copy=False)
        start_ts = self._has_speech_begin_timestamp if self._has_speech_begin_timestamp is not None else 0.0
        duration_seconds = len(audio_array) / self.sample_rate

        segment = AudioSegment(
            audio=audio_array,
            start=start_ts,
            duration_seconds=duration_seconds,
        )

        self.vad_model.reset_states()
        self._windows = []
        self._probs = []

        if self.on_segment_complete is not None:
            self.on_segment_complete(segment)

    def flush(self) -> None:
        """Force completion of any in-progress speech segment. Call at end of stream."""
        if self._windows:
            self._end_segment()

    def reset(self) -> None:
        """Reset all state (for stream restart or testing)."""
        self._prev_has_speech = False
        self._windows = []
        self._probs = []
        self._trailing_silence_windows = 0
        self._has_speech_begin_timestamp = None
        self._look_back_buffer.clear()
        self._pending_non_speech_seconds = 0.0
        self.vad_model.reset_states()

    @property
    def is_in_speech(self) -> bool:
        """Current speech/non-speech state."""
        return self._prev_has_speech

    @property
    def current_segment_duration(self) -> float:
        """Duration of current speech segment, including trailing silence."""
        return len(self._windows) * self._window_seconds

    @property
    def current_segment_audio(self) -> npt.NDArray[np.float32]:
        """Audio buffered for the in-progress segment."""
        if not self._windows:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._windows)

    @property
    def pending_non_speech_duration(self) -> float:
        """Duration of accumulated non-speech (for backlog tracking)."""
        return self._pending_non_speech_seconds

    def consume_non_speech(self) -> float:
        """
        Return and clear pending non-speech duration.
        Used by AudioTranscriber for backlog management.

        Returns:
            Pending non-speech duration in seconds
        """
        non_speech_duration = self._pending_non_speech_seconds
        self._pending_non_speech_seconds = 0.0
        return non_speech_duration
