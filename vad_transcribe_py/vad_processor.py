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
DEFAULT_HARD_LIMIT_SECONDS = 60 * 60
DEFAULT_SPEECH_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_DURATION_MS = 2000
DEFAULT_LOOK_BACK_SECONDS = 0.5
ADAPTIVE_MIN_SILENCE_MS = 32  # reduced silence threshold when over soft limit (~one window)

# Whisper backend limits
WHISPER_HARD_LIMIT_SECONDS = 60
WHISPER_SOFT_LIMIT_SECONDS = 6.0

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
    Detects speech segments in audio using Silero VAD with hysteresis-based state machine.

    Processes audio in fixed-size windows, applies an adaptive soft limit to
    encourage earlier segmentation, and enforces a hard limit by force-splitting
    segments at the sample level.
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

        # State machine variables
        self._prev_has_speech = False
        self._speech_section: list[float] = []
        self._has_speech_begin_timestamp: float | None = None
        self._look_back_buffer: deque[npt.NDArray[np.float32]] = deque()
        self._look_back_buffer_duration = 0.0
        self._pending_non_speech_seconds = 0.0
        self._silence_buffer: list[float] = []
        self._accumulated_silence_ms = 0.0

        # Window configuration
        self._window_size_samples = get_window_size_samples(self.sample_rate)
        self._window_seconds = self._window_size_samples / self.sample_rate
        self.look_back_seconds = max(0.0, look_back_seconds)

        # Hard limit in samples for precise splitting
        self._hard_max_speech_samples = int(round(self.hard_limit_seconds * self.sample_rate))

    def process_window(
        self,
        audio_window: npt.NDArray[np.float32],
        timestamp: float,
    ) -> bool:
        """Process a single audio window through VAD."""
        has_speech = self._detect_speech(audio_window)
        seconds = self.current_segment_duration

        if not self._prev_has_speech:
            if has_speech:
                has_speech = self._handle_speech_start(audio_window, timestamp)
            else:
                self._handle_non_speech(audio_window)
        else:
            effective_min_silence_ms = self._get_effective_min_silence_ms(seconds)

            if seconds < self.min_speech_seconds and not has_speech:
                has_speech = True

            if has_speech:
                self._flush_silence_buffer()
                has_speech = self._append_speech_chunk(audio_window)
            else:
                has_speech = self._append_silence_chunk(
                    audio_window, effective_min_silence_ms
                )

        self._prev_has_speech = has_speech
        return has_speech

    def _get_effective_min_silence_ms(self, current_duration_seconds: float) -> float:
        if (
            self.soft_limit_seconds is not None
            and self.soft_limit_seconds < self.hard_limit_seconds
            and current_duration_seconds > self.soft_limit_seconds
            and self.min_silence_duration_ms > 0
        ):
            return ADAPTIVE_MIN_SILENCE_MS
        return self.min_silence_duration_ms

    def _detect_speech(self, audio: npt.NDArray[np.float32]) -> bool:
        if len(audio) != self._window_size_samples:
            raise ValueError(
                f"Audio length {len(audio)} does not match window size {self._window_size_samples}"
            )

        audio_tensor = torch.from_numpy(audio)
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        return speech_prob > self.speech_threshold

    def _handle_speech_start(
        self,
        audio_window: npt.NDArray[np.float32],
        timestamp: float,
    ) -> bool:
        """Handle transition from no speech to speech."""
        self._has_speech_begin_timestamp = timestamp

        prepended_duration = 0.0
        if self._look_back_buffer:
            prepended_duration = self._look_back_buffer_duration
            for prev_window in self._look_back_buffer:
                self._speech_section.extend(prev_window)
            self._look_back_buffer.clear()
            self._look_back_buffer_duration = 0.0
            self._pending_non_speech_seconds = 0.0

        if prepended_duration > 0:
            self._has_speech_begin_timestamp = max(
                0.0, self._has_speech_begin_timestamp - prepended_duration
            )

        return self._append_speech_chunk(audio_window)

    def _handle_non_speech(self, audio_window: npt.NDArray[np.float32]) -> None:
        """Handle continued non-speech windows."""
        self._pending_non_speech_seconds = self._window_seconds

        if self.look_back_seconds <= 0.0:
            self._look_back_buffer.clear()
            self._look_back_buffer_duration = 0.0
            return

        window_copy = np.copy(audio_window)
        self._look_back_buffer.append(window_copy)
        self._look_back_buffer_duration += self._window_seconds

        while self._look_back_buffer and self._look_back_buffer_duration - 1e-9 > self.look_back_seconds:
            self._look_back_buffer.popleft()
            self._look_back_buffer_duration -= self._window_seconds

    def _flush_silence_buffer(self) -> None:
        """Flush buffered silence back into speech section."""
        if self._silence_buffer:
            self._speech_section.extend(self._silence_buffer)
            self._silence_buffer = []
        self._accumulated_silence_ms = 0.0

    def _append_speech_chunk(self, audio_chunk: npt.NDArray[np.float32]) -> bool:
        """Append speech audio, force-emitting segments if the hard cap is reached."""
        chunk = audio_chunk

        while len(chunk) > 0:
            remaining = self._remaining_hard_cap_samples()
            take = min(len(chunk), remaining)
            if take > 0:
                self._speech_section.extend(chunk[:take])
                chunk = chunk[take:]

            if self._remaining_hard_cap_samples() != 0:
                return True

            split_ts = self._segment_end_timestamp()
            self._force_emit_active_segment()

            if len(chunk) == 0:
                return False

            self._has_speech_begin_timestamp = split_ts

        return len(self._speech_section) > 0

    def _append_silence_chunk(
        self,
        audio_chunk: npt.NDArray[np.float32],
        effective_min_silence_ms: float,
    ) -> bool:
        """Append silence while in speech, honoring both hard and soft caps."""
        chunk = audio_chunk

        while len(chunk) > 0:
            remaining = self._remaining_hard_cap_samples()
            take = min(len(chunk), remaining)

            if take > 0:
                self._silence_buffer.extend(chunk[:take])
                self._accumulated_silence_ms += take / self.sample_rate * 1000
                chunk = chunk[take:]

            if self._remaining_hard_cap_samples() == 0:
                self._handle_speech_end()
                if len(chunk) > 0:
                    self._handle_non_speech(chunk)
                return False

            if self._accumulated_silence_ms >= effective_min_silence_ms:
                self._handle_speech_end()
                if len(chunk) > 0:
                    self._handle_non_speech(chunk)
                return False

        return True

    def _current_segment_samples(self) -> int:
        return len(self._speech_section) + len(self._silence_buffer)

    def _remaining_hard_cap_samples(self) -> int:
        return max(0, self._hard_max_speech_samples - self._current_segment_samples())

    def _segment_end_timestamp(self) -> float:
        start_ts = (
            self._has_speech_begin_timestamp
            if self._has_speech_begin_timestamp is not None
            else 0.0
        )
        return start_ts + self._current_segment_samples() / self.sample_rate

    def _force_emit_active_segment(self) -> None:
        """Force-emit the current segment at the hard limit."""
        self._silence_buffer = []
        self._accumulated_silence_ms = 0.0
        self._emit_segment()
        self._has_speech_begin_timestamp = None
        self._look_back_buffer.clear()
        self._look_back_buffer_duration = 0.0

    def _handle_speech_end(self) -> None:
        """Handle transition from speech to no speech."""
        self._speech_section.extend(self._silence_buffer)
        self._silence_buffer = []
        self._accumulated_silence_ms = 0.0
        self._emit_segment()
        self._has_speech_begin_timestamp = None
        self._look_back_buffer.clear()
        self._look_back_buffer_duration = 0.0

    def _emit_segment(self) -> None:
        """Create AudioSegment and invoke callback."""
        if not self._speech_section:
            return

        audio_array = np.asarray(self._speech_section)
        start_ts = self._has_speech_begin_timestamp if self._has_speech_begin_timestamp is not None else 0.0
        duration_seconds = len(audio_array) / self.sample_rate

        segment = AudioSegment(
            audio=audio_array,
            start=start_ts,
            duration_seconds=duration_seconds,
        )

        self.vad_model.reset_states()
        self._speech_section = []

        if self.on_segment_complete is not None:
            self.on_segment_complete(segment)

    def flush(self) -> None:
        """Force completion of any in-progress speech segment. Call at end of stream."""
        if self._silence_buffer:
            self._speech_section.extend(self._silence_buffer)
            self._silence_buffer = []
            self._accumulated_silence_ms = 0.0

        if len(self._speech_section) > 0:
            self._emit_segment()
            self._has_speech_begin_timestamp = None
            self._look_back_buffer.clear()
            self._look_back_buffer_duration = 0.0

    def reset(self) -> None:
        """Reset all state (for stream restart or testing)."""
        self._prev_has_speech = False
        self._speech_section = []
        self._has_speech_begin_timestamp = None
        self._look_back_buffer.clear()
        self._look_back_buffer_duration = 0.0
        self._pending_non_speech_seconds = 0.0
        self._silence_buffer = []
        self._accumulated_silence_ms = 0.0
        self.vad_model.reset_states()

    @property
    def is_in_speech(self) -> bool:
        """Current speech/non-speech state."""
        return self._prev_has_speech

    @property
    def current_segment_duration(self) -> float:
        """Duration of current speech segment including silence buffer."""
        return self._current_segment_samples() / self.sample_rate

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
