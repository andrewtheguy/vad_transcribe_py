"""
Voice Activity Detection (VAD) processor using Silero VAD model.

This module provides a SpeechDetector class that encapsulates the Silero VAD model
and state machine for detecting speech segments in audio streams.
"""

from collections import deque
from typing import Callable, Optional, Deque

import numpy as np
import numpy.typing as npt
import torch
from silero_vad import load_silero_vad

# Default sample rate for VAD processing
TARGET_SAMPLE_RATE = 16000


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
        duration_seconds: Optional[float] = None,
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

    The detector processes audio in fixed-size windows and maintains a state machine that:
    - Accumulates speech into segments
    - Enforces minimum segment duration to avoid false positives
    - Enforces maximum segment duration to prevent unbounded segments
    - Uses look-back buffer to capture speech onset
    - Tracks non-speech for backlog management

    Attributes:
        sample_rate: Audio sample rate (default: 16000 Hz)
        min_speech_seconds: Minimum valid speech duration (default: 3.0s)
        max_speech_seconds: Maximum segment duration before split (default: 60.0s)
        speech_threshold: Probability threshold for speech detection (default: 0.5)
        min_silence_duration_ms: Minimum silence duration to end speech segment (default: 2000)
        on_segment_complete: Callback when speech segment completes
    """

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        min_speech_seconds: float = 3.0,
        max_speech_seconds: float = 60.0,
        speech_threshold: float = 0.5,
        min_silence_duration_ms: int = 2000,
        look_back_seconds: Optional[float] = None,
        on_segment_complete: Optional[Callable[[AudioSegment], None]] = None,
    ):
        """
        Initialize the speech detector.

        Args:
            sample_rate: Audio sample rate
            min_speech_seconds: Minimum duration to consider as valid speech
            max_speech_seconds: Maximum duration before forcing segment split
            speech_threshold: Probability threshold for speech detection
            min_silence_duration_ms: Minimum silence duration to end speech segment (default: 2000ms)
            look_back_seconds: Amount of non-speech audio to prepend when speech starts
            on_segment_complete: Callback invoked when speech segment completes
        """
        self.sample_rate = sample_rate
        self.min_speech_seconds = min_speech_seconds
        self.max_speech_seconds = max_speech_seconds
        self.speech_threshold = speech_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.on_segment_complete = on_segment_complete

        # Load Silero VAD model
        self.vad_model = load_silero_vad()

        # State machine variables
        self._prev_has_speech = False
        self._speech_section: list = []
        self._has_speech_begin_timestamp: Optional[float] = None
        self._look_back_buffer: Deque[npt.NDArray[np.float32]] = deque()
        self._look_back_buffer_duration = 0.0
        self._pending_non_speech_seconds = 0.0
        self._silence_buffer: list = []  # Buffer for silence windows during speech
        self._accumulated_silence_ms = 0.0  # Track accumulated silence duration

        # Window configuration
        self._window_size_samples = get_window_size_samples(self.sample_rate)
        self._window_seconds = self._window_size_samples / self.sample_rate
        self.look_back_seconds = (
            0.5 if look_back_seconds is None else max(0.0, look_back_seconds)
        )

    def process_window(
        self,
        audio_window: npt.NDArray[np.float32],
        timestamp: float,
    ) -> bool:
        """
        Process a single audio window through VAD.

        Args:
            audio_window: Audio samples (must be window_size_samples length)
            timestamp: Timestamp in seconds from file beginning

        Returns:
            True if speech detected in this window

        Side effects:
            - May call on_segment_complete callback if speech segment ends
            - Updates internal state machine
        """
        # Detect speech in current window
        has_speech = self._detect_speech(audio_window)

        # Get current segment duration
        seconds = len(self._speech_section) / self.sample_rate

        # State machine transitions
        if not self._prev_has_speech:
            if has_speech:
                # Transition: No Speech → Speech Starting
                self._handle_speech_start(audio_window, timestamp)
            else:
                # Still no speech
                self._handle_non_speech(audio_window)
        else:
            # Apply hysteresis constraints
            if seconds > self.max_speech_seconds:
                # Force end of speech if exceeds max duration
                has_speech = False
                self._accumulated_silence_ms = self.min_silence_duration_ms  # Force immediate end
            elif seconds < self.min_speech_seconds and not has_speech:
                # Force continuation if below min duration
                has_speech = True

            if has_speech:
                # Speech continues - flush any buffered silence back into speech
                self._flush_silence_buffer()
                self._handle_speech_continue(audio_window)
            else:
                # Accumulate silence
                self._silence_buffer.extend(audio_window)
                self._accumulated_silence_ms += self._window_seconds * 1000

                if self._accumulated_silence_ms >= self.min_silence_duration_ms:
                    # Enough silence accumulated, end speech segment
                    self._handle_speech_end()
                else:
                    # Not enough silence yet, keep in speech state
                    has_speech = True

        self._prev_has_speech = has_speech
        return has_speech

    def _detect_speech(self, audio: npt.NDArray[np.float32]) -> bool:
        """
        Run VAD model on audio window.

        Args:
            audio: Audio samples (must be window_size_samples length)

        Returns:
            True if speech probability exceeds threshold
        """
        if len(audio) != self._window_size_samples:
            raise ValueError(
                f"Audio length {len(audio)} does not match window size {self._window_size_samples}"
            )

        # Convert to PyTorch tensor
        audio_tensor = torch.from_numpy(audio)
        speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()
        return speech_prob > self.speech_threshold

    def _handle_speech_start(
        self,
        audio_window: npt.NDArray[np.float32],
        timestamp: float,
    ) -> None:
        """Handle transition from no speech to speech."""
        self._has_speech_begin_timestamp = timestamp

        # Add look-back buffer if available
        prepended_duration = 0.0
        if self._look_back_buffer:
            prepended_duration = self._look_back_buffer_duration
            for prev_window in self._look_back_buffer:
                self._speech_section.extend(prev_window)
            self._look_back_buffer.clear()
            self._look_back_buffer_duration = 0.0
            self._pending_non_speech_seconds = 0.0

        # Adjust segment start to account for prepended audio
        if prepended_duration > 0 and self._has_speech_begin_timestamp is not None:
            self._has_speech_begin_timestamp = max(
                0.0, self._has_speech_begin_timestamp - prepended_duration
            )

        # Add current window
        self._speech_section.extend(audio_window)

    def _handle_non_speech(self, audio_window: npt.NDArray[np.float32]) -> None:
        """Handle continued non-speech windows."""
        # Track pending non-speech duration for backlog management
        self._pending_non_speech_seconds = self._window_seconds

        if self.look_back_seconds <= 0.0:
            self._look_back_buffer.clear()
            self._look_back_buffer_duration = 0.0
            return

        # Save into the rolling look-back buffer
        window_copy = np.copy(audio_window)
        self._look_back_buffer.append(window_copy)
        self._look_back_buffer_duration += self._window_seconds

        # Trim buffer to configured duration
        while self._look_back_buffer and self._look_back_buffer_duration - 1e-9 > self.look_back_seconds:
            self._look_back_buffer.popleft()
            self._look_back_buffer_duration -= self._window_seconds

    def _handle_speech_continue(self, audio_window: npt.NDArray[np.float32]) -> None:
        """Handle continued speech."""
        self._speech_section.extend(audio_window)

    def _flush_silence_buffer(self) -> None:
        """Flush buffered silence back into speech section."""
        if self._silence_buffer:
            self._speech_section.extend(self._silence_buffer)
            self._silence_buffer = []
        self._accumulated_silence_ms = 0.0

    def _handle_speech_end(self) -> None:
        """Handle transition from speech to no speech."""
        # Add buffered silence to segment
        self._speech_section.extend(self._silence_buffer)
        self._silence_buffer = []
        self._accumulated_silence_ms = 0.0

        # Create and emit segment
        self._emit_segment()

        # Reset state (note: _emit_segment() clears _speech_section)
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

        # Reset VAD model state
        self.vad_model.reset_states()

        # Clear speech section state
        self._speech_section = []

        # Invoke callback
        if self.on_segment_complete is not None:
            self.on_segment_complete(segment)

    def flush(self) -> None:
        """
        Force completion of any in-progress speech segment.
        Call at end of stream.
        """
        # Include any buffered silence in the final segment
        if self._silence_buffer:
            self._speech_section.extend(self._silence_buffer)
            self._silence_buffer = []
            self._accumulated_silence_ms = 0.0

        if len(self._speech_section) > 0:
            self._emit_segment()
            # Note: _emit_segment() clears _speech_section
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
        """Duration of current speech segment (if any)."""
        return len(self._speech_section) / self.sample_rate

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
