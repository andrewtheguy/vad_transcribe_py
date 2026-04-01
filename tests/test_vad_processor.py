"""
Unit tests for SpeechDetector class in vad_processor module.

Includes comprehensive tests for:
- Basic functionality
- Min/max duration enforcement
- Look-back buffer (prev_slice) inclusion
- Final window inclusion
- Edge cases and boundary conditions
"""

import numpy as np
import pytest
from unittest.mock import patch

from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment, get_window_size_samples


NON_SPEECH_LEVEL = 1e-3


def make_non_speech_window(length: int = 512, level: float = NON_SPEECH_LEVEL) -> np.ndarray:
    """Create a low-amplitude window that still represents non-speech."""
    return np.full(length, level, dtype=np.float32)


class TestGetWindowSizeSamples:
    """Tests for get_window_size_samples function."""

    def test_default_16khz(self):
        """Test default sample rate returns 512 samples."""
        assert get_window_size_samples() == 512
        assert get_window_size_samples(16000) == 512

    def test_8khz(self):
        """Test 8kHz sample rate returns 256 samples."""
        assert get_window_size_samples(8000) == 256

    def test_other_sample_rates(self):
        """Test other sample rates return 256 samples."""
        assert get_window_size_samples(44100) == 256
        assert get_window_size_samples(48000) == 256


class TestAudioSegment:
    """Tests for AudioSegment class."""

    def test_creation_basic(self):
        """Test basic AudioSegment creation."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(start=1.0, audio=audio)

        assert segment.start == 1.0
        assert np.array_equal(segment.audio, audio)
        assert segment.duration_seconds is None

    def test_creation_with_all_fields(self):
        """Test AudioSegment creation with all fields."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(
            start=2.5,
            audio=audio,
            duration_seconds=3.0,
        )

        assert segment.start == 2.5
        assert np.array_equal(segment.audio, audio)
        assert segment.duration_seconds == 3.0

    def test_repr(self):
        """Test AudioSegment string representation."""
        audio = np.array([0.1, 0.2], dtype=np.float32)
        segment = AudioSegment(start=1.0, audio=audio, duration_seconds=0.5)

        repr_str = repr(segment)
        assert "AudioSegment" in repr_str
        assert "start=1.0" in repr_str
        assert "duration=0.5" in repr_str


class TestSpeechDetector:
    """Tests for SpeechDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a basic SpeechDetector instance."""
        return SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=3.0,
            soft_limit_seconds=60.0,
            speech_threshold=0.5,
        )

    @pytest.fixture
    def mock_segment_callback(self):
        """Create a mock callback that stores called segments."""
        segments = []

        def callback(segment: AudioSegment):
            segments.append(segment)

        callback.segments = segments
        return callback

    def _create_audio_window(self, detector, value=0.5):
        """Create an audio window of appropriate size filled with given value."""
        return np.full(detector._window_size_samples, value, dtype=np.float32)

    def _create_non_speech_window(self, detector):
        """Create a low-level non-speech window to avoid true zero samples."""
        return self._create_audio_window(detector, value=NON_SPEECH_LEVEL)

    def test_initialization(self, detector):
        """Test SpeechDetector initialization."""
        assert detector.sample_rate == 16000
        assert detector.min_speech_seconds == 3.0
        assert detector.soft_limit_seconds == 60.0
        assert detector.speech_threshold == 0.5
        assert detector.is_in_speech is False
        assert detector.current_segment_duration == 0.0
        assert detector.pending_non_speech_duration == 0.0
        assert detector.look_back_seconds == pytest.approx(0.5)

    def test_process_window_wrong_size(self, detector):
        """Test that wrong window size raises ValueError."""
        wrong_size_audio = np.array([0.1, 0.2], dtype=np.float32)

        with pytest.raises(ValueError, match="Audio length.*does not match window size"):
            detector.process_window(wrong_size_audio, 0.0)

    def test_single_non_speech_window(self, detector, mock_segment_callback):
        """Test processing a single non-speech window."""
        detector.on_segment_complete = mock_segment_callback

        # Create low-level non-speech (avoids true zero-signal gaps)
        audio = self._create_non_speech_window(detector)
        has_speech = detector.process_window(audio, 0.0)

        assert has_speech is False
        assert detector.is_in_speech is False
        assert len(mock_segment_callback.segments) == 0

    def test_non_speech_to_speech_transition(self, detector, mock_segment_callback):
        """Test transition from non-speech to speech."""
        detector.on_segment_complete = mock_segment_callback

        # First window: non-speech
        non_speech = self._create_non_speech_window(detector)
        detector.process_window(non_speech, 0.0)
        assert detector.is_in_speech is False

        # Second window: speech (high amplitude triggers VAD)
        # Note: Actual VAD behavior depends on the model, this tests the state machine
        # We can't easily control VAD output without mocking, so we test state transitions
        speech = self._create_audio_window(detector, value=0.5)
        detector.process_window(speech, 0.032)  # 0.032s = 512 samples @ 16kHz

    def test_reset(self, detector):
        """Test reset clears all state."""
        # Process some audio to create state
        audio = self._create_audio_window(detector, value=0.5)
        detector.process_window(audio, 0.0)

        # Reset
        detector.reset()

        # Verify all state is cleared
        assert detector.is_in_speech is False
        assert detector.current_segment_duration == 0.0
        assert detector.pending_non_speech_duration == 0.0
        assert detector._prev_has_speech is False
        assert len(detector._speech_section) == 0
        assert detector._has_speech_begin_timestamp is None
        assert len(detector._look_back_buffer) == 0

    def test_flush_with_no_speech(self, detector, mock_segment_callback):
        """Test flush with no accumulated speech does nothing."""
        detector.on_segment_complete = mock_segment_callback

        detector.flush()

        assert len(mock_segment_callback.segments) == 0

    def test_consume_non_speech(self, detector):
        """Test consume_non_speech returns and clears pending non-speech."""
        # Manually set pending non-speech
        detector._pending_non_speech_seconds = 0.5

        consumed = detector.consume_non_speech()

        assert consumed == 0.5
        assert detector.pending_non_speech_duration == 0.0

    def test_consume_non_speech_when_zero(self, detector):
        """Test consume_non_speech when there's no pending non-speech."""
        consumed = detector.consume_non_speech()

        assert consumed == 0.0
        assert detector.pending_non_speech_duration == 0.0

    def test_properties(self, detector):
        """Test property accessors."""
        # Initial state
        assert detector.is_in_speech is False
        assert detector.current_segment_duration == 0.0
        assert detector.pending_non_speech_duration == 0.0

        # Simulate some state
        detector._prev_has_speech = True
        detector._speech_section = [0.1] * 16000  # 1 second of audio
        detector._pending_non_speech_seconds = 0.25

        assert detector.is_in_speech is True
        assert detector.current_segment_duration == 1.0
        assert detector.pending_non_speech_duration == 0.25

    def test_custom_parameters(self):
        """Test SpeechDetector with custom parameters."""
        detector = SpeechDetector(
            sample_rate=8000,
            min_speech_seconds=1.0,
            soft_limit_seconds=30.0,
            speech_threshold=0.7,
            look_back_seconds=0.5,
        )

        assert detector.sample_rate == 8000
        assert detector.min_speech_seconds == 1.0
        assert detector.soft_limit_seconds == 30.0
        assert detector.speech_threshold == 0.7
        assert detector.look_back_seconds == 0.5
        assert detector._window_size_samples == 256  # 8kHz uses 256 samples

    def test_segment_callback_invocation(self, mock_segment_callback):
        """Test that callback is invoked when provided."""
        detector = SpeechDetector(on_segment_complete=mock_segment_callback)

        # Manually trigger segment emission to test callback
        detector._speech_section = [0.1] * 16000  # 1 second
        detector._has_speech_begin_timestamp = 1.0
        detector._emit_segment()

        assert len(mock_segment_callback.segments) == 1
        segment = mock_segment_callback.segments[0]
        assert segment.start == 1.0
        assert segment.duration_seconds == 1.0
        assert len(segment.audio) == 16000

    def test_no_callback(self):
        """Test SpeechDetector works without callback."""
        detector = SpeechDetector(on_segment_complete=None)

        # Should not raise error even without callback
        detector._speech_section = [0.1] * 16000
        detector._has_speech_begin_timestamp = 1.0
        detector._emit_segment()  # Should complete without error

    def test_multiple_flushes(self, detector, mock_segment_callback):
        """Test that multiple flushes don't cause issues."""
        detector.on_segment_complete = mock_segment_callback

        detector.flush()
        detector.flush()
        detector.flush()

        assert len(mock_segment_callback.segments) == 0

    def test_window_seconds_calculation(self, detector):
        """Test that window seconds is correctly calculated."""
        expected_window_seconds = detector._window_size_samples / detector.sample_rate
        assert detector._window_seconds == expected_window_seconds
        assert detector._window_seconds == pytest.approx(0.032, rel=0.001)  # 512/16000

    def test_speech_detection_internal(self, detector):
        """Test internal speech detection mechanism."""
        # Create audio window
        audio = self._create_audio_window(detector, value=0.5)

        # This tests that _detect_speech runs without error
        # Actual VAD results depend on the Silero model
        result = detector._detect_speech(audio)

        # Result should be boolean
        assert isinstance(result, bool)


class TestSpeechDetectorIntegration:
    """Integration tests for SpeechDetector with realistic scenarios."""

    def test_short_non_speech_periods(self):
        """Test that short non-speech periods don't break speech segments."""
        segments = []

        def callback(segment):
            segments.append(segment)

        detector = SpeechDetector(
            min_speech_seconds=0.1,  # Very short for testing
            soft_limit_seconds=10.0,
            on_segment_complete=callback,
        )

        # Process several windows
        # Note: Actual VAD behavior is hard to test without mocking
        # This tests the framework works correctly
        window = np.full(512, 0.5, dtype=np.float32)
        for i in range(10):
            timestamp = i * 0.032
            detector.process_window(window, timestamp)

        # Flush any remaining speech
        detector.flush()

        # We can't predict exact VAD behavior, but system should handle it
        assert isinstance(segments, list)

    def test_state_machine_reset_between_segments(self):
        """Test that state is properly reset between speech segments."""
        segments = []

        def callback(segment):
            segments.append(segment)

        detector = SpeechDetector(on_segment_complete=callback)

        # Manually create a segment
        detector._speech_section = [0.1] * 16000
        detector._has_speech_begin_timestamp = 1.0
        detector._emit_segment()

        # Verify state was reset
        assert len(detector._speech_section) == 0
        assert len(segments) == 1

        # Create another segment
        detector._speech_section = [0.2] * 8000
        detector._has_speech_begin_timestamp = 3.0
        detector._emit_segment()

        # Verify second segment was created
        assert len(segments) == 2
        assert segments[0].audio[0] == pytest.approx(0.1, abs=0.001)
        assert segments[1].audio[0] == pytest.approx(0.2, abs=0.001)


class TestMinDurationEnforcement:
    """Test minimum speech duration hysteresis enforcement."""

    def test_min_duration_prevents_early_cutoff(self):
        """Test that speech below min duration continues even if VAD detects non-speech."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.1,  # 0.1s = ~3 windows
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        # Mock the _detect_speech to control VAD output
        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Window 1: non-speech
            mock_vad.return_value = False
            detector.process_window(window, 0.0)
            assert detector.is_in_speech is False

            # Window 2: speech starts
            mock_vad.return_value = True
            detector.process_window(window, 0.032)
            assert detector.is_in_speech is True
            assert len(detector._speech_section) == 1024  # 512 (prev_slice) + 512 (current)

            # Window 3: VAD says no speech, but min duration forces continuation
            mock_vad.return_value = False
            detector.process_window(window, 0.064)
            assert detector.is_in_speech is True  # Forced to continue
            assert detector.current_segment_duration < 0.1  # Still below min

            # Window 4: Still below min, should continue
            mock_vad.return_value = False
            detector.process_window(window, 0.096)
            assert detector.is_in_speech is True  # Still forced to continue

            # Window 5: Now above min duration, can end
            mock_vad.return_value = False
            detector.process_window(window, 0.128)
            assert detector.is_in_speech is False  # Can end now
            assert len(segments) == 1

            # Verify segment includes all windows
            segment = segments[0]
            assert len(segment.audio) == 512 * 5  # prev_slice + 4 speech windows

    def test_exactly_min_duration_boundary(self):
        """Test behavior at exactly min duration boundary."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.064,  # Exactly 2 windows
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Start speech
            mock_vad.return_value = True
            detector.process_window(window, 0.0)
            detector.process_window(window, 0.032)

            # At exactly min duration, VAD no speech should end it
            mock_vad.return_value = False
            detector.process_window(window, 0.064)

            assert len(segments) == 1
            assert segments[0].duration_seconds == pytest.approx(0.096, abs=0.001)  # 3 windows


class TestMinSilenceDuration:
    """Test minimum silence duration enforcement for ending speech segments."""

    def test_short_silence_does_not_end_speech(self):
        """Silence shorter than min_silence_duration_ms should not end speech."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        # 100ms = ~3 windows at 32ms each
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=100,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size)

            # Start with speech
            mock_vad.return_value = True
            detector.process_window(speech_window, 0.0)
            detector.process_window(speech_window, window_seconds)

            # Short silence (1 window = ~32ms < 100ms)
            mock_vad.return_value = False
            detector.process_window(silence_window, window_seconds * 2)

            # Should still be in speech, no segment emitted
            assert detector.is_in_speech is True
            assert len(segments) == 0

            # Speech resumes
            mock_vad.return_value = True
            detector.process_window(speech_window, window_seconds * 3)

            # Still in speech, silence was absorbed
            assert detector.is_in_speech is True
            assert len(segments) == 0

    def test_silence_at_threshold_ends_speech(self):
        """Silence reaching min_silence_duration_ms should end speech."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        # 64ms = 2 windows at 32ms each
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=64,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size)

            ts = 0.0
            # Start with speech
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # First silence window (~32ms < 64ms)
            mock_vad.return_value = False
            detector.process_window(silence_window, ts)
            ts += window_seconds
            assert detector.is_in_speech is True  # Not yet ended
            assert len(segments) == 0

            # Second silence window (~64ms >= 64ms)
            detector.process_window(silence_window, ts)
            assert detector.is_in_speech is False  # Now ended
            assert len(segments) == 1

    def test_speech_interrupts_silence_accumulation(self):
        """Speech detected during silence accumulation should reset silence counter."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        # 100ms = ~3 windows
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=100,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size)

            ts = 0.0
            # Start with speech
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # Two silence windows (~64ms)
            mock_vad.return_value = False
            detector.process_window(silence_window, ts)
            ts += window_seconds
            detector.process_window(silence_window, ts)
            ts += window_seconds

            assert detector.is_in_speech is True  # Still in speech
            assert len(segments) == 0

            # Speech interrupts - should reset silence counter
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds

            assert detector.is_in_speech is True
            assert len(segments) == 0

            # Two more silence windows (~64ms) - should not end since counter reset
            mock_vad.return_value = False
            detector.process_window(silence_window, ts)
            ts += window_seconds
            detector.process_window(silence_window, ts)
            ts += window_seconds

            assert detector.is_in_speech is True  # Still in speech, need more silence
            assert len(segments) == 0

            # Third silence window (~96ms) - still not enough
            detector.process_window(silence_window, ts)
            ts += window_seconds

            assert detector.is_in_speech is True
            assert len(segments) == 0

            # Fourth silence window (~128ms > 100ms) - now ends
            detector.process_window(silence_window, ts)

            assert detector.is_in_speech is False
            assert len(segments) == 1

    def test_silence_buffer_included_in_segment(self):
        """Buffered silence should be included in the final segment."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=64,  # 2 windows
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size, level=0.1)

            ts = 0.0
            # Two speech windows
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # Two silence windows to trigger end
            mock_vad.return_value = False
            detector.process_window(silence_window, ts)
            ts += window_seconds
            detector.process_window(silence_window, ts)

            assert len(segments) == 1
            # Segment should include: 2 speech + 2 silence = 4 windows
            assert len(segments[0].audio) == window_size * 4
            # Last portion should be the silence
            assert np.allclose(segments[0].audio[-window_size:], silence_window)

    def test_interrupted_silence_flushed_to_speech(self):
        """When speech interrupts silence, buffered silence should be in speech section."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=100,  # ~3 windows
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size, level=0.1)

            ts = 0.0
            # Two speech windows
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # One silence window (not enough to end)
            mock_vad.return_value = False
            detector.process_window(silence_window, ts)
            ts += window_seconds

            # Speech resumes - silence should be flushed to speech section
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # Check internal state - speech section should have all 4 windows
            assert len(detector._speech_section) == window_size * 4

            # End with enough silence
            mock_vad.return_value = False
            for _ in range(4):  # 4 windows = ~128ms > 100ms
                detector.process_window(silence_window, ts)
                ts += window_seconds

            assert len(segments) == 1
            # 2 speech + 1 silence (flushed) + 1 speech + 4 silence = 8 windows
            assert len(segments[0].audio) == window_size * 8

    def test_flush_includes_buffered_silence(self):
        """Flush should include any buffered silence in the final segment."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=200,  # High threshold
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size, level=0.1)

            ts = 0.0
            # Two speech windows
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # One silence window (not enough to end)
            mock_vad.return_value = False
            detector.process_window(silence_window, ts)

            assert len(segments) == 0  # Not ended yet

            # Flush should emit segment including buffered silence
            detector.flush()

            assert len(segments) == 1
            # 2 speech + 1 silence = 3 windows
            assert len(segments[0].audio) == window_size * 3

    def test_multiple_short_silences_do_not_accumulate(self):
        """Multiple short silences separated by speech should each reset the counter."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=100,  # ~3 windows
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(window_size, dtype=np.float32) * 0.8
            silence_window = make_non_speech_window(length=window_size)

            ts = 0.0
            # Start speech
            mock_vad.return_value = True
            detector.process_window(speech_window, ts)
            ts += window_seconds

            # Pattern: 2 silence, speech, 2 silence, speech, 2 silence, speech
            for _ in range(3):
                mock_vad.return_value = False
                detector.process_window(silence_window, ts)
                ts += window_seconds
                detector.process_window(silence_window, ts)
                ts += window_seconds

                mock_vad.return_value = True
                detector.process_window(speech_window, ts)
                ts += window_seconds

            # Should still be in speech - no silence period exceeded threshold
            assert detector.is_in_speech is True
            assert len(segments) == 0

            # Now end with enough silence
            mock_vad.return_value = False
            for _ in range(4):
                detector.process_window(silence_window, ts)
                ts += window_seconds

            assert len(segments) == 1


class TestMaxDurationEnforcement:
    """Test maximum speech duration enforcement."""

    def test_max_duration_forces_split_with_silence(self):
        """Test that speech exceeding max duration ends at next silence boundary."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=0.2,  # Only 0.2s max (~6 windows)
            min_silence_duration_ms=100,  # Normal silence threshold
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Continuous speech for windows exceeding max duration
            mock_vad.return_value = True
            for i in range(8):
                detector.process_window(window, i * 0.032)

            # No split yet - adaptive mode waits for silence
            assert len(segments) == 0

            # Now a silence window triggers immediate split (adaptive threshold)
            mock_vad.return_value = False
            detector.process_window(window, 8 * 0.032)

            # Should have split at the silence boundary
            assert len(segments) == 1
            # Segment includes speech + the silence window
            assert segments[0].duration_seconds >= 0.2

    def test_max_duration_immediate_split_when_no_silence_threshold(self):
        """Test that min_silence_duration_ms=0 causes immediate split at max."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=0.2,  # Only 0.2s max (~6 windows)
            min_silence_duration_ms=0,  # Immediate end on any silence
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Continuous speech for many windows
            mock_vad.return_value = True
            for i in range(8):
                detector.process_window(window, i * 0.032)

            # With min_silence_duration_ms=0, still needs a silence to end
            assert len(segments) == 0

            # One silence window ends it immediately
            mock_vad.return_value = False
            detector.process_window(window, 8 * 0.032)

            assert len(segments) == 1

    def test_exactly_max_duration_boundary_with_silence(self):
        """Test behavior at exactly max duration boundary with silence."""
        segments = []
        # 0.192s = 6 windows of 512 samples @ 16kHz
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=0.192,  # Exactly 6 windows
            min_silence_duration_ms=100,  # Normal threshold
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()
            mock_vad.return_value = True

            # Process windows up to and beyond max duration
            for i in range(7):
                detector.process_window(window, i * 0.032)

            # Still in speech - adaptive mode
            assert len(segments) == 0

            # Silence triggers split
            mock_vad.return_value = False
            detector.process_window(window, 7 * 0.032)

            assert len(segments) >= 1
            # Segment should be around max duration
            assert segments[0].duration_seconds >= 0.192

    def test_adaptive_threshold_only_when_over_max_speech(self):
        """
        Test that ADAPTIVE_MIN_SILENCE_MS is only used when over soft_limit_seconds.

        Under soft_limit_seconds, the normal min_silence_duration_ms should apply.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 10,  # 10 windows (~320ms)
            min_silence_duration_ms=200,  # High threshold (~6 windows)
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Start speech - 3 windows (under max)
            mock_vad.return_value = True
            for i in range(3):
                detector.process_window(window, i * window_seconds)

            # One silence window - under 200ms threshold
            mock_vad.return_value = False
            detector.process_window(window, 3 * window_seconds)

            # Should NOT end (under max_speech, under min_silence_duration_ms)
            assert len(segments) == 0
            assert detector.is_in_speech  # Still in speech state

            # Continue with speech
            mock_vad.return_value = True
            detector.process_window(window, 4 * window_seconds)

            # Now flush to verify segment was extended
            detector.flush()
            assert len(segments) == 1
            # Should include: 3 speech + 1 silence + 1 speech = 5 windows
            assert len(segments[0].audio) == window_size * 5

    def test_adaptive_threshold_activates_after_max_speech(self):
        """
        Test that a single silence window ends segment when over soft_limit_seconds.

        With min_silence_duration_ms=2000 (default), normally 62+ windows of silence
        would be needed. But after exceeding soft_limit_seconds, ADAPTIVE_MIN_SILENCE_MS
        (32ms = 1 window) should trigger the end.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 3,  # 3 windows (~96ms)
            min_silence_duration_ms=2000,  # Very high threshold
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # 5 speech windows (exceeds max of 3 windows)
            mock_vad.return_value = True
            for i in range(5):
                detector.process_window(window, i * window_seconds)

            # Over soft_limit_seconds, adaptive mode is active
            assert len(segments) == 0

            # One silence window should trigger end (due to adaptive threshold)
            mock_vad.return_value = False
            detector.process_window(window, 5 * window_seconds)

            # Should have split at the silence boundary
            assert len(segments) == 1
            # 5 speech + 1 silence = 6 windows
            assert len(segments[0].audio) == window_size * 6

    def test_normal_silence_threshold_under_max_speech(self):
        """
        Test that under soft_limit_seconds, min_silence_duration_ms is respected.

        Multiple short silences should not end the segment if they don't meet
        the min_silence_duration_ms threshold.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        # Each window is ~32ms, so 4 windows = ~128ms
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 20,  # High max to stay under
            min_silence_duration_ms=128,  # ~4 windows of silence needed
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Start with speech
            mock_vad.return_value = True
            for i in range(3):
                detector.process_window(window, i * window_seconds)

            # 3 silence windows (under 128ms threshold)
            mock_vad.return_value = False
            for i in range(3, 6):
                detector.process_window(window, i * window_seconds)

            # Should NOT end yet (under threshold)
            assert len(segments) == 0

            # Speech resumes, silence buffer should be flushed to segment
            mock_vad.return_value = True
            detector.process_window(window, 6 * window_seconds)

            # 4th silence window would meet threshold
            mock_vad.return_value = False
            for i in range(7, 11):
                detector.process_window(window, i * window_seconds)

            # Should end now (4 windows >= 128ms)
            assert len(segments) == 1

    def test_no_adaptive_threshold_when_min_silence_is_zero(self):
        """
        Test that adaptive threshold doesn't apply when min_silence_duration_ms=0.

        With min_silence_duration_ms=0, even a single silence window immediately
        ends the segment, regardless of whether we're over soft_limit_seconds.
        This is the same behavior before and after soft_limit_seconds.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 10,  # High max
            min_silence_duration_ms=0,  # Immediate end on silence
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Start speech - 3 windows (under max)
            mock_vad.return_value = True
            for i in range(3):
                detector.process_window(window, i * window_seconds)

            # One silence window should end immediately (not wait for threshold)
            mock_vad.return_value = False
            detector.process_window(window, 3 * window_seconds)

            # Should have ended immediately
            assert len(segments) == 1
            # 3 speech + 1 silence = 4 windows
            assert len(segments[0].audio) == window_size * 4

    def test_adaptive_vs_non_adaptive_comparison(self):
        """
        Compare behavior with min_silence_duration_ms > 0 vs = 0 when over max.

        This test shows the difference between adaptive (>0) and non-adaptive (=0) modes
        when the segment exceeds soft_limit_seconds.
        """
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate

        # Test with min_silence_duration_ms=0 (non-adaptive)
        segments_non_adaptive = []
        detector_non_adaptive = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 3,  # 3 windows
            min_silence_duration_ms=0,
            on_segment_complete=lambda s: segments_non_adaptive.append(s),
        )

        # Test with min_silence_duration_ms=2000 (adaptive)
        segments_adaptive = []
        detector_adaptive = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 3,  # Same 3 windows
            min_silence_duration_ms=2000,  # High threshold
            on_segment_complete=lambda s: segments_adaptive.append(s),
        )

        with patch.object(detector_non_adaptive, '_detect_speech') as mock_na, \
             patch.object(detector_adaptive, '_detect_speech') as mock_a:
            window = make_non_speech_window()

            # Both: 5 speech windows (over max of 3)
            mock_na.return_value = True
            mock_a.return_value = True
            for i in range(5):
                detector_non_adaptive.process_window(window, i * window_seconds)
                detector_adaptive.process_window(window, i * window_seconds)

            # Neither has emitted yet (continuous speech)
            assert len(segments_non_adaptive) == 0
            assert len(segments_adaptive) == 0

            # One silence window
            mock_na.return_value = False
            mock_a.return_value = False
            detector_non_adaptive.process_window(window, 5 * window_seconds)
            detector_adaptive.process_window(window, 5 * window_seconds)

            # Both should have ended with 1 silence (adaptive kicks in for high threshold)
            assert len(segments_non_adaptive) == 1
            assert len(segments_adaptive) == 1

            # Both should have the same content (5 speech + 1 silence)
            assert len(segments_non_adaptive[0].audio) == window_size * 6
            assert len(segments_adaptive[0].audio) == window_size * 6


class TestLookBackBuffer:
    """Test look-back buffer (prev_slice) inclusion at speech start."""

    def test_prev_slice_included_at_speech_start(self):
        """Test that the previous non-speech window is included when speech starts."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            # Create distinguishable windows
            non_speech_window = make_non_speech_window(level=0.1)
            speech_window = np.ones(512, dtype=np.float32) * 0.9

            # Window 1: non-speech (should be saved as prev_slice)
            mock_vad.return_value = False
            detector.process_window(non_speech_window, 0.0)
            assert len(detector._look_back_buffer) == 1
            assert np.array_equal(detector._look_back_buffer[0], non_speech_window)

            # Window 2: speech starts (should include prev_slice)
            mock_vad.return_value = True
            detector.process_window(speech_window, 0.032)
            assert len(detector._look_back_buffer) == 0  # Should be consumed
            # Speech section should have prev_slice + current
            assert len(detector._speech_section) == 1024  # 512 + 512

            # Verify first part is the non-speech window
            assert detector._speech_section[0] == pytest.approx(0.1, abs=0.01)
            # Verify second part is the speech window
            assert detector._speech_section[512] == pytest.approx(0.9, abs=0.01)

            # End speech
            mock_vad.return_value = False
            detector.process_window(non_speech_window, 0.064)

            # Verify segment includes look-back buffer
            assert len(segments) == 1
            assert len(segments[0].audio) == 1536  # prev + speech + final

    def test_no_prev_slice_on_first_window(self):
        """Test speech starting on first window (no prev_slice available)."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # First window is speech (no prev_slice available)
            mock_vad.return_value = True
            detector.process_window(window, 0.0)
            assert len(detector._speech_section) == 512  # Only current window

            # Continue to meet min duration
            detector.process_window(window, 0.032)

            # Now end
            mock_vad.return_value = False
            detector.process_window(window, 0.064)

            assert len(segments) == 1
            assert len(segments[0].audio) == 1536  # 3 speech windows

    def test_multiple_non_speech_windows_only_last_included(self):
        """Test that only the immediately preceding non-speech window is included."""
        segments = []
        window_seconds = get_window_size_samples(16000) / 16000
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            look_back_seconds=window_seconds,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            non_speech1 = make_non_speech_window(level=0.1)
            non_speech2 = make_non_speech_window(level=0.2)
            non_speech3 = make_non_speech_window(level=0.3)
            speech = np.ones(512, dtype=np.float32) * 0.9

            # Three non-speech windows
            mock_vad.return_value = False
            detector.process_window(non_speech1, 0.0)
            detector.process_window(non_speech2, 0.032)
            detector.process_window(non_speech3, 0.064)

            # Only non_speech3 should remain in look-back buffer
            assert len(detector._look_back_buffer) == 1
            assert detector._look_back_buffer[0][0] == pytest.approx(0.3, abs=0.01)

            # Speech starts
            mock_vad.return_value = True
            detector.process_window(speech, 0.096)

            # First window in speech_section should be non_speech3
            assert detector._speech_section[0] == pytest.approx(0.3, abs=0.01)
            assert detector._speech_section[512] == pytest.approx(0.9, abs=0.01)

    def test_configurable_look_back_buffer_accumulates_multiple_windows(self):
        """Test that extended look-back buffer prepends the configured duration."""
        segments = []
        look_back_seconds = 0.5
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            look_back_seconds=look_back_seconds,
            on_segment_complete=lambda s: segments.append(s),
        )

        window_size = detector._window_size_samples
        window_seconds = detector._window_seconds
        total_windows = 20
        non_speech_windows = [
            np.ones(window_size, dtype=np.float32) * (0.01 + i * 0.002) for i in range(total_windows)
        ]
        speech_window = np.ones(window_size, dtype=np.float32) * 0.9

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            for win in non_speech_windows:
                mock_vad.return_value = False
                detector.process_window(win, ts)
                ts += window_seconds

            mock_vad.return_value = True
            detector.process_window(speech_window, ts)

        expected_windows = min(
            len(non_speech_windows),
            int((detector.look_back_seconds + 1e-9) / window_seconds),
        )
        assert len(detector._look_back_buffer) == 0  # Consumed at speech start
        assert len(detector._speech_section) == window_size * (expected_windows + 1)

        if expected_windows > 0:
            expected_prefix = np.concatenate(non_speech_windows[-expected_windows:])
            assert np.allclose(
                detector._speech_section[: expected_windows * window_size],
                expected_prefix,
            )

        assert np.allclose(
            detector._speech_section[-window_size:],
            speech_window,
        )

    def test_look_back_adjusts_start_timestamp(self):
        """Look-back duration should shift the segment start timestamp."""
        segments = []
        window_seconds = get_window_size_samples(16000) / 16000
        look_back_seconds = window_seconds * 2
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=window_seconds * 0.5,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            look_back_seconds=look_back_seconds,
            on_segment_complete=lambda s: segments.append(s),
        )

        non_speech = make_non_speech_window(level=0.1)
        speech = np.ones(512, dtype=np.float32) * 0.9

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0
            mock_vad.return_value = False
            detector.process_window(non_speech, ts)
            ts += window_seconds
            detector.process_window(non_speech, ts)

            mock_vad.return_value = True
            ts += window_seconds
            detector.process_window(speech, ts)
            ts += window_seconds
            detector.process_window(speech, ts)

            mock_vad.return_value = False
            ts += window_seconds
            detector.process_window(non_speech, ts)

        assert len(segments) == 1
        # With look-back, segment should start before the actual speech timestamp
        assert segments[0].start == pytest.approx(0.0, abs=1e-6)  # Pulled back by look-back buffer


class TestFinalWindowInclusion:
    """Test that final window is included when speech ends."""

    def test_final_window_included_in_segment(self):
        """Test that the window triggering speech end is included in segment."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(512, dtype=np.float32) * 0.9
            final_window = make_non_speech_window(level=0.1)

            # Two speech windows
            mock_vad.return_value = True
            detector.process_window(speech_window, 0.0)
            detector.process_window(speech_window, 0.032)

            # Final window that ends speech
            mock_vad.return_value = False
            detector.process_window(final_window, 0.064)

            # Segment should include all three windows
            assert len(segments) == 1
            assert len(segments[0].audio) == 1536  # 3 windows

            # Last 512 samples should be the final window
            final_samples = segments[0].audio[-512:]
            assert final_samples[0] == pytest.approx(0.1, abs=0.01)

    def test_final_window_after_max_duration_split(self):
        """Test final window inclusion when max duration forces split via silence."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=0.128,  # 4 windows
            min_silence_duration_ms=100,  # Normal threshold
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(512, dtype=np.float32)
            silence_window = make_non_speech_window()

            # Process 6 speech windows - exceeds max duration
            mock_vad.return_value = True
            for i in range(6):
                detector.process_window(speech_window, i * 0.032)

            # No split yet - adaptive mode waits for silence
            assert len(segments) == 0

            # Silence triggers adaptive split
            mock_vad.return_value = False
            detector.process_window(silence_window, 6 * 0.032)

            assert len(segments) == 1
            # Should include speech windows + silence window
            assert len(segments[0].audio) >= 512 * 4


class TestMultiSegmentScenarios:
    """Test multiple speech segments in sequence."""

    def test_two_separate_segments_with_timestamps(self):
        """Test two speech segments separated by non-speech have correct timestamps."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # First segment: non-speech, speech, speech, non-speech
            mock_vad.return_value = False
            detector.process_window(window, 0.0)

            mock_vad.return_value = True
            detector.process_window(window, 0.032)
            detector.process_window(window, 0.064)

            mock_vad.return_value = False
            detector.process_window(window, 0.096)

            assert len(segments) == 1

            # Gap of non-speech
            detector.process_window(window, 0.128)
            detector.process_window(window, 0.160)

            # Second segment: speech, speech, non-speech
            mock_vad.return_value = True
            detector.process_window(window, 0.192)
            detector.process_window(window, 0.224)

            mock_vad.return_value = False
            detector.process_window(window, 0.256)

            assert len(segments) == 2

            # Verify timestamps
            # Start timestamps should include look-back duration
            assert segments[0].start == 0.0  # Includes 1 look-back window
            assert segments[1].start == 0.128  # Includes 2 look-back windows

    def test_file_mode_timestamps(self):
        """Test that file timestamps are tracked correctly."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Speech starts and continues to meet min duration
            mock_vad.return_value = True
            detector.process_window(window, 0.0)
            detector.process_window(window, 0.032)

            # Speech ends
            mock_vad.return_value = False
            detector.process_window(window, 0.064)

            assert len(segments) == 1
            assert segments[0].start == 0.0


class TestEdgeCasesAndBoundaries:
    """Test various edge cases and boundary conditions."""

    def test_flush_with_incomplete_segment(self):
        """Test flush completes an in-progress segment."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Start speech but don't end it
            mock_vad.return_value = True
            detector.process_window(window, 0.0)
            detector.process_window(window, 0.032)

            # Flush should complete the segment
            detector.flush()

            assert len(segments) == 1
            assert len(segments[0].audio) == 1024

    def test_reset_clears_incomplete_segment_without_emitting(self):
        """Test reset clears in-progress segment without emitting."""
        segments = []
        detector = SpeechDetector(
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Start speech
            mock_vad.return_value = True
            detector.process_window(window, 0.0)

            assert detector.is_in_speech is True

            # Reset should clear without emitting
            detector.reset()

            assert detector.is_in_speech is False
            assert len(segments) == 0
            assert len(detector._speech_section) == 0

    def test_duration_calculation_precision(self):
        """Test duration calculation is precise for segments."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            soft_limit_seconds=10.0,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Need multiple windows to meet min duration
            mock_vad.return_value = True
            detector.process_window(window, 0.0)
            detector.process_window(window, 0.032)

            mock_vad.return_value = False
            detector.process_window(window, 0.064)

            assert len(segments) == 1
            # 1536 samples @ 16000 Hz = 0.096 seconds
            expected_duration = 1536 / 16000
            assert segments[0].duration_seconds == pytest.approx(expected_duration, abs=0.0001)

    def test_pending_non_speech_tracking(self):
        """Test that pending non-speech is tracked correctly for backlog management."""
        detector = SpeechDetector(sample_rate=16000)

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = make_non_speech_window()

            # Non-speech should track pending duration
            mock_vad.return_value = False
            detector.process_window(window, 0.0)

            assert detector.pending_non_speech_duration > 0

            # Consuming non-speech should clear it
            consumed = detector.consume_non_speech()
            assert consumed > 0
            assert detector.pending_non_speech_duration == 0

    def test_min_and_max_interaction_handles_flapping_vad(self):
        """Test that min/max enforcement handles VAD outputs that oscillate."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds * 2,
            soft_limit_seconds=window_seconds * 2.5,
            min_silence_duration_ms=100,  # Use adaptive mode
            on_segment_complete=lambda s: segments.append(s),
        )

        def make_window(value: float) -> np.ndarray:
            return np.ones(window_size, dtype=np.float32) * value

        windows = [
            make_window(0.1),
            make_window(0.2),
            make_window(0.3),
            make_window(0.5),
        ]
        vad_outputs = [True, False, True, True]  # False should be ignored until min met

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0
            for win, vad in zip(windows, vad_outputs):
                mock_vad.return_value = vad
                detector.process_window(win, ts)
                ts += window_seconds

            # At this point, exceeds soft_limit_seconds but no silence yet
            assert len(segments) == 0

            # Add silence to trigger adaptive split
            mock_vad.return_value = False
            detector.process_window(make_window(0.05), ts)

        assert len(segments) == 1
        segment = segments[0]
        # 4 speech windows + 1 silence window
        assert len(segment.audio) == window_size * 5

        # Second window should be kept despite VAD returning False (due to min enforcement)
        second_window = segment.audio[window_size:window_size * 2]
        assert np.allclose(second_window, windows[1])


class TestMixedBoundarySegments:
    """Test segments with varied boundary behaviors."""

    def test_segments_with_and_without_boundary_buffers(self):
        """Ensure segments can mix look-back starts and explicit non-speech windows."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 10,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            look_back_seconds=window_seconds,
            on_segment_complete=lambda s: segments.append(s),
        )

        def make_window(value: float) -> np.ndarray:
            return np.ones(window_size, dtype=np.float32) * value

        speech_a1 = make_window(0.1)
        speech_a2 = make_window(0.2)
        non_speech_end = make_non_speech_window(length=window_size, level=0.01)
        non_speech_buffer1 = make_non_speech_window(length=window_size, level=0.02)
        non_speech_buffer2 = make_non_speech_window(length=window_size, level=0.03)
        speech_b1 = make_window(0.4)
        speech_b2 = make_window(0.5)
        non_speech_end2 = make_non_speech_window(length=window_size, level=0.04)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window: np.ndarray, vad: bool) -> None:
                nonlocal ts
                mock_vad.return_value = vad
                detector.process_window(window, ts)
                ts += window_seconds

            # Segment 1: starts immediately with speech, ends with explicit non-speech window
            run(speech_a1, True)
            run(speech_a2, True)
            run(non_speech_end, False)

            # Additional non-speech windows create look-back buffer for next segment
            run(non_speech_buffer1, False)
            run(non_speech_buffer2, False)

            # Segment 2: includes look-back at start and ends with silence
            run(speech_b1, True)
            run(speech_b2, True)
            run(non_speech_end2, False)  # Ends segment 2

        assert len(segments) == 2

        first_segment = segments[0]
        assert len(first_segment.audio) == window_size * 3
        assert np.allclose(first_segment.audio[:window_size], speech_a1)
        assert np.allclose(first_segment.audio[-window_size:], non_speech_end)

        second_segment = segments[1]
        assert len(second_segment.audio) == window_size * 4  # look-back + 2 speech + silence
        assert np.allclose(second_segment.audio[:window_size], non_speech_buffer2)
        assert np.allclose(
            second_segment.audio[window_size:window_size * 2],
            speech_b1,
        )
        assert np.allclose(second_segment.audio[window_size * 2:window_size * 3], speech_b2)
        assert np.allclose(second_segment.audio[-window_size:], non_speech_end2)

    def test_single_non_speech_between_segments_not_duplicated(self):
        """
        Ensure exactly one non-speech window between audio segments isn't counted twice.

        This simulates a stream that emits one segment (ending with silence), has exactly
        one window's worth of low-energy audio gap, and immediately emits the next segment.
        The gap window should appear once at the start of the second segment via look-back.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 10,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        def make_window(value: float) -> np.ndarray:
            return np.ones(window_size, dtype=np.float32) * value

        speech1 = make_window(0.5)
        silence_end = make_non_speech_window(length=window_size, level=0.01)
        gap = make_non_speech_window(length=window_size, level=0.02)
        speech2 = make_window(0.6)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window: np.ndarray, vad: bool) -> None:
                nonlocal ts
                mock_vad.return_value = vad
                detector.process_window(window, ts)
                ts += window_seconds

            # First segment: speech then silence ends it
            run(speech1, True)
            run(speech1, True)
            run(silence_end, False)  # Ends first segment

            # Gap window (goes to look-back buffer)
            run(gap, False)

            # Second segment resumes immediately after the gap
            run(speech2, True)
            run(speech2, True)

        # Flush to emit the second segment
        detector.flush()

        assert len(segments) == 2

        first_segment = segments[0]
        assert len(first_segment.audio) == window_size * 3  # 2 speech + 1 silence
        assert np.allclose(first_segment.audio, np.concatenate([speech1, speech1, silence_end]))

        second_segment = segments[1]
        assert len(second_segment.audio) == window_size * 3  # 1 gap (look-back) + 2 speech
        assert np.allclose(
            second_segment.audio,
            np.concatenate([gap, speech2, speech2]),
        )

    def test_max_speech_split_preserves_contiguous_audio(self):
        """
        Ensure soft_limit_seconds splits create contiguous segments without losing samples.

        With adaptive behavior, a silence window triggers the split after exceeding
        soft_limit_seconds. The silence window is included in the first segment.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 1.5,
            min_silence_duration_ms=0,  # Immediate end on silence
            on_segment_complete=lambda s: segments.append(s),
        )

        def make_window(value: float) -> np.ndarray:
            return np.full(window_size, value, dtype=np.float32)

        speech_a1 = make_window(0.1)
        speech_a2 = make_window(0.2)
        silence = make_non_speech_window(length=window_size, level=0.01)
        speech_b1 = make_window(0.4)
        speech_b2 = make_window(0.5)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window: np.ndarray, is_speech: bool) -> None:
                nonlocal ts
                mock_vad.return_value = is_speech
                detector.process_window(window, ts)
                ts += window_seconds

            # First segment: speech exceeds max, then silence triggers split
            run(speech_a1, True)
            run(speech_a2, True)
            run(silence, False)  # Triggers split (over max + silence)

            # Second segment: speech continues
            run(speech_b1, True)
            run(speech_b2, True)

        detector.flush()

        assert len(segments) == 2

        first_segment = segments[0]
        assert len(first_segment.audio) == window_size * 3  # 2 speech + 1 silence
        assert np.allclose(
            first_segment.audio,
            np.concatenate([speech_a1, speech_a2, silence]),
        )

        second_segment = segments[1]
        assert len(second_segment.audio) == window_size * 2
        assert np.allclose(
            second_segment.audio,
            np.concatenate([speech_b1, speech_b2]),
        )

    def test_default_look_back_gap_under_half_second_keeps_gap(self):
        """
        With default 0.5s look-back, a short non-speech gap should prepend to the next segment.
        """
        segments: list[AudioSegment] = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 3,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        speech_window = np.ones(window_size, dtype=np.float32) * 0.8
        non_speech_window = make_non_speech_window(length=window_size, level=0.02)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window: np.ndarray, has_speech: bool):
                nonlocal ts
                mock_vad.return_value = has_speech
                detector.process_window(window, ts)
                ts += window_seconds

            # First segment: 2 speech windows
            run(speech_window, True)
            run(speech_window, True)
            run(non_speech_window, False)  # ends first segment

            # Short gap: 0.2 seconds (approx 6 windows at 512/16k)
            for _ in range(6):
                run(non_speech_window, False)

            # Second segment: resumes speech immediately
            run(speech_window, True)
            run(speech_window, True)

        detector.flush()
        assert len(segments) == 2

        first_segment = segments[0]
        assert len(first_segment.audio) == window_size * 3
        assert np.allclose(
            first_segment.audio,
            np.concatenate([speech_window, speech_window, non_speech_window]),
        )

        second_segment = segments[1]
        assert len(second_segment.audio) == window_size * (2 + 6)
        assert np.allclose(
            second_segment.audio,
            np.concatenate([np.tile(non_speech_window, 6), speech_window, speech_window]),
        )

    def test_default_look_back_caps_to_half_second(self):
        """When non-speech exceeds 0.5s, only the most recent 0.5s should prepend."""
        segments: list[AudioSegment] = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=window_seconds * 3,
            min_silence_duration_ms=0,  # Immediate end on silence for this test
            on_segment_complete=lambda s: segments.append(s),
        )

        speech_window = np.ones(window_size, dtype=np.float32) * 0.7
        non_speech_window = make_non_speech_window(length=window_size, level=0.03)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window: np.ndarray, has_speech: bool):
                nonlocal ts
                mock_vad.return_value = has_speech
                detector.process_window(window, ts)
                ts += window_seconds

            # First segment ends with non-speech
            run(speech_window, True)
            run(speech_window, True)
            run(non_speech_window, False)

            # Long gap: 20 windows (~0.64s)
            for _ in range(20):
                run(non_speech_window, False)

            # Second segment resumes
            run(speech_window, True)
            run(speech_window, True)

        detector.flush()
        assert len(segments) == 2
        second_segment = segments[1]
        max_windows = int((detector.look_back_seconds + 1e-9) / window_seconds)
        assert len(second_segment.audio) == window_size * (2 + max_windows)
        assert np.allclose(
            second_segment.audio[: max_windows * window_size],
            np.tile(non_speech_window, max_windows),
        )


class TestHardLimitForceSplit:
    """Tests for hard limit force-split behavior."""

    def test_force_split_at_hard_limit(self):
        """Segment is emitted exactly at the hard limit, not one window over."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 5  # 5 windows

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=hard_limit,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=2000,
            on_segment_complete=lambda s: segments.append(s),
        )

        speech = np.full(window_size, 0.5, dtype=np.float32)

        with patch.object(detector, '_detect_speech', return_value=True):
            for i in range(8):
                detector.process_window(speech, i * window_seconds)

        detector.flush()

        # First segment should be exactly 5 windows (the hard limit)
        assert len(segments) == 2
        assert len(segments[0].audio) == window_size * 5
        assert segments[0].duration_seconds == pytest.approx(hard_limit)
        # Second segment gets the remaining 3 windows
        assert len(segments[1].audio) == window_size * 3

    def test_force_split_contiguous_audio(self):
        """No samples lost or duplicated across force-split boundary."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 3

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=hard_limit,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=2000,
            on_segment_complete=lambda s: segments.append(s),
        )

        # Each window has a unique value so we can verify order
        windows = [np.full(window_size, i * 0.1, dtype=np.float32) for i in range(7)]

        with patch.object(detector, '_detect_speech', return_value=True):
            for i, w in enumerate(windows):
                detector.process_window(w, i * window_seconds)

        detector.flush()

        # Should split into: [0,1,2], [3,4,5], [6]
        assert len(segments) == 3

        all_audio = np.concatenate([s.audio for s in segments])
        expected = np.concatenate(windows)
        assert np.allclose(all_audio, expected)

    def test_force_split_timestamps(self):
        """Timestamps are correct after force-split."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 3

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=hard_limit,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=2000,
            look_back_seconds=0.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        speech = np.full(window_size, 0.5, dtype=np.float32)

        with patch.object(detector, '_detect_speech', return_value=True):
            for i in range(7):
                detector.process_window(speech, i * window_seconds)

        detector.flush()

        assert len(segments) == 3
        # First segment starts at 0
        assert segments[0].start == pytest.approx(0.0)
        # Second segment starts where first ended
        assert segments[1].start == pytest.approx(hard_limit)
        # Third segment starts where second ended
        assert segments[2].start == pytest.approx(hard_limit * 2)

    def test_force_split_then_silence_ends_normally(self):
        """After a force-split, silence still ends the next segment normally."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 3

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=hard_limit,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=0,
            on_segment_complete=lambda s: segments.append(s),
        )

        speech = np.full(window_size, 0.5, dtype=np.float32)
        silence = make_non_speech_window(length=window_size, level=0.01)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window, is_speech):
                nonlocal ts
                mock_vad.return_value = is_speech
                detector.process_window(window, ts)
                ts += window_seconds

            # 4 speech windows — force-split at 3
            run(speech, True)
            run(speech, True)
            run(speech, True)
            run(speech, True)  # starts new segment after force-split
            run(speech, True)
            run(silence, False)  # ends second segment via silence

        assert len(segments) == 2
        assert len(segments[0].audio) == window_size * 3  # hard limit
        assert len(segments[1].audio) == window_size * 3  # 2 speech + 1 silence


    def test_force_split_with_silence_buffer_near_limit(self):
        """Silence buffer is included in duration check so segment doesn't exceed hard limit.

        This tests the bug where _silence_buffer was not counted in the duration,
        causing segments to overshoot the hard limit when silence accumulated.
        """
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 5  # 5 windows

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=hard_limit,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=int(window_seconds * 3 * 1000),  # 3 windows of silence needed
            on_segment_complete=lambda s: segments.append(s),
        )

        speech = np.full(window_size, 0.5, dtype=np.float32)
        silence = make_non_speech_window(length=window_size, level=0.01)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window, is_speech):
                nonlocal ts
                mock_vad.return_value = is_speech
                detector.process_window(window, ts)
                ts += window_seconds

            # 3 speech windows, then silence windows approaching hard limit
            run(speech, True)   # 1 window in _speech_section
            run(speech, True)   # 2 windows
            run(speech, True)   # 3 windows
            run(silence, False)  # 3 speech + 1 silence buffer = 4 windows total
            run(silence, False)  # 3 speech + 2 silence buffer = 5 windows = hard limit

        detector.flush()

        # The hard limit should trigger at 5 windows total (speech + silence)
        # Segment must NOT exceed hard_limit
        assert len(segments) >= 1
        for seg in segments:
            assert seg.duration_seconds <= hard_limit + 1e-9, (
                f"Segment duration {seg.duration_seconds:.4f}s exceeds "
                f"hard limit {hard_limit:.4f}s"
            )


    def test_soft_limit_none_skips_adaptive(self):
        """With soft_limit_seconds=None, adaptive silence is never used."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 6

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=None,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=int(window_seconds * 3 * 1000),  # 3 windows
            on_segment_complete=lambda s: segments.append(s),
        )

        speech = np.full(window_size, 0.5, dtype=np.float32)
        silence = make_non_speech_window(length=window_size, level=0.01)

        with patch.object(detector, '_detect_speech') as mock_vad:
            ts = 0.0

            def run(window, is_speech):
                nonlocal ts
                mock_vad.return_value = is_speech
                detector.process_window(window, ts)
                ts += window_seconds

            # 3 speech + 1 silence — without adaptive, 1 silence window
            # is NOT enough (need 3 windows of silence)
            run(speech, True)
            run(speech, True)
            run(speech, True)
            run(silence, False)  # only 1 silence window, need 3
            run(speech, True)    # speech resumes, silence flushed back

        detector.flush()

        # Should be 1 segment — the single silence window didn't end it
        assert len(segments) == 1
        assert len(segments[0].audio) == window_size * 5  # 3 speech + 1 silence + 1 speech

    def test_soft_limit_none_still_force_splits_at_hard_limit(self):
        """With soft_limit_seconds=None, hard limit still force-splits."""
        segments = []
        sample_rate = 16000
        window_size = get_window_size_samples(sample_rate)
        window_seconds = window_size / sample_rate
        hard_limit = window_seconds * 3

        detector = SpeechDetector(
            sample_rate=sample_rate,
            min_speech_seconds=window_seconds,
            soft_limit_seconds=None,
            hard_limit_seconds=hard_limit,
            min_silence_duration_ms=2000,
            on_segment_complete=lambda s: segments.append(s),
        )

        speech = np.full(window_size, 0.5, dtype=np.float32)

        with patch.object(detector, '_detect_speech', return_value=True):
            for i in range(7):
                detector.process_window(speech, i * window_seconds)

        detector.flush()

        # Should split into [0,1,2], [3,4,5], [6]
        assert len(segments) == 3
        assert len(segments[0].audio) == window_size * 3
        assert len(segments[1].audio) == window_size * 3
        assert len(segments[2].audio) == window_size * 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
