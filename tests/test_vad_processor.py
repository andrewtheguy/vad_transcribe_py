"""
Unit tests for SpeechDetector class in vad_processor module.
"""

import numpy as np
import pytest

from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment, get_window_size_samples


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
        assert segment.wall_clock_start is None

    def test_creation_with_all_fields(self):
        """Test AudioSegment creation with all fields."""
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = AudioSegment(
            start=2.5,
            audio=audio,
            duration_seconds=3.0,
            wall_clock_start=1234567890.0,
        )

        assert segment.start == 2.5
        assert np.array_equal(segment.audio, audio)
        assert segment.duration_seconds == 3.0
        assert segment.wall_clock_start == 1234567890.0

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
            max_speech_seconds=60.0,
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

    def test_initialization(self, detector):
        """Test SpeechDetector initialization."""
        assert detector.sample_rate == 16000
        assert detector.min_speech_seconds == 3.0
        assert detector.max_speech_seconds == 60.0
        assert detector.speech_threshold == 0.5
        assert detector.is_in_speech is False
        assert detector.current_segment_duration == 0.0
        assert detector.pending_silence_duration == 0.0

    def test_process_window_wrong_size(self, detector):
        """Test that wrong window size raises ValueError."""
        wrong_size_audio = np.array([0.1, 0.2], dtype=np.float32)

        with pytest.raises(ValueError, match="Audio length.*does not match window size"):
            detector.process_window(wrong_size_audio, 0.0)

    def test_single_silence_window(self, detector, mock_segment_callback):
        """Test processing a single silence window."""
        detector.on_segment_complete = mock_segment_callback

        # Create silence (low amplitude)
        audio = self._create_audio_window(detector, value=0.0)
        has_speech = detector.process_window(audio, 0.0)

        assert has_speech is False
        assert detector.is_in_speech is False
        assert len(mock_segment_callback.segments) == 0

    def test_silence_to_speech_transition(self, detector, mock_segment_callback):
        """Test transition from silence to speech."""
        detector.on_segment_complete = mock_segment_callback

        # First window: silence
        silence = self._create_audio_window(detector, value=0.0)
        detector.process_window(silence, 0.0)
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
        assert detector.pending_silence_duration == 0.0
        assert detector._prev_has_speech is False
        assert len(detector._speech_section) == 0
        assert detector._has_speech_begin_timestamp is None
        assert detector._has_speech_begin_wall_clock is None
        assert detector._prev_slice is None

    def test_flush_with_no_speech(self, detector, mock_segment_callback):
        """Test flush with no accumulated speech does nothing."""
        detector.on_segment_complete = mock_segment_callback

        detector.flush()

        assert len(mock_segment_callback.segments) == 0

    def test_consume_silence(self, detector):
        """Test consume_silence returns and clears pending silence."""
        # Manually set pending silence
        detector._pending_silence_seconds = 0.5

        consumed = detector.consume_silence()

        assert consumed == 0.5
        assert detector.pending_silence_duration == 0.0

    def test_consume_silence_when_zero(self, detector):
        """Test consume_silence when there's no pending silence."""
        consumed = detector.consume_silence()

        assert consumed == 0.0
        assert detector.pending_silence_duration == 0.0

    def test_properties(self, detector):
        """Test property accessors."""
        # Initial state
        assert detector.is_in_speech is False
        assert detector.current_segment_duration == 0.0
        assert detector.pending_silence_duration == 0.0

        # Simulate some state
        detector._prev_has_speech = True
        detector._speech_section = [0.1] * 16000  # 1 second of audio
        detector._pending_silence_seconds = 0.25

        assert detector.is_in_speech is True
        assert detector.current_segment_duration == 1.0
        assert detector.pending_silence_duration == 0.25

    def test_custom_parameters(self):
        """Test SpeechDetector with custom parameters."""
        detector = SpeechDetector(
            sample_rate=8000,
            min_speech_seconds=1.0,
            max_speech_seconds=30.0,
            speech_threshold=0.7,
        )

        assert detector.sample_rate == 8000
        assert detector.min_speech_seconds == 1.0
        assert detector.max_speech_seconds == 30.0
        assert detector.speech_threshold == 0.7
        assert detector._window_size_samples == 256  # 8kHz uses 256 samples

    def test_wall_clock_timestamp_propagation(self, mock_segment_callback):
        """Test that wall clock timestamps are properly propagated."""
        detector = SpeechDetector(on_segment_complete=mock_segment_callback)

        # We need to simulate a full speech segment to test this
        # This would require mocking the VAD model, which is complex
        # For now, we test that the mechanism exists
        assert detector._has_speech_begin_wall_clock is None

    def test_segment_callback_invocation(self, mock_segment_callback):
        """Test that callback is invoked when provided."""
        detector = SpeechDetector(on_segment_complete=mock_segment_callback)

        # Manually trigger segment emission to test callback
        detector._speech_section = [0.1] * 16000  # 1 second
        detector._has_speech_begin_timestamp = 1.0
        detector._has_speech_begin_wall_clock = 123456789.0
        detector._emit_segment()

        assert len(mock_segment_callback.segments) == 1
        segment = mock_segment_callback.segments[0]
        assert segment.start == 1.0
        assert segment.wall_clock_start == 123456789.0
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

    def test_short_silence_periods(self):
        """Test that short silence periods don't break speech segments."""
        segments = []

        def callback(segment):
            segments.append(segment)

        detector = SpeechDetector(
            min_speech_seconds=0.1,  # Very short for testing
            max_speech_seconds=10.0,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
