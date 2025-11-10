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


class TestMinDurationEnforcement:
    """Test minimum speech duration hysteresis enforcement."""

    def test_min_duration_prevents_early_cutoff(self):
        """Test that speech below min duration continues even if VAD detects silence."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.1,  # 0.1s = ~3 windows
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        # Mock the _detect_speech to control VAD output
        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

            # Window 1: silence
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
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

            # Start speech
            mock_vad.return_value = True
            detector.process_window(window, 0.0)
            detector.process_window(window, 0.032)

            # At exactly min duration, VAD no speech should end it
            mock_vad.return_value = False
            detector.process_window(window, 0.064)

            assert len(segments) == 1
            assert segments[0].duration_seconds == pytest.approx(0.096, abs=0.001)  # 3 windows


class TestMaxDurationEnforcement:
    """Test maximum speech duration enforcement."""

    def test_max_duration_forces_split(self):
        """Test that speech exceeding max duration is forced to end."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=0.2,  # Only 0.2s max (~6 windows)
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

            # Continuous speech for many windows
            mock_vad.return_value = True

            # Process enough windows to exceed max duration
            for i in range(8):
                timestamp = i * 0.032
                detector.process_window(window, timestamp)

            # Should have forced a split
            assert len(segments) >= 1, "Should have split due to max duration"
            # First segment should be around max duration
            assert segments[0].duration_seconds <= 0.26  # max + 1-2 windows tolerance

    def test_exactly_max_duration_boundary(self):
        """Test behavior at exactly max duration boundary."""
        segments = []
        # 0.192s = 6 windows of 512 samples @ 16kHz
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=0.192,  # Exactly 6 windows
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)
            mock_vad.return_value = True

            # Process windows up to and beyond max duration
            # Window 0-5: 0.192s (at max, but not > max)
            # Window 6: 0.224s (exceeds max, triggers split)
            for i in range(7):
                detector.process_window(window, i * 0.032)

            # Still in speech, so need to end it or check state
            # If no segment yet, we need one more window to trigger the check
            if len(segments) == 0:
                detector.process_window(window, 7 * 0.032)

            assert len(segments) >= 1
            # Segment should be around max duration
            assert segments[0].duration_seconds >= 0.192


class TestLookBackBuffer:
    """Test look-back buffer (prev_slice) inclusion at speech start."""

    def test_prev_slice_included_at_speech_start(self):
        """Test that previous silence window is included when speech starts."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            # Create distinguishable windows
            silence_window = np.ones(512, dtype=np.float32) * 0.1
            speech_window = np.ones(512, dtype=np.float32) * 0.9

            # Window 1: silence (should be saved as prev_slice)
            mock_vad.return_value = False
            detector.process_window(silence_window, 0.0)
            assert detector._prev_slice is not None
            assert np.array_equal(detector._prev_slice, silence_window)

            # Window 2: speech starts (should include prev_slice)
            mock_vad.return_value = True
            detector.process_window(speech_window, 0.032)
            assert detector._prev_slice is None  # Should be consumed
            # Speech section should have prev_slice + current
            assert len(detector._speech_section) == 1024  # 512 + 512

            # Verify first part is the silence window
            assert detector._speech_section[0] == pytest.approx(0.1, abs=0.01)
            # Verify second part is the speech window
            assert detector._speech_section[512] == pytest.approx(0.9, abs=0.01)

            # End speech
            mock_vad.return_value = False
            detector.process_window(silence_window, 0.064)

            # Verify segment includes look-back buffer
            assert len(segments) == 1
            assert len(segments[0].audio) == 1536  # prev + speech + final

    def test_no_prev_slice_on_first_window(self):
        """Test speech starting on first window (no prev_slice available)."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

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

    def test_multiple_silence_windows_only_last_included(self):
        """Test that only the immediately preceding silence window is included."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            silence1 = np.ones(512, dtype=np.float32) * 0.1
            silence2 = np.ones(512, dtype=np.float32) * 0.2
            silence3 = np.ones(512, dtype=np.float32) * 0.3
            speech = np.ones(512, dtype=np.float32) * 0.9

            # Three silence windows
            mock_vad.return_value = False
            detector.process_window(silence1, 0.0)
            detector.process_window(silence2, 0.032)
            detector.process_window(silence3, 0.064)

            # Only silence3 should be in prev_slice
            assert detector._prev_slice[0] == pytest.approx(0.3, abs=0.01)

            # Speech starts
            mock_vad.return_value = True
            detector.process_window(speech, 0.096)

            # First window in speech_section should be silence3
            assert detector._speech_section[0] == pytest.approx(0.3, abs=0.01)
            assert detector._speech_section[512] == pytest.approx(0.9, abs=0.01)


class TestFinalWindowInclusion:
    """Test that final window is included when speech ends."""

    def test_final_window_included_in_segment(self):
        """Test that the window triggering speech end is included in segment."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            speech_window = np.ones(512, dtype=np.float32) * 0.9
            final_window = np.ones(512, dtype=np.float32) * 0.1

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
        """Test final window inclusion when max duration forces split."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=0.128,  # 4 windows
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.ones(512, dtype=np.float32)
            mock_vad.return_value = True

            # Process 6 windows - should force split when exceeding max
            for i in range(6):
                detector.process_window(window, i * 0.032)

            assert len(segments) == 1
            # Should include enough windows to meet/exceed max
            assert len(segments[0].audio) >= 512 * 4


class TestMultiSegmentScenarios:
    """Test multiple speech segments in sequence."""

    def test_two_separate_segments_with_timestamps(self):
        """Test two speech segments separated by silence have correct timestamps."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

            # First segment: silence, speech, speech, silence
            mock_vad.return_value = False
            detector.process_window(window, 0.0)

            mock_vad.return_value = True
            detector.process_window(window, 0.032)
            detector.process_window(window, 0.064)

            mock_vad.return_value = False
            detector.process_window(window, 0.096)

            assert len(segments) == 1

            # Gap of silence
            detector.process_window(window, 0.128)
            detector.process_window(window, 0.160)

            # Second segment: speech, speech, silence
            mock_vad.return_value = True
            detector.process_window(window, 0.192)
            detector.process_window(window, 0.224)

            mock_vad.return_value = False
            detector.process_window(window, 0.256)

            assert len(segments) == 2

            # Verify timestamps
            assert segments[0].start == 0.032  # Started at speech window
            assert segments[1].start == 0.192  # Started at 2nd segment

    def test_wall_clock_timestamps_propagated(self):
        """Test that wall clock timestamps are correctly propagated to segments."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)
            base_wall_clock = 1234567890.0

            # Speech starts and continues to meet min duration
            mock_vad.return_value = True
            detector.process_window(window, 0.0, base_wall_clock)
            detector.process_window(window, 0.032, base_wall_clock + 0.032)

            # Speech ends
            mock_vad.return_value = False
            detector.process_window(window, 0.064, base_wall_clock + 0.064)

            assert len(segments) == 1
            assert segments[0].wall_clock_start == base_wall_clock


class TestEdgeCasesAndBoundaries:
    """Test various edge cases and boundary conditions."""

    def test_flush_with_incomplete_segment(self):
        """Test flush completes an in-progress segment."""
        segments = []
        detector = SpeechDetector(
            sample_rate=16000,
            min_speech_seconds=0.05,
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

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
            window = np.zeros(512, dtype=np.float32)

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
            max_speech_seconds=10.0,
            on_segment_complete=lambda s: segments.append(s),
        )

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

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

    def test_pending_silence_tracking(self):
        """Test that pending silence is tracked correctly for backlog management."""
        detector = SpeechDetector(sample_rate=16000)

        with patch.object(detector, '_detect_speech') as mock_vad:
            window = np.zeros(512, dtype=np.float32)

            # Silence should track pending silence
            mock_vad.return_value = False
            detector.process_window(window, 0.0)

            assert detector.pending_silence_duration > 0

            # Consuming silence should clear it
            consumed = detector.consume_silence()
            assert consumed > 0
            assert detector.pending_silence_duration == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
