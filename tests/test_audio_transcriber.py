import queue
import numpy as np
import pytest

import whisper_transcribe_py.audio_transcriber as audio_transcriber
from whisper_transcribe_py.vad_processor import AudioSegment


class TrackingSpeechDetector:
    """Mock SpeechDetector for testing."""

    def __init__(self, sample_rate: int, min_speech_seconds: float = 3.0, max_speech_seconds: float = 60.0, on_segment_complete=None):
        self.sample_rate = sample_rate
        self.min_speech_seconds = min_speech_seconds
        self.max_speech_seconds = max_speech_seconds
        self.on_segment_complete = on_segment_complete
        self.calls: list[tuple[float, float | None, int]] = []
        self.pending_non_speech_duration = 0.0
        self.is_in_speech = False
        self._script: list[tuple[bool, bool] | tuple[bool, bool, bool]] = []
        self._script_index = 0
        self._accumulated_audio = []

    def process_window(self, audio_window, timestamp, wall_clock_timestamp):
        self.calls.append((timestamp, wall_clock_timestamp, len(audio_window)))
        self._accumulated_audio.append(audio_window.copy())

        if self._script_index < len(self._script):
            script_entry = self._script[self._script_index]
            has_speech = script_entry[0]
            in_speech = script_entry[1]
            trigger_complete = script_entry[2] if len(script_entry) > 2 else False

            self._script_index += 1
            self.is_in_speech = in_speech
            if not has_speech and not in_speech:
                self.pending_non_speech_duration = len(audio_window) / self.sample_rate
            else:
                self.pending_non_speech_duration = 0.0

            # Trigger segment completion if requested
            if trigger_complete and self.on_segment_complete:
                combined_audio = np.concatenate(self._accumulated_audio)
                segment = AudioSegment(
                    start=timestamp,
                    audio=combined_audio,
                    wall_clock_start=None,
                    duration_seconds=len(combined_audio) / self.sample_rate
                )
                self.on_segment_complete(segment)
                self._accumulated_audio = []

            return has_speech
        return False

    def flush(self):
        pass

    def consume_non_speech(self) -> float:
        return 0.0

    def reset(self) -> None:
        """Reset the detector state."""
        self._script_index = 0
        self.is_in_speech = False
        self.pending_non_speech_duration = 0.0
        self._accumulated_audio = []

    def set_script(self, script: list[tuple[bool, bool] | tuple[bool, bool, bool]]) -> None:
        self._script = script
        self._script_index = 0


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    monkeypatch.setattr(audio_transcriber.AudioTranscriber, "_load_whisper_cpp", lambda self: None)
    monkeypatch.setattr(audio_transcriber, "SpeechDetector", TrackingSpeechDetector)


@pytest.fixture
def recorded_transcribe(monkeypatch):
    recorded: list = []

    def fake_transcribe(self):
        while True:
            item = self.transcribe_queue.get()
            recorded.append(item)
            if item is None:
                break

    monkeypatch.setattr(audio_transcriber.AudioTranscriber, "_transcribe", fake_transcribe)
    return recorded


@pytest.fixture
def make_transcriber():
    def _builder(*, audio_queue=None, **kwargs):
        if audio_queue is None:
            audio_queue = queue.Queue()
        return audio_transcriber.AudioTranscriber(audio_queue, **kwargs)

    return _builder


def test_process_input_feeds_speech_detector_windows(make_transcriber, recorded_transcribe):
    """Test that audio is fed to speech detector in correct window sizes."""
    audio_queue = queue.Queue()
    audio = np.zeros(1024, dtype=np.float32)
    duration = len(audio) / audio_transcriber.TARGET_SAMPLE_RATE
    segment = AudioSegment(start=5.0, audio=audio, wall_clock_start=None, duration_seconds=duration)
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber._process_input_prerecorded(audio_transcriber.TARGET_SAMPLE_RATE)

    calls = transcriber.speech_detector.calls
    assert len(calls) == 2
    first_ts, first_wall, first_len = calls[0]
    second_ts, _, _ = calls[1]
    assert first_ts == 5.0
    assert first_wall is None  # File mode: no wall_clock_timestamp
    assert first_len == 512


def test_transcriber_initialization(make_transcriber):
    """Test that transcriber initializes with correct parameters."""
    audio_queue = queue.Queue()
    transcriber = make_transcriber(
        audio_queue=audio_queue,
        language="en",
        show_name="test",
        model="large-v3-turbo",
        backend="whisper_cpp"
    )

    assert transcriber.language == "en"
    assert transcriber.show_name == "test"
    assert transcriber.model == "large-v3-turbo"
    assert transcriber.backend == "whisper_cpp"


def test_transcriber_vad_min_max_speech(make_transcriber):
    """Test that VAD min/max speech seconds are configured correctly."""
    audio_queue = queue.Queue()
    transcriber = make_transcriber(
        audio_queue=audio_queue,
        language="en",
        vad_min_speech_seconds=2.0,
        vad_max_speech_seconds=45.0
    )

    assert transcriber.speech_detector.min_speech_seconds == 2.0
    assert transcriber.speech_detector.max_speech_seconds == 45.0


def test_transform_segment_file_mode(make_transcriber):
    """Test that segment transformation returns prerecorded mode format."""
    audio_queue = queue.Queue()
    transcriber = make_transcriber(audio_queue=audio_queue, language="en", show_name="test_show")

    class MockSegment:
        t0 = 100  # milliseconds
        t1 = 200  # milliseconds
        text = "hello world"

    result = transcriber._transform_segment(MockSegment())

    assert result.show_name == "test_show"
    assert result.language == "en"
    assert result.text == "hello world"
    assert result.relative_start == 0.1
    assert result.relative_end == 0.2
    assert result.start_timestamp is None  # File mode: no timestamps
    assert result.end_timestamp is None


def test_pcm_conversion_functions():
    """Test PCM audio conversion functions."""
    # Test pcm_int16_to_float32
    int16_audio = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    float32_audio = audio_transcriber.pcm_int16_to_float32(int16_audio)

    assert float32_audio.dtype == np.float32
    assert float32_audio[0] == 0.0
    assert -1.0 <= float32_audio[3] <= 1.0

    # Test pcm_s16le_to_float32
    pcm_bytes = int16_audio.tobytes()
    float32_from_bytes = audio_transcriber.pcm_s16le_to_float32(pcm_bytes)

    assert float32_from_bytes.dtype == np.float32
    assert len(float32_from_bytes) == len(int16_audio)


def test_vad_segment_callback(make_transcriber):
    """Test that VAD segment callback is called when segment completes."""
    audio_queue = queue.Queue()
    callback_results = []

    def mock_callback(segment: AudioSegment):
        callback_results.append(segment)

    transcriber = make_transcriber(
        audio_queue=audio_queue,
        language="en",
        audio_segment_callback=mock_callback
    )

    # Create a mock audio segment and call the VAD callback
    test_audio = np.zeros(512, dtype=np.float32)
    test_segment = AudioSegment(
        start=0.0,
        audio=test_audio,
        wall_clock_start=None,
        duration_seconds=0.032
    )

    transcriber._handle_vad_segment(test_segment)

    # Callback should have been called
    assert len(callback_results) == 1
    assert callback_results[0] == test_segment
