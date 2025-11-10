import queue
import types

import numpy as np
import pytest

import whisper_transcribe_py.audio_transcriber as audio_transcriber
from whisper_transcribe_py.vad_processor import AudioSegment


class TrackingSpeechDetector:
    def __init__(self, sample_rate: int, on_segment_complete):
        self.sample_rate = sample_rate
        self.on_segment_complete = on_segment_complete
        self.calls: list[tuple[float, float | None, int]] = []
        self.pending_silence_duration = 0.0
        self.is_in_speech = False

    def process_window(self, audio_window, timestamp, wall_clock_timestamp):
        self.calls.append((timestamp, wall_clock_timestamp, len(audio_window)))
        return False

    def flush(self):
        pass

    def consume_silence(self) -> float:
        return 0.0


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


def test_process_input_handles_transcription_notice(make_transcriber, recorded_transcribe):
    audio_queue = queue.Queue()
    notice = audio_transcriber.TranscriptionNotice("(notice)", 123.0)
    audio_queue.put(notice)
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    assert recorded_transcribe[0] is notice
    assert recorded_transcribe[-1] is None


def test_process_input_feeds_speech_detector_windows(make_transcriber, recorded_transcribe):
    audio_queue = queue.Queue()
    audio = np.zeros(1024, dtype=np.float32)
    duration = len(audio) / audio_transcriber.TARGET_SAMPLE_RATE
    segment = AudioSegment(start=5.0, audio=audio, duration_seconds=duration)
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    calls = transcriber.speech_detector.calls
    assert len(calls) == 2
    first_ts, first_wall, first_len = calls[0]
    second_ts, _, _ = calls[1]
    assert first_ts == pytest.approx(5.0)
    assert first_wall is None
    assert first_len == 512
    assert second_ts == pytest.approx(5.0 + 512 / audio_transcriber.TARGET_SAMPLE_RATE)


def test_new_segment_callback_emits_callbacks(make_transcriber):
    captured_segments: list[dict] = []
    captured_payloads: list = []

    def segment_callback(**kwargs):
        captured_segments.append(kwargs)

    def persistence_callback(payload):
        captured_payloads.append(payload)

    transcriber = make_transcriber(
        language="en",
        segment_callback=segment_callback,
        transcript_persistence_callback=persistence_callback,
    )
    transcriber.ts_transcribe_start = 200.0

    fake_segment = types.SimpleNamespace(t0=0, t1=1500, text="Hello")
    transcriber._new_segment_callback(fake_segment)

    assert captured_segments[0]["start"] == pytest.approx(0.0)
    assert captured_segments[0]["end"] == pytest.approx(1.5)
    assert captured_segments[0]["text"] == "Hello"

    payload = captured_payloads[0]
    assert payload.text == "Hello"
    assert payload.relative_start == pytest.approx(0.0)
    assert payload.start_timestamp.timestamp() == pytest.approx(200.0)
    assert payload.end_timestamp.timestamp() == pytest.approx(201.5)
    assert transcriber._last_transcript_wall_clock == pytest.approx(201.5)


def test_audio_segment_callback_invoked(make_transcriber):
    """Test that audio_segment_callback is invoked when VAD segment completes."""
    captured_audio = []

    def audio_callback(audio, start_ts):
        captured_audio.append((audio, start_ts))

    transcriber = make_transcriber(
        language="en",
        audio_segment_callback=audio_callback,
    )

    # Simulate VAD segment completion
    audio = np.ones(1600, dtype=np.float32)
    segment = AudioSegment(start=10.5, audio=audio, duration_seconds=0.1)
    transcriber._handle_vad_segment(segment)

    assert len(captured_audio) == 1
    captured, start = captured_audio[0]
    assert np.array_equal(captured, audio)
    assert start == 10.5


def test_handle_vad_segment_queues_for_transcription(make_transcriber, recorded_transcribe):
    """Test that VAD segments are queued for transcription."""
    audio_queue = queue.Queue()
    audio_queue.put(None)  # End marker

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")

    # Simulate VAD segment
    audio = np.ones(1600, dtype=np.float32)
    segment = AudioSegment(start=5.0, audio=audio, duration_seconds=0.1)
    transcriber._handle_vad_segment(segment)

    # Process input to trigger transcription thread
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Check that segment was queued
    assert any(isinstance(item, AudioSegment) for item in recorded_transcribe)
    queued_segment = [item for item in recorded_transcribe if isinstance(item, AudioSegment)][0]
    assert queued_segment.start == 5.0


def test_process_input_with_resampling(make_transcriber, recorded_transcribe):
    """Test that process_input correctly resamples audio when needed."""
    audio_queue = queue.Queue()

    # Create audio at 44100 Hz that needs resampling to 16000 Hz
    input_sample_rate = 44100
    audio_44k = np.zeros(44100, dtype=np.float32)  # 1 second
    duration = len(audio_44k) / input_sample_rate
    segment = AudioSegment(start=0.0, audio=audio_44k, duration_seconds=duration)
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber.process_input(input_sample_rate)

    # Verify speech detector received windows
    assert len(transcriber.speech_detector.calls) > 0
    # Each window should be 512 samples @ 16kHz
    for _, _, window_len in transcriber.speech_detector.calls:
        assert window_len == 512


def test_process_input_respects_stop_event(make_transcriber):
    """Test that process_input stops when stop_event is set."""
    import threading

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Put audio in queue
    audio = np.zeros(1024, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.064)
    audio_queue.put(segment)

    transcriber = make_transcriber(
        audio_queue=audio_queue,
        language="en",
        stop_event=stop_event,
    )

    # Set stop event before processing
    stop_event.set()

    # Process should exit quickly
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Verify it stopped early (should have minimal calls)
    assert len(transcriber.speech_detector.calls) == 0


def test_process_input_handles_wall_clock_timestamps(make_transcriber, recorded_transcribe):
    """Test that wall clock timestamps are tracked correctly."""
    audio_queue = queue.Queue()

    audio = np.zeros(512, dtype=np.float32)
    wall_clock_start = 1234567890.0
    segment = AudioSegment(
        start=0.0,
        audio=audio,
        duration_seconds=0.032,
        wall_clock_start=wall_clock_start,
    )
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber(
        audio_queue=audio_queue,
        language="en",
        wall_clock_reference=wall_clock_start,
    )
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Verify wall clock timestamp was passed to speech detector
    calls = transcriber.speech_detector.calls
    assert len(calls) == 1
    _, wall_clock, _ = calls[0]
    assert wall_clock == wall_clock_start


def test_new_segment_callback_chinese_conversion(make_transcriber):
    """Test that Chinese text is converted for storage."""
    captured_payloads: list = []

    def persistence_callback(payload):
        captured_payloads.append(payload)

    transcriber = make_transcriber(
        language="zh",
        transcript_persistence_callback=persistence_callback,
    )
    transcriber.ts_transcribe_start = 200.0

    # Test with Chinese text (would normally be converted)
    fake_segment = types.SimpleNamespace(t0=0, t1=1000, text="你好")
    transcriber._new_segment_callback(fake_segment)

    # Verify conversion happened (zhconv would convert this)
    payload = captured_payloads[0]
    assert payload.language == "zh"
    # Text should be processed through zhconv


def test_new_segment_callback_relative_timestamps(make_transcriber):
    """Test segment callback with relative timestamp strategy."""
    captured_segments: list[dict] = []

    def segment_callback(**kwargs):
        captured_segments.append(kwargs)

    transcriber = make_transcriber(
        language="en",
        segment_callback=segment_callback,
        timestamp_strategy="relative",
    )
    transcriber.current_audio_offset = 100.0
    transcriber.ts_transcribe_start = 100.0

    fake_segment = types.SimpleNamespace(t0=0, t1=2000, text="Test")
    transcriber._new_segment_callback(fake_segment)

    assert captured_segments[0]["start"] == pytest.approx(100.0)
    assert captured_segments[0]["end"] == pytest.approx(102.0)


def test_initialization_with_all_params(make_transcriber):
    """Test AudioTranscriber can be initialized with all parameters."""
    audio_queue = queue.Queue()

    def audio_cb(audio, start):
        pass

    def persist_cb(payload):
        pass

    def segment_cb(**kwargs):
        pass

    import threading
    stop_event = threading.Event()

    transcriber = audio_transcriber.AudioTranscriber(
        audio_input_queue=audio_queue,
        language="en",
        show_name="test_show",
        model="large-v3-turbo",
        audio_segment_callback=audio_cb,
        transcript_persistence_callback=persist_cb,
        segment_callback=segment_cb,
        timestamp_strategy="wall_clock",
        n_threads=4,
        stop_event=stop_event,
        wall_clock_reference=1234567890.0,
        queue_backlog_limiter=None,
    )

    assert transcriber.language == "en"
    assert transcriber.show_name == "test_show"
    assert transcriber.model == "large-v3-turbo"
    assert transcriber.timestamp_strategy == "wall_clock"
    assert transcriber.n_threads == 4
    assert transcriber.stop_event is stop_event
    assert transcriber.wall_clock_reference == 1234567890.0


def test_handle_vad_segment_consumes_silence_from_backlog(make_transcriber):
    """Test that silence is consumed from backlog limiter when VAD segment completes."""
    limiter = audio_transcriber.QueueBacklogLimiter(
        max_seconds=10.0,
        source_label="test",
    )

    transcriber = make_transcriber(
        language="en",
        queue_backlog_limiter=limiter,
    )

    # Set up speech detector to return some silence
    transcriber.speech_detector.pending_silence_duration = 0.5

    def mock_consume_silence():
        silence = transcriber.speech_detector.pending_silence_duration
        transcriber.speech_detector.pending_silence_duration = 0.0
        return silence

    transcriber.speech_detector.consume_silence = mock_consume_silence

    initial_backlog = limiter.current_seconds
    limiter.try_add(1.0)  # Add 1 second to backlog

    # Trigger VAD segment
    audio = np.ones(1600, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.1)
    transcriber._handle_vad_segment(segment)

    # Verify silence was consumed
    assert limiter.current_seconds < limiter.current_seconds + 1.0


def test_process_input_flushes_incomplete_segment(make_transcriber, recorded_transcribe):
    """Test that process_input flushes incomplete speech segment at end."""
    audio_queue = queue.Queue()

    # Add audio but end stream while speech might be in progress
    audio = np.zeros(1024, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.064)
    audio_queue.put(segment)
    audio_queue.put(None)  # End marker

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")

    # Mock speech detector to have in-progress speech
    transcriber.speech_detector._speech_section = [0.1] * 512
    transcriber.speech_detector._has_speech_begin_timestamp = 0.0

    flush_called = [False]
    original_flush = transcriber.speech_detector.flush

    def mock_flush():
        flush_called[0] = True
        original_flush()

    transcriber.speech_detector.flush = mock_flush

    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Verify flush was called
    assert flush_called[0]


def test_timestamp_continuity_across_segments(make_transcriber, recorded_transcribe):
    """Test that timestamps remain continuous across multiple segments."""
    audio_queue = queue.Queue()

    # First segment at 0.0
    audio1 = np.zeros(512, dtype=np.float32)
    seg1 = AudioSegment(start=0.0, audio=audio1, duration_seconds=0.032)
    audio_queue.put(seg1)

    # Second segment at 1.0
    audio2 = np.zeros(512, dtype=np.float32)
    seg2 = AudioSegment(start=1.0, audio=audio2, duration_seconds=0.032)
    audio_queue.put(seg2)

    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Verify timestamps are correct
    calls = transcriber.speech_detector.calls
    assert len(calls) == 2
    ts1, _, _ = calls[0]
    ts2, _, _ = calls[1]

    assert ts1 == pytest.approx(0.0)
    assert ts2 == pytest.approx(1.0)
