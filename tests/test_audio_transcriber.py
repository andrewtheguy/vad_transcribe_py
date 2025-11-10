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
        self._script: list[tuple[bool, bool]] = []
        self._script_index = 0

    def process_window(self, audio_window, timestamp, wall_clock_timestamp):
        self.calls.append((timestamp, wall_clock_timestamp, len(audio_window)))
        if self._script_index < len(self._script):
            has_speech, in_speech = self._script[self._script_index]
            self._script_index += 1
            self.is_in_speech = in_speech
            if not has_speech and not in_speech:
                self.pending_silence_duration = len(audio_window) / self.sample_rate
            else:
                self.pending_silence_duration = 0.0
            return has_speech
        return False

    def flush(self):
        pass

    def consume_silence(self) -> float:
        return 0.0

    def set_script(self, script: list[tuple[bool, bool]]) -> None:
        self._script = script
        self._script_index = 0


class StaticTimestampLimiter:
    def __init__(self, pending_timestamp: float):
        self.pending_timestamp = pending_timestamp

    def pending_chunk_start_timestamp(self):
        return self.pending_timestamp

    def note_timestamp_progress(self, _duration):
        pass

    def consume(self, _duration):
        pass

    def register_drop_callback(self, _callback):
        pass


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
    segment = AudioSegment(start=5.0, audio=audio, wall_clock_start=1000.0, duration_seconds=duration)
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    calls = transcriber.speech_detector.calls
    assert len(calls) == 2
    first_ts, first_wall, first_len = calls[0]
    second_ts, _, _ = calls[1]
    assert first_ts == pytest.approx(5.0)
    assert first_wall == pytest.approx(1000.0)  # Now always provided
    assert first_len == 512
    assert second_ts == pytest.approx(5.0 + 512 / audio_transcriber.TARGET_SAMPLE_RATE)


def test_drop_notice_waits_until_silence_release(monkeypatch, make_transcriber, recorded_transcribe):
    window_samples = audio_transcriber.get_window_size_samples()
    audio = np.zeros(window_samples * 3, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, wall_clock_start=1000.0, duration_seconds=len(audio) / audio_transcriber.TARGET_SAMPLE_RATE)

    audio_queue = queue.Queue()
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber.speech_detector.set_script([
        (True, True),
        (False, True),
        (False, False),
    ])
    transcriber._handle_drop_notice(5.0)

    release_calls = []
    original_release = audio_transcriber.AudioTranscriber._release_ready_drop_notices

    def recording_release(self, ts):
        release_calls.append((len(self.speech_detector.calls), ts, self.speech_detector.is_in_speech))
        return original_release(self, ts)

    monkeypatch.setattr(audio_transcriber.AudioTranscriber, "_release_ready_drop_notices", recording_release)

    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    assert release_calls, "expected drop release attempts"
    first_call = release_calls[0]
    assert first_call[0] == 3  # release occurs after three windows processed
    assert first_call[2] is False
    notices = [item for item in recorded_transcribe if isinstance(item, audio_transcriber.TranscriptionNotice)]
    assert len(notices) == 1


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


def test_drop_notice_positions_between_segments(make_transcriber, recorded_transcribe):
    audio_queue = queue.Queue()
    audio_queue.put(None)

    transcriber = make_transcriber(audio_queue=audio_queue, language="en")
    transcriber._handle_drop_notice(25.0)

    early_audio = np.ones(1600, dtype=np.float32)
    late_audio = np.ones(1600, dtype=np.float32)
    seg_early = AudioSegment(start=0.0, audio=early_audio, duration_seconds=0.1, wall_clock_start=20.0)
    seg_late = AudioSegment(start=5.0, audio=late_audio, duration_seconds=0.1, wall_clock_start=30.0)

    transcriber._handle_vad_segment(seg_early)
    transcriber._handle_vad_segment(seg_late)

    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    queue_order = [type(item) for item in recorded_transcribe[:-1]]  # drop final sentinel
    assert queue_order.count(AudioSegment) == 2
    notice_index = next(i for i, item in enumerate(recorded_transcribe) if isinstance(item, audio_transcriber.TranscriptionNotice))
    early_index = next(i for i, item in enumerate(recorded_transcribe) if item is seg_early)
    late_index = next(i for i, item in enumerate(recorded_transcribe) if item is seg_late)
    assert early_index < notice_index < late_index


def test_segments_after_drop_use_notice_timestamp(make_transcriber):
    drop_ts = 200.0
    limiter = StaticTimestampLimiter(pending_timestamp=drop_ts - 30.0)
    transcriber = make_transcriber(language="en", queue_backlog_limiter=limiter, wall_clock_reference=0.0)
    transcriber._handle_drop_notice(drop_ts)

    window_samples = audio_transcriber.get_window_size_samples()
    audio = np.zeros(window_samples, dtype=np.float32)
    # Use drop_ts as wall_clock_start to maintain test intent - segments now provide real timestamps
    segment = AudioSegment(start=0.0, audio=audio, wall_clock_start=drop_ts, duration_seconds=len(audio) / audio_transcriber.TARGET_SAMPLE_RATE)

    audio_queue = transcriber.audio_input_queue
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    assert transcriber.speech_detector.calls, "expected windows"
    _, wall_clock, _ = transcriber.speech_detector.calls[0]
    assert wall_clock == pytest.approx(drop_ts)


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
    segment = AudioSegment(start=10.5, audio=audio, wall_clock_start=1010.5, duration_seconds=0.1)
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
    segment = AudioSegment(start=5.0, audio=audio, wall_clock_start=1005.0, duration_seconds=0.1)
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
    segment = AudioSegment(start=0.0, audio=audio_44k, wall_clock_start=1000.0, duration_seconds=duration)
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
    segment = AudioSegment(start=0.0, audio=audio, wall_clock_start=1000.0, duration_seconds=0.064)
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
    segment = AudioSegment(start=0.0, audio=audio, wall_clock_start=1000.0, duration_seconds=0.1)
    transcriber._handle_vad_segment(segment)

    # Verify silence was consumed
    assert limiter.current_seconds < limiter.current_seconds + 1.0


def test_process_input_flushes_incomplete_segment(make_transcriber, recorded_transcribe):
    """Test that process_input flushes incomplete speech segment at end."""
    audio_queue = queue.Queue()

    # Add audio but end stream while speech might be in progress
    audio = np.zeros(1024, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, wall_clock_start=1000.0, duration_seconds=0.064)
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
    seg1 = AudioSegment(start=0.0, audio=audio1, wall_clock_start=1000.0, duration_seconds=0.032)
    audio_queue.put(seg1)

    # Second segment at 1.0
    audio2 = np.zeros(512, dtype=np.float32)
    seg2 = AudioSegment(start=1.0, audio=audio2, wall_clock_start=1001.0, duration_seconds=0.032)
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


# ============================================================================
# Tests for stream_url_thread
# ============================================================================


def test_stream_url_thread_basic_operation(monkeypatch):
    """Test basic streaming and queuing of audio chunks."""
    import threading
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Create fake PCM data (1 second worth)
    fake_pcm = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE, dtype=np.int16).tobytes()

    # Mock stream_url context manager
    mock_stdout = MagicMock()
    read_sequence = [fake_pcm, fake_pcm]  # Two chunks
    read_count = [0]

    def mock_read(size):
        if read_count[0] < len(read_sequence):
            result = read_sequence[read_count[0]]
            read_count[0] += 1
            return result
        return b''  # EOF after sequence

    mock_stdout.read.side_effect = mock_read

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stdout)
    mock_stream.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(audio_transcriber, "stream_url", lambda url: mock_stream)

    # Run in thread with timeout
    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event}
    )
    thread.start()

    # Wait for EOF and retry, then stop
    import time
    time.sleep(0.5)
    stop_event.set()
    thread.join(timeout=2.0)

    # Verify chunks were queued
    assert audio_queue.qsize() >= 2
    segment = audio_queue.get()
    assert isinstance(segment, AudioSegment)
    assert segment.start == 0.0
    assert segment.duration_seconds == pytest.approx(1.0)

    # Second segment should have incremented timestamp
    segment2 = audio_queue.get()
    assert segment2.start == pytest.approx(1.0)


def test_stream_url_thread_stop_event(monkeypatch):
    """Test that stop_event terminates the thread."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Mock to keep streaming indefinitely
    mock_stdout = MagicMock()
    fake_pcm = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE, dtype=np.int16).tobytes()

    def blocking_read(size):
        # Check stop event during read
        if stop_event.is_set():
            return b''
        time.sleep(0.05)  # Small delay to simulate streaming
        return fake_pcm

    mock_stdout.read.side_effect = blocking_read

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stdout)
    mock_stream.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(audio_transcriber, "stream_url", lambda url: mock_stream)

    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event}
    )
    thread.start()

    # Let it run briefly then stop
    time.sleep(0.1)
    stop_event.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()


def test_stream_url_thread_queue_limiter_drops(monkeypatch):
    """Test that queue limiter drops chunks when backlog exceeds limit."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=0.5, source_label="test")

    # Fill limiter to trigger drops (set high backlog)
    limiter.try_add(1.0)  # Over limit of 0.5

    fake_pcm = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE, dtype=np.int16).tobytes()
    mock_stdout = MagicMock()
    read_sequence = [fake_pcm, fake_pcm, fake_pcm]  # 3 chunks to ensure some are dropped
    read_count = [0]

    def mock_read(size):
        if read_count[0] < len(read_sequence):
            result = read_sequence[read_count[0]]
            read_count[0] += 1
            return result
        return b''  # EOF after sequence

    mock_stdout.read.side_effect = mock_read

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stdout)
    mock_stream.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(audio_transcriber, "stream_url", lambda url: mock_stream)

    initial_dropped = limiter.dropped_seconds

    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event, "queue_limiter": limiter}
    )
    thread.start()
    time.sleep(0.3)
    stop_event.set()
    thread.join(timeout=2.0)

    # Some chunks should be dropped (limiter tracks dropped seconds)
    assert limiter.dropped_seconds > initial_dropped


def test_stream_url_thread_queue_limiter_accepts_when_below_limit(monkeypatch):
    """Test that queue limiter accepts chunks when under limit."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    limiter = audio_transcriber.QueueBacklogLimiter(max_seconds=10.0, source_label="test")

    fake_pcm = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE, dtype=np.int16).tobytes()
    mock_stdout = MagicMock()
    read_sequence = [fake_pcm, fake_pcm]
    read_count = [0]

    def mock_read(size):
        if read_count[0] < len(read_sequence):
            result = read_sequence[read_count[0]]
            read_count[0] += 1
            return result
        return b''  # EOF after sequence

    mock_stdout.read.side_effect = mock_read

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stdout)
    mock_stream.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(audio_transcriber, "stream_url", lambda url: mock_stream)

    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event, "queue_limiter": limiter}
    )
    thread.start()
    time.sleep(0.5)
    stop_event.set()
    thread.join(timeout=2.0)

    # Both audio chunks should be queued (may also have error notices from retry)
    items = []
    while not audio_queue.empty():
        items.append(audio_queue.get())

    audio_segments = [item for item in items if isinstance(item, AudioSegment)]
    assert len(audio_segments) == 2
    assert limiter.dropped_seconds == 0


def test_stream_url_thread_retry_on_error(monkeypatch):
    """Test retry logic when stream fails."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    call_count = [0]

    def mock_stream_url_failing(url):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call fails
            raise ValueError("Stream error")
        else:
            # Second call succeeds then EOF
            mock_stdout = MagicMock()
            mock_stdout.read.return_value = b''
            mock_stream = MagicMock()
            mock_stream.__enter__ = Mock(return_value=mock_stdout)
            mock_stream.__exit__ = Mock(return_value=False)
            return mock_stream

    monkeypatch.setattr(audio_transcriber, "stream_url", mock_stream_url_failing)

    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event}
    )
    thread.start()

    # Wait for retry
    time.sleep(1.0)
    stop_event.set()
    thread.join(timeout=2.0)

    # Should have retried after error
    assert call_count[0] >= 2
    # Should have queued error notice
    items = []
    while not audio_queue.empty():
        items.append(audio_queue.get())
    assert any(isinstance(item, audio_transcriber.TranscriptionNotice) for item in items)


def test_stream_url_thread_timestamp_continuity(monkeypatch):
    """Test that timestamps increment correctly across chunks."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Create chunks of different sizes
    chunk1 = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE, dtype=np.int16).tobytes()  # 1.0 sec
    chunk2 = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE * 2, dtype=np.int16).tobytes()  # 2.0 sec

    mock_stdout = MagicMock()
    read_sequence = [chunk1, chunk2]
    read_count = [0]

    def mock_read(size):
        if read_count[0] < len(read_sequence):
            result = read_sequence[read_count[0]]
            read_count[0] += 1
            return result
        return b''  # EOF after sequence

    mock_stdout.read.side_effect = mock_read

    mock_stream = MagicMock()
    mock_stream.__enter__ = Mock(return_value=mock_stdout)
    mock_stream.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(audio_transcriber, "stream_url", lambda url: mock_stream)

    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event}
    )
    thread.start()
    time.sleep(0.5)
    stop_event.set()
    thread.join(timeout=2.0)

    # Verify timestamp progression
    seg1 = audio_queue.get()
    seg2 = audio_queue.get()

    assert seg1.start == pytest.approx(0.0)
    assert seg1.duration_seconds == pytest.approx(1.0)

    assert seg2.start == pytest.approx(1.0)  # Continues from previous
    assert seg2.duration_seconds == pytest.approx(2.0)


def test_stream_url_thread_eof_handling(monkeypatch):
    """Test that EOF properly terminates the stream and triggers retry."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    call_count = [0]

    def mock_stream_url_with_eof(url):
        call_count[0] += 1
        mock_stdout = MagicMock()
        if call_count[0] == 1:
            # First stream: one chunk then EOF
            fake_pcm = np.zeros(audio_transcriber.TARGET_SAMPLE_RATE, dtype=np.int16).tobytes()
            read_count = [0]
            read_sequence = [fake_pcm]

            def mock_read(size):
                if read_count[0] < len(read_sequence):
                    result = read_sequence[read_count[0]]
                    read_count[0] += 1
                    return result
                return b''

            mock_stdout.read.side_effect = mock_read
        else:
            # Second stream: immediate EOF (will stop with stop_event)
            mock_stdout.read.return_value = b''

        mock_stream = MagicMock()
        mock_stream.__enter__ = Mock(return_value=mock_stdout)
        mock_stream.__exit__ = Mock(return_value=False)
        return mock_stream

    monkeypatch.setattr(audio_transcriber, "stream_url", mock_stream_url_with_eof)

    thread = threading.Thread(
        target=audio_transcriber.stream_url_thread,
        args=("fake://url", audio_queue),
        kwargs={"stop_event": stop_event}
    )
    thread.start()

    # Wait for first stream to complete and retry
    time.sleep(1.0)
    stop_event.set()
    thread.join(timeout=2.0)

    # Should have attempted retry after EOF
    assert call_count[0] >= 2


def test_stream_url_thread_stop_during_retry_wait(monkeypatch):
    """Test that stop_event is checked during retry sleep."""
    import threading
    import time
    from unittest.mock import Mock, MagicMock, patch

    audio_queue = queue.Queue()
    stop_event = threading.Event()

    # Always raise error to trigger retry
    def mock_stream_url_error(url):
        raise ValueError("Persistent error")

    monkeypatch.setattr(audio_transcriber, "stream_url", mock_stream_url_error)

    # Mock sleep to track if it's called
    sleep_called = [False]
    original_sleep = time.sleep

    def tracked_sleep(duration):
        sleep_called[0] = True
        # Use shorter sleep for testing
        original_sleep(0.1)

    with patch('whisper_transcribe_py.audio_transcriber.sleep', tracked_sleep):
        thread = threading.Thread(
            target=audio_transcriber.stream_url_thread,
            args=("fake://url", audio_queue),
            kwargs={"stop_event": stop_event}
        )
        thread.start()

        # Let it fail and start retry
        time.sleep(0.2)
        stop_event.set()
        thread.join(timeout=2.0)

        # Should have entered retry logic
        assert sleep_called[0]
        assert not thread.is_alive()
