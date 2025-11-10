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
