import queue
from typing import Callable, Optional

import numpy as np
import pytest

import whisper_transcribe_py.audio_transcriber as audio_transcriber
from whisper_transcribe_py.vad_processor import AudioSegment


class DummySpeechDetector:
    def __init__(
        self,
        sample_rate: int,
        on_segment_complete: Optional[Callable[[AudioSegment], None]] = None,
        **_: dict,
    ) -> None:
        self.sample_rate = sample_rate
        self.on_segment_complete = on_segment_complete
        self.pending_silence_duration = 0.0
        self.is_in_speech = False

    def process_window(self, *_, **__):
        return False

    def flush(self) -> None:
        pass

    def consume_silence(self) -> float:
        return 0.0


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    monkeypatch.setattr(audio_transcriber, "SpeechDetector", DummySpeechDetector)
    monkeypatch.setattr(audio_transcriber.AudioTranscriber, "_load_whisper_cpp", lambda self: None)


@pytest.fixture
def make_transcriber():
    def _builder():
        return audio_transcriber.AudioTranscriber(
            audio_input_queue=queue.Queue(),
            language="en",
        )

    return _builder


def test_drop_notice_waits_until_wall_clock(make_transcriber):
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(10.0)

    transcriber._release_ready_drop_notices(5.0)
    assert transcriber.transcribe_queue.empty()

    transcriber._release_ready_drop_notices(12.0)
    emitted = transcriber.transcribe_queue.get_nowait()
    assert isinstance(emitted, audio_transcriber.TranscriptionNotice)
    assert emitted.timestamp == pytest.approx(10.0)


def test_drop_notice_emitted_before_segment_crossing_threshold(make_transcriber):
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(42.0)

    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=45.0)

    transcriber._handle_vad_segment(segment)

    first = transcriber.transcribe_queue.get_nowait()
    second = transcriber.transcribe_queue.get_nowait()

    assert isinstance(first, audio_transcriber.TranscriptionNotice)
    assert first.timestamp == pytest.approx(42.0)
    assert second is segment


def test_drop_notice_timestamp_never_regresses(make_transcriber):
    transcriber = make_transcriber()
    transcriber._last_transcript_wall_clock = 100.0

    transcriber._handle_drop_notice(10.0)
    transcriber._flush_pending_drop_notices()
    notice = transcriber.transcribe_queue.get_nowait()

    assert notice.timestamp == pytest.approx(100.0)


def test_multiple_drop_notices_release_in_order(make_transcriber):
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(5.0)
    transcriber._handle_drop_notice(8.0)

    transcriber._release_ready_drop_notices(6.0)

    first = transcriber.transcribe_queue.get_nowait()
    assert first.timestamp == pytest.approx(5.0)
    assert transcriber.transcribe_queue.empty()

    transcriber._release_ready_drop_notices(9.0)
    second = transcriber.transcribe_queue.get_nowait()
    assert second.timestamp == pytest.approx(8.0)


def test_segment_without_wall_clock_flushes_pending_notice(make_transcriber):
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(12.0)

    audio = np.zeros(160, dtype=np.float32)
    # wall_clock_start is now required for all segments
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=1000.0)

    transcriber._handle_vad_segment(segment)

    first = transcriber.transcribe_queue.get_nowait()
    second = transcriber.transcribe_queue.get_nowait()

    assert isinstance(first, audio_transcriber.TranscriptionNotice)
    assert first.timestamp == pytest.approx(12.0)
    assert second is segment


def test_flush_pending_drop_notices_drains_queue(make_transcriber):
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(3.0)
    transcriber._handle_drop_notice(4.0)

    transcriber._flush_pending_drop_notices()

    timestamps = [transcriber.transcribe_queue.get_nowait().timestamp for _ in range(2)]
    assert timestamps == pytest.approx([3.0, 4.0])
