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

    def reset(self) -> None:
        pass


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


def test_drop_notice_put_in_audio_input_queue(make_transcriber):
    """Test that drop notice is put directly into audio_input_queue."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(10.0)

    # Notice should be in audio_input_queue
    notice = transcriber.audio_input_queue.get_nowait()
    assert isinstance(notice, audio_transcriber.TranscriptionNotice)
    assert notice.timestamp == pytest.approx(10.0)


def test_drop_notice_timestamp_never_regresses(make_transcriber):
    """Test that drop notice timestamp is adjusted to maintain monotonicity."""
    transcriber = make_transcriber()
    transcriber._last_transcript_wall_clock = 100.0

    # When drop occurs at 10.0, it should be adjusted to 100.0
    transcriber._handle_drop_notice(10.0)

    # Check notice in audio_input_queue
    notice = transcriber.audio_input_queue.get_nowait()
    assert notice.timestamp == pytest.approx(100.0)


def test_multiple_drop_notices_queued_in_order(make_transcriber):
    """Test that multiple drop notices are queued in audio_input_queue in order."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(5.0)
    transcriber._handle_drop_notice(8.0)

    # Both notices should be in audio_input_queue in order
    first = transcriber.audio_input_queue.get_nowait()
    second = transcriber.audio_input_queue.get_nowait()

    assert isinstance(first, audio_transcriber.TranscriptionNotice)
    assert first.timestamp == pytest.approx(5.0)
    assert isinstance(second, audio_transcriber.TranscriptionNotice)
    assert second.timestamp == pytest.approx(8.0)
