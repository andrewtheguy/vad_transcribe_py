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


def test_drop_notice_stored_until_segment_complete(make_transcriber):
    """Test that drop notice is stored but not emitted until segment completes."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(10.0)

    # Notice should be pending but not yet in transcribe queue
    assert transcriber.transcribe_queue.empty()
    assert transcriber._pending_drop_notice is not None
    assert transcriber._pending_drop_notice.timestamp == pytest.approx(10.0)

    # Emit the pending notice when segment completes
    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=12.0)
    transcriber._handle_vad_segment(segment)

    # Now both segment and notice should be in queue (segment first, then notice)
    first = transcriber.transcribe_queue.get_nowait()
    second = transcriber.transcribe_queue.get_nowait()

    assert first is segment
    assert isinstance(second, audio_transcriber.TranscriptionNotice)
    assert second.timestamp == pytest.approx(10.0)


def test_drop_notice_emitted_after_segment(make_transcriber):
    """Test that drop notice is emitted after segment in new flow."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(42.0)

    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=45.0)

    transcriber._handle_vad_segment(segment)

    # New flow: segment first, then notice
    first = transcriber.transcribe_queue.get_nowait()
    second = transcriber.transcribe_queue.get_nowait()

    assert first is segment
    assert isinstance(second, audio_transcriber.TranscriptionNotice)
    assert second.timestamp == pytest.approx(42.0)


def test_drop_notice_timestamp_never_regresses(make_transcriber):
    """Test that drop notice timestamp is adjusted to maintain monotonicity."""
    transcriber = make_transcriber()
    transcriber._last_transcript_wall_clock = 100.0

    # When drop occurs at 10.0, it should be adjusted to 100.0
    transcriber._handle_drop_notice(10.0)

    # Emit by triggering segment complete
    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=105.0)
    transcriber._handle_vad_segment(segment)

    _segment = transcriber.transcribe_queue.get_nowait()
    notice = transcriber.transcribe_queue.get_nowait()

    assert notice.timestamp == pytest.approx(100.0)


def test_single_pending_notice_overwrites_previous(make_transcriber):
    """Test that only one drop notice is stored at a time (latest overwrites)."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(5.0)
    transcriber._handle_drop_notice(8.0)

    # Only the latest notice should be pending
    assert transcriber._pending_drop_notice is not None
    assert transcriber._pending_drop_notice.timestamp == pytest.approx(8.0)

    # Emit by triggering segment complete
    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=10.0)
    transcriber._handle_vad_segment(segment)

    _segment = transcriber.transcribe_queue.get_nowait()
    notice = transcriber.transcribe_queue.get_nowait()

    # Should get only the second notice (8.0), not the first
    assert notice.timestamp == pytest.approx(8.0)
    assert transcriber.transcribe_queue.empty()


def test_segment_with_wall_clock_emits_pending_notice(make_transcriber):
    """Test that segment with wall_clock emits pending notice after segment."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(12.0)

    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=1000.0)

    transcriber._handle_vad_segment(segment)

    # New flow: segment first, then notice
    first = transcriber.transcribe_queue.get_nowait()
    second = transcriber.transcribe_queue.get_nowait()

    assert first is segment
    assert isinstance(second, audio_transcriber.TranscriptionNotice)
    assert second.timestamp == pytest.approx(12.0)


def test_emit_pending_notice_clears_state(make_transcriber):
    """Test that emitting pending notice clears the stored notice."""
    transcriber = make_transcriber()
    transcriber._handle_drop_notice(3.0)

    assert transcriber._pending_drop_notice is not None

    # Trigger emission
    audio = np.zeros(160, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, duration_seconds=0.01, wall_clock_start=5.0)
    transcriber._handle_vad_segment(segment)

    # Pending notice should be cleared after emission
    assert transcriber._pending_drop_notice is None

    # Queue should have segment and notice
    _segment = transcriber.transcribe_queue.get_nowait()
    notice = transcriber.transcribe_queue.get_nowait()
    assert notice.timestamp == pytest.approx(3.0)
    assert transcriber.transcribe_queue.empty()
