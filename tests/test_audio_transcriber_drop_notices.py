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
        self.pending_non_speech_duration = 0.0
        self.is_in_speech = False

    def process_window(self, *_, **__):
        return False

    def flush(self) -> None:
        pass

    def consume_non_speech(self) -> float:
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


@pytest.fixture
def recorded_transcribe(monkeypatch):
    """Fixture to capture items sent to transcribe queue."""
    recorded: list = []

    def fake_transcribe(self):
        while True:
            item = self.transcribe_queue.get()
            recorded.append(item)
            if item is None:
                break

    monkeypatch.setattr(audio_transcriber.AudioTranscriber, "_transcribe", fake_transcribe)
    return recorded


def test_notice_at_beginning_before_segments(make_transcriber, recorded_transcribe):
    """Test TranscriptionNotice at the beginning before any audio segments."""
    window_samples = audio_transcriber.get_window_size_samples()
    audio = np.zeros(window_samples, dtype=np.float32)
    segment = AudioSegment(start=10.0, audio=audio, wall_clock_start=1000.0, duration_seconds=len(audio) / audio_transcriber.TARGET_SAMPLE_RATE)

    audio_queue = queue.Queue()
    # Put notice first, then segment
    notice = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 5.0)
    audio_queue.put(notice)
    audio_queue.put(segment)
    audio_queue.put(None)

    transcriber = make_transcriber()
    transcriber.audio_input_queue = audio_queue
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Notice should appear first in transcribe queue
    assert len(recorded_transcribe) >= 2
    assert isinstance(recorded_transcribe[0], audio_transcriber.TranscriptionNotice)
    assert recorded_transcribe[0].timestamp == pytest.approx(5.0)


def test_notice_at_end_after_segments(make_transcriber, recorded_transcribe):
    """Test TranscriptionNotice at the end after all audio segments."""
    window_samples = audio_transcriber.get_window_size_samples()
    audio = np.zeros(window_samples, dtype=np.float32)
    segment = AudioSegment(start=0.0, audio=audio, wall_clock_start=1000.0, duration_seconds=len(audio) / audio_transcriber.TARGET_SAMPLE_RATE)

    audio_queue = queue.Queue()
    # Put segment first, then notice at end
    audio_queue.put(segment)
    notice = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 15.0)
    audio_queue.put(notice)
    audio_queue.put(None)

    transcriber = make_transcriber()
    transcriber.audio_input_queue = audio_queue
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Find notice in recorded items (should be after segment but before None)
    notices = [item for item in recorded_transcribe if isinstance(item, audio_transcriber.TranscriptionNotice)]
    assert len(notices) == 1
    assert notices[0].timestamp == pytest.approx(15.0)


def test_multiple_consecutive_notices(make_transcriber, recorded_transcribe):
    """Test multiple consecutive TranscriptionNotices without segments between them."""
    audio_queue = queue.Queue()

    # Put multiple notices consecutively
    notice1 = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 10.0)
    notice2 = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 20.0)
    notice3 = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 30.0)

    audio_queue.put(notice1)
    audio_queue.put(notice2)
    audio_queue.put(notice3)
    audio_queue.put(None)

    transcriber = make_transcriber()
    transcriber.audio_input_queue = audio_queue
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # All notices should be in transcribe queue in order
    notices = [item for item in recorded_transcribe if isinstance(item, audio_transcriber.TranscriptionNotice)]
    assert len(notices) == 3
    assert notices[0].timestamp == pytest.approx(10.0)
    assert notices[1].timestamp == pytest.approx(20.0)
    assert notices[2].timestamp == pytest.approx(30.0)


def test_alternating_segments_and_notices(make_transcriber, recorded_transcribe):
    """Test alternating pattern of segments and notices."""
    window_samples = audio_transcriber.get_window_size_samples()
    audio1 = np.zeros(window_samples, dtype=np.float32)
    audio2 = np.zeros(window_samples, dtype=np.float32)

    seg1 = AudioSegment(start=0.0, audio=audio1, wall_clock_start=1000.0, duration_seconds=len(audio1) / audio_transcriber.TARGET_SAMPLE_RATE)
    notice1 = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 5.0)
    seg2 = AudioSegment(start=10.0, audio=audio2, wall_clock_start=1010.0, duration_seconds=len(audio2) / audio_transcriber.TARGET_SAMPLE_RATE)
    notice2 = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 15.0)

    audio_queue = queue.Queue()
    audio_queue.put(seg1)
    audio_queue.put(notice1)
    audio_queue.put(seg2)
    audio_queue.put(notice2)
    audio_queue.put(None)

    transcriber = make_transcriber()
    transcriber.audio_input_queue = audio_queue
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Verify notices are in transcribe queue in correct order
    notices = [item for item in recorded_transcribe if isinstance(item, audio_transcriber.TranscriptionNotice)]
    assert len(notices) == 2, f"Expected 2 notices, got {len(notices)}"

    # Find indices to verify order
    notice1_idx = recorded_transcribe.index(notice1)
    notice2_idx = recorded_transcribe.index(notice2)

    assert notice1_idx < notice2_idx, "Notices should appear in order"


def test_notice_triggers_reset(make_transcriber, recorded_transcribe, monkeypatch):
    """Test that TranscriptionNotice triggers SpeechDetector reset."""
    window_samples = audio_transcriber.get_window_size_samples()
    audio = np.zeros(window_samples * 2, dtype=np.float32)

    seg1 = AudioSegment(start=0.0, audio=audio, wall_clock_start=1000.0, duration_seconds=len(audio) / audio_transcriber.TARGET_SAMPLE_RATE)
    notice = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 5.0)
    seg2 = AudioSegment(start=10.0, audio=audio, wall_clock_start=1010.0, duration_seconds=len(audio) / audio_transcriber.TARGET_SAMPLE_RATE)

    audio_queue = queue.Queue()
    audio_queue.put(seg1)
    audio_queue.put(notice)
    audio_queue.put(seg2)
    audio_queue.put(None)

    # Track reset calls
    reset_called = []
    original_reset = DummySpeechDetector.reset
    def tracking_reset(self):
        reset_called.append(True)
        original_reset(self)

    monkeypatch.setattr(DummySpeechDetector, "reset", tracking_reset)

    transcriber = make_transcriber()
    transcriber.audio_input_queue = audio_queue
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Reset should have been called when notice was processed
    assert len(reset_called) > 0, "Expected SpeechDetector.reset() to be called when TranscriptionNotice is processed"


def test_notice_only_queue(make_transcriber, recorded_transcribe):
    """Test queue containing only a TranscriptionNotice (no audio segments)."""
    audio_queue = queue.Queue()
    notice = audio_transcriber.TranscriptionNotice("(transcript temporarily dropped)", 10.0)
    audio_queue.put(notice)
    audio_queue.put(None)

    transcriber = make_transcriber()
    transcriber.audio_input_queue = audio_queue
    transcriber.process_input(audio_transcriber.TARGET_SAMPLE_RATE)

    # Should have notice in transcribe queue
    notices = [item for item in recorded_transcribe if isinstance(item, audio_transcriber.TranscriptionNotice)]
    assert len(notices) == 1
    assert notices[0].timestamp == pytest.approx(10.0)
