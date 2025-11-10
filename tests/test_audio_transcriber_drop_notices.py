import queue
import sys
import types
from typing import Callable, Optional

import numpy as np
import pytest

# only relevant for converting Chinese simplified/traditional, not needed for tests
if "zhconv_rs" not in sys.modules:
    stub_module = types.ModuleType("zhconv_rs")

    def _identity(text: str, _locale: str) -> str:
        return text

    stub_module.zhconv = _identity  # type: ignore[attr-defined]
    sys.modules["zhconv_rs"] = stub_module

if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")

    def _from_numpy(array):  # pragma: no cover - helper for import-time stub
        return array

    torch_module.from_numpy = _from_numpy  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_module

if "silero_vad" not in sys.modules:
    silero_module = types.ModuleType("silero_vad")

    class _DummyPrediction:
        def item(self) -> float:
            return 0.0

    class _DummyVadModel:
        def __call__(self, *_args, **_kwargs):
            return _DummyPrediction()

        def reset_states(self) -> None:
            pass

    def _load_silero_vad():
        return _DummyVadModel()

    silero_module.load_silero_vad = _load_silero_vad  # type: ignore[attr-defined]
    sys.modules["silero_vad"] = silero_module

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
