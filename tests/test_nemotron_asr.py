import sys
from types import ModuleType

import numpy as np
import pytest
import torch

import vad_transcribe_py.audio_transcriber as audio_transcriber
from vad_transcribe_py.backends.nemotron_asr import (
    NEMOTRON_ASR_DEFAULT_MODEL,
    NemotronASRBackend,
    resolve_language,
    split_sentences,
    timed_tokens,
)
from vad_transcribe_py.vad_processor import (
    NEMOTRON_ASR_HARD_LIMIT_SECONDS,
    NEMOTRON_ASR_SOFT_LIMIT_SECONDS,
)

BLANK = 0
PROMPTS = {"en": 0, "en-US": 0, "de": 9, "zh-CN": 4, "zh-TW": 5, "ja-JP": 10, "auto": 101}


@pytest.fixture()
def stub_nemotron(monkeypatch):
    """Stub out NemotronASRBackend model loading and device detection."""
    monkeypatch.setattr(NemotronASRBackend, "_load_model", lambda _self: None)
    monkeypatch.setattr(
        "vad_transcribe_py.backends.nemotron_asr._detect_device", lambda _device=None: "cpu"
    )


def test_resolve_language():
    assert resolve_language(None, PROMPTS) == "auto"
    assert resolve_language("en", PROMPTS) == "en"
    assert resolve_language("de", PROMPTS) == "de"
    assert resolve_language("zh", PROMPTS) == "zh-CN"
    assert resolve_language("ja", PROMPTS) == "ja-JP"
    with pytest.raises(ValueError, match="Unrecognized language code"):
        resolve_language("xx", PROMPTS)


def test_timed_tokens_drops_blanks_and_tracks_frames():
    # blank advances one frame; emitted tokens do not advance.
    sequence = [BLANK, BLANK, 5, BLANK, 6, 7, BLANK]
    durations = [1, 1, 0, 1, 0, 0, 1, 1]  # one extra trailing step, as the model emits
    pieces = ["<pad>", "<pad>", "▁Hi", "<pad>", "▁there", ".", "<pad>"]
    tokens = timed_tokens(sequence, durations, {BLANK}, pieces)
    assert [(t.token_id, t.frame) for t in tokens] == [(5, 2), (6, 3), (7, 3)]


def test_split_sentences_on_terminal_punctuation():
    sequence = [1, 2, 3, 4, 5, 6, 7, 8]
    pieces = ["▁Hi", ".", "▁It", "▁is", "▁3", ".", "5", "?"]
    tokens = timed_tokens(sequence, [0] * len(sequence), {BLANK}, pieces)
    sentences = split_sentences(tokens)
    assert [[t.piece for t in s] for s in sentences] == [
        ["▁Hi", "."],
        ["▁It", "▁is", "▁3", ".", "5", "?"],
    ]


def test_split_sentences_keeps_unterminated_tail():
    tokens = timed_tokens([1, 2, 3], [0, 0, 0], {BLANK}, ["▁Done", ".", "▁and"])
    assert [[t.piece for t in s] for s in split_sentences(tokens)] == [["▁Done", "."], ["▁and"]]


def test_nemotron_defaults(stub_nemotron):
    backend = NemotronASRBackend(language="en")
    assert backend.model == NEMOTRON_ASR_DEFAULT_MODEL
    assert backend.hard_limit_seconds == NEMOTRON_ASR_HARD_LIMIT_SECONDS
    assert backend.soft_limit_seconds == NEMOTRON_ASR_SOFT_LIMIT_SECONDS


def test_create_transcriber_nemotron(stub_nemotron):
    transcriber = audio_transcriber.create_transcriber(
        language="en", backend="nemotron-asr", sub_timestamps=False
    )
    assert isinstance(transcriber, NemotronASRBackend)
    assert transcriber._sub_timestamps is False


def test_create_transcriber_nemotron_rejects_condition(stub_nemotron):
    with pytest.raises(ValueError, match="condition=True is not supported"):
        audio_transcriber.create_transcriber(language="en", backend="nemotron-asr", condition=True)


class _StubInputs(dict):
    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self


class _StubTokenizer:
    VOCAB = {0: "<blank>", 1: "▁Hello", 2: ".", 3: "▁Bye", 4: "!", 98: "<pad>", 99: "<en-US>"}

    def convert_ids_to_tokens(self, ids):
        return [self.VOCAB[i] for i in ids]

    def decode(self, ids, skip_special_tokens=False):
        pieces = [self.VOCAB[i] for i in ids if not (skip_special_tokens and i in (0, 98, 99))]
        return "".join(pieces).replace("▁", " ")


class _StubProcessor:
    prompt_dictionary = PROMPTS
    blank_token_id = BLANK

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.calls: list[dict[str, object]] = []

    def __call__(self, audio, *, sampling_rate, language, return_tensors):
        self.calls.append({"sampling_rate": sampling_rate, "language": language})
        return _StubInputs(input_features=audio)

    def decode(self, sequences, skip_special_tokens=False):
        return [self.tokenizer.decode(sequences[0].tolist(), skip_special_tokens)]


class _StubModel:
    device = "cpu"
    dtype = torch.float32

    def generate(self, **_inputs):
        # Frames: "Hello." emitted at frame 2, "Bye!" at frame 10, then a lang tag.
        seq = [0, 0, 1, 2] + [0] * 8 + [3, 4, 99, 0]
        dur = [1, 1, 0, 0] + [1] * 8 + [0, 0, 0, 1, 1]
        return type("Out", (), {"sequences": torch.tensor([seq]), "durations": torch.tensor([dur])})()


def _make_backend(stub_nemotron, language, sub_timestamps=True) -> tuple[NemotronASRBackend, _StubProcessor]:
    backend = NemotronASRBackend(language=language, sub_timestamps=sub_timestamps)
    processor = _StubProcessor()
    backend._processor = processor
    backend._model = _StubModel()
    return backend, processor


def test_nemotron_transcribe_splits_sentences(stub_nemotron):
    backend, processor = _make_backend(stub_nemotron, "zh")
    segments = backend.transcribe(np.zeros(16000 * 2, dtype=np.float32), start_offset=5.0)
    assert processor.calls[0] == {"sampling_rate": 16000, "language": "zh-CN"}
    assert [s.text for s in segments] == ["Hello.", "Bye!"]
    assert segments[0].start == 5.0
    assert segments[0].end == pytest.approx(5.0 + 10 * 0.08)
    assert segments[1].start == pytest.approx(5.0 + 10 * 0.08)
    assert segments[1].end == 7.0


def test_nemotron_transcribe_without_sub_timestamps(stub_nemotron):
    backend, processor = _make_backend(stub_nemotron, None, sub_timestamps=False)
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32), start_offset=1.0)
    assert processor.calls[0]["language"] == "auto"
    assert len(segments) == 1
    assert segments[0].text == "Hello. Bye!"
    assert (segments[0].start, segments[0].end) == (1.0, 2.0)


def test_nemotron_transcribe_without_sub_timestamps_empty_text(stub_nemotron):
    backend, _ = _make_backend(stub_nemotron, "en", sub_timestamps=False)

    class SilentModel(_StubModel):
        def generate(self, **_inputs):
            return type("Out", (), {"sequences": torch.tensor([[0, 0, 99]]), "durations": torch.tensor([[1, 1, 0]])})()

    backend._model = SilentModel()
    assert backend.transcribe(np.zeros(16000, dtype=np.float32)) == []


def test_nemotron_invalid_language_fails_at_init(monkeypatch):
    class StubModelLoader:
        @staticmethod
        def from_pretrained(_model_id, dtype=None):
            model = _StubModel()
            model.to = lambda _device: model  # type: ignore[attr-defined]
            model.eval = lambda: model  # type: ignore[attr-defined]
            return model

    class StubProcessorLoader:
        @staticmethod
        def from_pretrained(_model_id):
            return _StubProcessor()

    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = StubProcessorLoader  # type: ignore[attr-defined]
    transformers_module.AutoModelForRNNT = StubModelLoader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr("vad_transcribe_py.backends.nemotron_asr._detect_device", lambda _device=None: "cpu")

    with pytest.raises(ValueError, match="Unrecognized language code"):
        NemotronASRBackend(language="xx")
    assert NemotronASRBackend(language="zh")._language_key == "zh-CN"


def test_split_sentences_on_non_latin_terminal_marks():
    pieces = ["▁كيف", "؟", "حسنا", "▁नमस्ते", "।", "ठीक", "你好", "。", "再见", "！", "！"]
    tokens = timed_tokens(list(range(1, 12)), [0] * 11, {BLANK}, pieces)
    assert [[t.piece for t in s] for s in split_sentences(tokens)] == [
        ["▁كيف", "؟"],
        ["حسنا", "▁नमस्ते", "।"],
        ["ठीक", "你好", "。"],
        ["再见", "！", "！"],
    ]


def test_nemotron_drops_model_blank_id_that_differs_from_processor(stub_nemotron):
    # Real model: processor.blank_token_id (0 here) != config.blank_token_id == pad (98).
    backend, _ = _make_backend(stub_nemotron, "en")

    class PadBlankModel(_StubModel):
        config = type("Cfg", (), {"blank_token_id": 98})()

        def generate(self, **_inputs):
            seq = [98, 98, 1, 2, 98] + [98] * 7 + [3, 4, 98]
            dur = [1, 1, 0, 0, 1] + [1] * 7 + [0, 0, 1]
            return type("Out", (), {"sequences": torch.tensor([seq]), "durations": torch.tensor([dur])})()

    backend._model = PadBlankModel()
    segments = backend.transcribe(np.zeros(16000 * 2, dtype=np.float32))
    assert [s.text for s in segments] == ["Hello.", "Bye!"]
    assert segments[1].start == pytest.approx(10 * 0.08)
