import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vad_transcribe_py.audio_transcriber as audio_transcriber
from vad_transcribe_py.backends.qwen import QwenASRBackend
from vad_transcribe_py.backends.whisper import WhisperBackend, _resolve_whisper_model_id
from vad_transcribe_py.vad_processor import (
    MOONSHINE_NON_STREAMING_HARD_LIMIT_SECONDS,
    MOONSHINE_STREAMING_HARD_LIMIT_SECONDS,
    WHISPER_HARD_LIMIT_SECONDS,
)


@pytest.fixture(autouse=True)
def stub_whisper(monkeypatch):
    """Stub out Whisper model loading."""
    monkeypatch.setattr(WhisperBackend, "_load_whisper", lambda _self: None)


def test_transcriber_initialization():
    """Test that transcriber initializes with correct parameters."""
    transcriber = WhisperBackend(
        language="en",
        model="large-v3-turbo",
    )

    assert transcriber.language == "en"
    assert transcriber.model == "large-v3-turbo"


def test_create_transcriber_factory():
    """Test the factory function creates transcriber correctly."""
    transcriber = audio_transcriber.create_transcriber(
        language="zh",
        model="large-v3",
        backend="whisper",
    )

    assert isinstance(transcriber, audio_transcriber.AudioTranscriber)


def test_get_window_size_samples():
    """Test window size calculation."""
    window_size = audio_transcriber.get_window_size_samples()
    assert window_size == 512  # For 16kHz sample rate


def test_transcribed_segment_dataclass():
    """Test TranscribedSegment dataclass."""
    segment = audio_transcriber.TranscribedSegment(
        text="Hello world",
        start=1.5,
        end=3.0,
    )

    assert segment.text == "Hello world"
    assert segment.start == 1.5
    assert segment.end == 3.0


def test_process_text_chinese_no_conversion():
    """Test that Chinese text is not converted by default."""
    result = audio_transcriber.process_text("简体中文", "zh", "none")
    assert result == "简体中文"


def test_process_text_chinese_to_traditional():
    """Test that Chinese text is converted to Traditional Chinese."""
    result = audio_transcriber.process_text("简体中文", "zh", "traditional")
    assert result == "簡體中文"


def test_process_text_chinese_to_simplified():
    """Test that Chinese text is converted to Simplified Chinese."""
    result = audio_transcriber.process_text("簡體中文", "zh", "simplified")
    assert result == "简体中文"


def test_process_text_cantonese_no_conversion():
    """Test that Cantonese text is not converted by default."""
    result = audio_transcriber.process_text("简体中文", "yue", "none")
    assert result == "简体中文"


def test_process_text_cantonese_to_traditional():
    """Test that Cantonese text is converted to Traditional Chinese."""
    result = audio_transcriber.process_text("简体中文", "yue", "traditional")
    assert result == "簡體中文"


def test_process_text_english():
    """Test that English text is not modified."""
    result = audio_transcriber.process_text("Hello world", "en", "none")
    assert result == "Hello world"


def test_unsupported_backend():
    """Test that unsupported backend raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        audio_transcriber.create_transcriber(
            language="en",
            model="large-v3-turbo",
            backend="unsupported_backend",
        )


def test_hard_limit_seconds_whisper():
    """Test that whisper backend reports correct hard limit."""
    transcriber = WhisperBackend(
        language="en",
        model="large-v3-turbo",
    )
    assert transcriber.hard_limit_seconds == WHISPER_HARD_LIMIT_SECONDS


def test_moonshine_hard_limits_from_model_config():
    """Test that moonshine hard limits come from model config."""
    assert MOONSHINE_STREAMING_HARD_LIMIT_SECONDS == 60
    assert MOONSHINE_NON_STREAMING_HARD_LIMIT_SECONDS == 9


def test_resolve_whisper_model_id():
    """Test whisper model ID resolution."""
    assert _resolve_whisper_model_id("large-v3-turbo") == "openai/whisper-large-v3-turbo"
    assert _resolve_whisper_model_id("openai/whisper-large-v3") == "openai/whisper-large-v3"


def test_moonshine_resolve_model():
    """Test moonshine model resolution via models.py."""
    from vad_transcribe_py.moonshine.models import resolve_model

    # English defaults to small-streaming
    name, lang, arch, is_streaming, url, hard_limit, soft_limit = resolve_model("en")
    assert name == "small-streaming-en"
    assert is_streaming is True
    assert hard_limit == MOONSHINE_STREAMING_HARD_LIMIT_SECONDS
    assert soft_limit == 6.0

    # Chinese defaults to base
    name, lang, arch, is_streaming, url, hard_limit, soft_limit = resolve_model("zh")
    assert name == "base-zh"
    assert is_streaming is False
    assert hard_limit == MOONSHINE_NON_STREAMING_HARD_LIMIT_SECONDS
    assert soft_limit == 6.0

    # Explicit model
    name, *_ = resolve_model("en", "tiny")
    assert name == "tiny-en"


def test_moonshine_resolve_model_invalid_language():
    """Test moonshine rejects unsupported language."""
    from vad_transcribe_py.moonshine.models import resolve_model

    with pytest.raises(ValueError, match="Unknown language"):
        resolve_model("fr")


def test_qwen_uses_non_streaming_transformers_backend(monkeypatch):
    """Test qwen-asr is initialized in non-streaming transformers mode."""

    class StubInputs(dict):
        def to(self, *args, **kwargs):
            return self

    class StubProcessor:
        def __call__(self, text, audio, return_tensors, padding):
            assert text == ["prompt:English"]
            assert len(audio) == 1
            assert return_tensors == "pt"
            assert padding is True
            return StubInputs({
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "input_features": torch.zeros((1, 1, 1), dtype=torch.float32),
                "feature_attention_mask": torch.ones((1, 1), dtype=torch.long),
            })

        def batch_decode(self, sequences, **_kwargs):
            assert tuple(sequences.shape) == (1, 1)
            return ["hello"]

    class StubQwen3ASRModel:
        llm_called = False
        from_pretrained_calls = []
        generate_calls = []

        def __init__(self):
            self.backend = "transformers"
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.max_new_tokens = 64

            def generate(**kwargs):
                type(self).generate_calls.append(kwargs)
                return torch.tensor([[11, 22, 33]], dtype=torch.long)

            self.model = SimpleNamespace(
                thinker=SimpleNamespace(
                    rope_deltas=torch.tensor([[1.0]], dtype=torch.float32),
                    generate=generate,
                ),
            )
            self.processor = StubProcessor()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.from_pretrained_calls.append((args, kwargs))
            return cls()

        @classmethod
        def LLM(cls, *args, **kwargs):
            cls.llm_called = True
            raise AssertionError("streaming/vLLM path should not be used")

        def _build_text_prompt(self, context, force_language):
            assert context == ""
            return f"prompt:{force_language}"

    qwen_module = ModuleType("qwen_asr")
    qwen_module.Qwen3ASRModel = StubQwen3ASRModel
    inference_module = ModuleType("qwen_asr.inference")
    utils_module = ModuleType("qwen_asr.inference.utils")
    utils_module.parse_asr_output = lambda raw, user_language=None: (user_language or "", raw)

    monkeypatch.setitem(sys.modules, "qwen_asr", qwen_module)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference", inference_module)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference.utils", utils_module)

    backend = QwenASRBackend(language="en")
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert StubQwen3ASRModel.from_pretrained_calls
    assert StubQwen3ASRModel.llm_called is False
    assert StubQwen3ASRModel.generate_calls
    assert StubQwen3ASRModel.generate_calls[0]["return_dict_in_generate"] is False
    assert backend._qwen_model.backend == "transformers"
    assert backend._qwen_model.model.thinker.rope_deltas is None
    assert segments[0].text == "hello"


def test_qwen_rejects_streaming_or_vllm_backend(monkeypatch):
    """Test qwen-asr fails fast if it is not using the transformers backend."""

    class StubQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(backend="vllm")

    qwen_module = ModuleType("qwen_asr")
    qwen_module.Qwen3ASRModel = StubQwen3ASRModel
    inference_module = ModuleType("qwen_asr.inference")
    utils_module = ModuleType("qwen_asr.inference.utils")
    utils_module.parse_asr_output = lambda raw, user_language=None: (user_language or "", raw)

    monkeypatch.setitem(sys.modules, "qwen_asr", qwen_module)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference", inference_module)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference.utils", utils_module)

    with pytest.raises(RuntimeError, match="non-streaming transformers backend"):
        QwenASRBackend(language="en")
