import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vad_transcribe_py.audio_transcriber as audio_transcriber
from vad_transcribe_py.backends.glm_asr import GLM_ASR_DEFAULT_MODEL, GLMASRBackend
from vad_transcribe_py.backends.glm_asr_mlx import (
    GLM_ASR_MLX_DEFAULT_MODEL,
    GLM_ASR_MLX_MAX_TOKENS,
    GLMASRMLXBackend,
)
from vad_transcribe_py.backends.qwen_asr_mlx import QwenASRMLXBackend
from vad_transcribe_py.backends.qwen_rs import QwenASRRsBackend
from vad_transcribe_py.backends.whisper import WhisperBackend, _resolve_whisper_model_id
from vad_transcribe_py.vad_processor import (
    GLM_ASR_HARD_LIMIT_SECONDS,
    GLM_ASR_SOFT_LIMIT_SECONDS,
    MOONSHINE_NON_STREAMING_HARD_LIMIT_SECONDS,
    MOONSHINE_STREAMING_HARD_LIMIT_SECONDS,
    WHISPER_HARD_LIMIT_NO_SUB_TIMESTAMPS_SECONDS,
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
    result = audio_transcriber.process_text("简体中文", "none")
    assert result == "简体中文"


def test_process_text_chinese_to_traditional():
    """Test that Chinese text is converted to Traditional Chinese."""
    result = audio_transcriber.process_text("简体中文", "traditional")
    assert result == "簡體中文"


def test_process_text_chinese_to_simplified():
    """Test that Chinese text is converted to Simplified Chinese."""
    result = audio_transcriber.process_text("簡體中文", "simplified")
    assert result == "简体中文"


def test_process_text_english_none():
    """Test that English text is not modified with none conversion."""
    result = audio_transcriber.process_text("Hello world", "none")
    assert result == "Hello world"


def test_unsupported_backend():
    """Test that unsupported backend raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        audio_transcriber.create_transcriber(
            language="en",
            model="large-v3-turbo",
            backend="unsupported_backend",
        )


def test_whisper_conditioning_enabled_by_default():
    """Test that conditioning is enabled by default."""
    transcriber = WhisperBackend(
        language="en",
        model="large-v3-turbo",
    )
    assert transcriber._condition is True
    assert transcriber._prompt_ids is None


def test_whisper_conditioning_disabled():
    """Test that conditioning can be disabled."""
    transcriber = WhisperBackend(
        language="en",
        model="large-v3-turbo",
        condition=False,
    )
    assert transcriber._condition is False


def test_create_transcriber_with_no_condition():
    """Test that factory passes condition=False to whisper backend."""
    transcriber = audio_transcriber.create_transcriber(
        language="en",
        model="large-v3-turbo",
        backend="whisper",
        condition=False,
    )
    assert transcriber._condition is False


def test_hard_limit_seconds_whisper():
    """Test that whisper backend reports the long-form hard limit by default."""
    transcriber = WhisperBackend(
        language="en",
        model="large-v3-turbo",
    )
    assert transcriber.hard_limit_seconds == WHISPER_HARD_LIMIT_SECONDS
    assert WHISPER_HARD_LIMIT_SECONDS == 60


def test_hard_limit_seconds_whisper_no_sub_timestamps():
    """sub_timestamps=False caps the limit at the model's native 30s window."""
    transcriber = WhisperBackend(
        language="en",
        model="large-v3-turbo",
        sub_timestamps=False,
    )
    assert transcriber.hard_limit_seconds == WHISPER_HARD_LIMIT_NO_SUB_TIMESTAMPS_SECONDS
    assert WHISPER_HARD_LIMIT_NO_SUB_TIMESTAMPS_SECONDS == 30


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

    # Spanish defaults to base (non-streaming)
    name, lang, arch, is_streaming, url, hard_limit, soft_limit = resolve_model("es")
    assert name == "base-es"
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


@pytest.fixture()
def stub_qwen_rs(monkeypatch):
    """Stub out QwenASRRsBackend model loading and device detection."""
    monkeypatch.setattr(QwenASRRsBackend, "_load_model", lambda _self, _model: None)
    monkeypatch.setattr(QwenASRRsBackend, "_detect_device", staticmethod(lambda: "cpu"))


def test_qwen_rs_conditioning_enabled_by_default(stub_qwen_rs):
    """Test that conditioning is enabled by default for qwen-asr-rs."""
    transcriber = QwenASRRsBackend(language="en")
    assert transcriber._condition is True
    assert transcriber._previous_text == ""


def test_qwen_rs_conditioning_disabled(stub_qwen_rs):
    """Test that conditioning can be disabled for qwen-asr-rs."""
    transcriber = QwenASRRsBackend(language="en", condition=False)
    assert transcriber._condition is False


def test_qwen_rs_device_default(stub_qwen_rs):
    """Test that qwen-asr-rs auto-detects device when not specified."""
    transcriber = QwenASRRsBackend(language="en")
    assert transcriber._device == "cpu"  # stubbed _detect_device returns "cpu"


def test_qwen_rs_device_custom(stub_qwen_rs):
    """Test that qwen-asr-rs accepts a custom device."""
    transcriber = QwenASRRsBackend(language="en", device="metal")
    assert transcriber._device == "metal"


def test_create_transcriber_qwen_rs_with_condition(stub_qwen_rs):
    """Test that factory passes condition to qwen-asr-rs backend."""
    transcriber = audio_transcriber.create_transcriber(
        language="en",
        backend="qwen-asr-rs",
        condition=False,
    )
    assert transcriber._condition is False


def test_create_transcriber_qwen_rs_with_device(stub_qwen_rs):
    """Test that factory passes device to qwen-asr-rs backend."""
    transcriber = audio_transcriber.create_transcriber(
        language="en",
        backend="qwen-asr-rs",
        device="metal",
    )
    assert transcriber._device == "metal"


def test_qwen_rs_hard_limit(stub_qwen_rs):
    """Test that qwen-asr-rs reports correct hard limit."""
    from vad_transcribe_py.vad_processor import QWEN_ASR_HARD_LIMIT_SECONDS
    transcriber = QwenASRRsBackend(language="en")
    assert transcriber.hard_limit_seconds == QWEN_ASR_HARD_LIMIT_SECONDS


def test_qwen_rs_transcribe_integration(monkeypatch):
    """Test qwen-asr-rs transcribe via a stub qwencandle module."""

    class StubQwenAsr:
        def __init__(self, device: str, model_id: str | None = None):
            self.device = device
            self.model_id = model_id
            self.transcribe_calls: list[dict[str, object]] = []

        def transcribe(self, samples, *, language=None, context=None):
            self.transcribe_calls.append({"samples": samples, "language": language, "context": context})
            return "hello world"

    qwencandle_module = ModuleType("qwencandle")
    qwencandle_module.QwenAsr = StubQwenAsr  # type: ignore[attr-defined]
    qwencandle_module.DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"  # type: ignore[attr-defined]
    qwencandle_module.is_cuda_available = lambda: False  # type: ignore[attr-defined]
    qwencandle_module.is_metal_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwencandle", qwencandle_module)

    backend = QwenASRRsBackend(language="en")
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert isinstance(backend._model, StubQwenAsr)
    assert backend._model.transcribe_calls
    assert backend._model.transcribe_calls[0]["language"] == "English"
    assert segments[0].text == "hello world"


@pytest.fixture()
def stub_qwen_mlx(monkeypatch):
    """Stub out QwenASRMLXBackend model loading."""
    monkeypatch.setattr(QwenASRMLXBackend, "_load_model", lambda _self: None)


def test_qwen_mlx_conditioning_enabled_by_default(stub_qwen_mlx):
    """Test that conditioning is enabled by default for qwen-asr-mlx."""
    transcriber = QwenASRMLXBackend(language="en")
    assert transcriber._condition is True
    assert transcriber._previous_text == ""


def test_qwen_mlx_conditioning_disabled(stub_qwen_mlx):
    """Test that conditioning can be disabled for qwen-asr-mlx."""
    transcriber = QwenASRMLXBackend(language="en", condition=False)
    assert transcriber._condition is False


def test_qwen_mlx_hard_limit(stub_qwen_mlx):
    """Test that qwen-asr-mlx reports correct hard limit."""
    from vad_transcribe_py.vad_processor import QWEN_ASR_HARD_LIMIT_SECONDS
    transcriber = QwenASRMLXBackend(language="en")
    assert transcriber.hard_limit_seconds == QWEN_ASR_HARD_LIMIT_SECONDS


def test_qwen_mlx_soft_limit(stub_qwen_mlx):
    """Test that qwen-asr-mlx reports correct soft limit."""
    from vad_transcribe_py.vad_processor import QWEN_ASR_SOFT_LIMIT_SECONDS
    transcriber = QwenASRMLXBackend(language="en")
    assert transcriber.soft_limit_seconds == QWEN_ASR_SOFT_LIMIT_SECONDS


def test_qwen_mlx_device_ignored(stub_qwen_mlx, caplog):
    """Test that non-metal device is accepted but logged+ignored."""
    import logging
    caplog.set_level(logging.WARNING)
    _ = QwenASRMLXBackend(language="en", device="cuda")
    assert any("ignored" in r.message for r in caplog.records)


def test_create_transcriber_qwen_mlx_with_condition(stub_qwen_mlx):
    """Test that factory passes condition to qwen-asr-mlx backend."""
    transcriber = audio_transcriber.create_transcriber(
        language="en",
        backend="qwen-asr-mlx",
        condition=False,
    )
    assert transcriber._condition is False
    assert isinstance(transcriber, audio_transcriber.AudioTranscriber)


def test_qwen_mlx_transcribe_integration(monkeypatch):
    """Test qwen-asr-mlx transcribe via a stub mlx_audio module."""

    from vad_transcribe_py.backends.qwen_asr_mlx import QWEN_ASR_MLX_MAX_TOKENS

    class StubMLXModel:
        def __init__(self):
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, audio, *, language=None, system_prompt=None, max_tokens=8192, verbose=False, **_kwargs):
            self.generate_calls.append(
                {
                    "audio": audio,
                    "language": language,
                    "system_prompt": system_prompt,
                    "max_tokens": max_tokens,
                }
            )
            return SimpleNamespace(text="hello mlx")

    stub = StubMLXModel()
    mlx_audio_module = ModuleType("mlx_audio")
    stt_module = ModuleType("mlx_audio.stt")
    utils_module = ModuleType("mlx_audio.stt.utils")
    utils_module.load_model = lambda _model_id: stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils_module)

    backend = QwenASRMLXBackend(language="en")
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert stub.generate_calls
    assert stub.generate_calls[0]["language"] == "English"
    # First pass: _previous_text is empty → system_prompt stays None.
    assert stub.generate_calls[0]["system_prompt"] is None
    # Cap generation so larger Qwen3-ASR variants can't hang on non-speech audio.
    assert stub.generate_calls[0]["max_tokens"] == QWEN_ASR_MLX_MAX_TOKENS
    assert segments[0].text == "hello mlx"
    assert backend._previous_text == "hello mlx"

    # Second call: _previous_text flows into system_prompt.
    _ = backend.transcribe(np.zeros(16000, dtype=np.float32))
    assert stub.generate_calls[1]["system_prompt"] == "hello mlx"


def test_qwen_mlx_unknown_language_raises(stub_qwen_mlx):
    """Test that an unrecognized language code surfaces ValueError at transcribe time."""
    backend = QwenASRMLXBackend(language="xx")
    backend._mlx_model = object()  # satisfy the assert in transcribe()
    with pytest.raises(ValueError, match="Unrecognized language code"):
        backend.transcribe(np.zeros(16000, dtype=np.float32))


@pytest.fixture()
def stub_glm_asr(monkeypatch):
    """Stub out GLMASRBackend model loading and device detection."""
    monkeypatch.setattr(GLMASRBackend, "_load_model", lambda _self: None)
    monkeypatch.setattr("vad_transcribe_py.backends.glm_asr._detect_device", lambda _device=None: "cpu")


def test_glm_asr_defaults(stub_glm_asr):
    """Test that glm-asr initializes with the default model and limits."""
    transcriber = GLMASRBackend(language="en")
    assert transcriber.model == GLM_ASR_DEFAULT_MODEL
    assert transcriber.hard_limit_seconds == GLM_ASR_HARD_LIMIT_SECONDS
    assert transcriber.soft_limit_seconds == GLM_ASR_SOFT_LIMIT_SECONDS


def test_create_transcriber_glm_asr(stub_glm_asr):
    """Test that factory creates the glm-asr backend."""
    transcriber = audio_transcriber.create_transcriber(
        language="en",
        backend="glm-asr",
        device="cuda",
    )
    assert isinstance(transcriber, GLMASRBackend)
    assert isinstance(transcriber, audio_transcriber.AudioTranscriber)


def test_glm_asr_sets_torch_threads(stub_glm_asr, monkeypatch):
    """Test that glm-asr forwards num_threads to torch CPU threading."""
    set_threads_calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", set_threads_calls.append)

    _ = GLMASRBackend(language="en", num_threads=3)

    assert set_threads_calls == [3]


def test_glm_asr_unknown_language_raises(stub_glm_asr):
    """Test that an unrecognized GLM-ASR language code raises ValueError."""
    backend = GLMASRBackend(language="xx")
    backend._processor = object()
    backend._model = object()
    with pytest.raises(ValueError, match="Unrecognized language code"):
        backend.transcribe(np.zeros(16000, dtype=np.float32))


def test_glm_asr_transcribe_integration(monkeypatch):
    """Test glm-asr transcribe via stub Transformers classes."""

    class StubInputs(dict):
        def __init__(self):
            super().__init__(input_ids=np.array([[10, 11, 12]]))
            self.device: str | None = None
            self.dtype: torch.dtype | None = None

        def to(self, device: str, dtype: torch.dtype | None = None):
            self.device = device
            self.dtype = dtype
            return self

    class StubProcessor:
        def __init__(self):
            self.apply_calls: list[dict[str, object]] = []
            self.decode_calls: list[dict[str, object]] = []
            self.last_inputs: StubInputs | None = None

        def apply_transcription_request(self, audio, *, prompt=None, return_tensors=None):
            self.apply_calls.append(
                {"audio": audio, "prompt": prompt, "return_tensors": return_tensors}
            )
            self.last_inputs = StubInputs()
            return self.last_inputs

        def batch_decode(self, generated_ids, *, skip_special_tokens=False):
            self.decode_calls.append(
                {"generated_ids": generated_ids, "skip_special_tokens": skip_special_tokens}
            )
            return ["hello glm"]

    class StubAutoProcessor:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == GLM_ASR_DEFAULT_MODEL
            return stub_processor

    class StubGLMModel:
        def __init__(self):
            self.device: str | None = None
            self.dtype = torch.bfloat16
            self.eval_called = False
            self.generate_calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, model_id, *, dtype=None):
            assert model_id == GLM_ASR_DEFAULT_MODEL
            assert dtype == "auto"
            return stub_model

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

        def generate(self, **inputs):
            self.generate_calls.append(inputs)
            return np.array([[10, 11, 12, 20, 21]])

    stub_processor = StubProcessor()
    stub_model = StubGLMModel()
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = StubAutoProcessor  # type: ignore[attr-defined]
    transformers_module.GlmAsrForConditionalGeneration = StubGLMModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = GLMASRBackend(language="en", device="cuda")
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32), start_offset=2.0)

    assert stub_model.device == "cuda:0"
    assert stub_model.eval_called is True
    assert stub_processor.apply_calls
    assert stub_processor.apply_calls[0]["prompt"] == "Transcribe the input speech in English."
    assert stub_processor.apply_calls[0]["return_tensors"] == "pt"
    assert stub_processor.last_inputs is not None
    assert stub_processor.last_inputs.device == "cuda:0"
    assert stub_processor.last_inputs.dtype == torch.bfloat16
    assert stub_model.generate_calls
    assert stub_model.generate_calls[0]["max_new_tokens"] == 500
    assert stub_processor.decode_calls[0]["generated_ids"].tolist() == [[20, 21]]
    assert stub_processor.decode_calls[0]["skip_special_tokens"] is True
    assert segments[0].text == "hello glm"
    assert segments[0].start == 2.0
    assert segments[0].end == 3.0


def test_glm_asr_default_prompt(monkeypatch):
    """Test that omitting language lets the processor use its default prompt."""

    class StubInputs(dict):
        def __init__(self):
            super().__init__(input_ids=np.array([[1]]))

        def to(self, _device: str):
            return self

    class StubProcessor:
        def __init__(self):
            self.prompt = "unset"

        def apply_transcription_request(self, _audio, *, prompt=None, return_tensors=None):
            self.prompt = prompt
            return StubInputs()

        def batch_decode(self, _generated_ids, *, skip_special_tokens=False):
            return ["hello"]

    class StubAutoProcessor:
        @staticmethod
        def from_pretrained(_model_id):
            return stub_processor

    class StubGLMModel:
        @classmethod
        def from_pretrained(cls, _model_id, *, dtype=None):
            return stub_model

        def to(self, _device: str):
            return self

        def eval(self):
            return self

        def generate(self, **_inputs):
            return np.array([[1, 2]])

    stub_processor = StubProcessor()
    stub_model = StubGLMModel()
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = StubAutoProcessor  # type: ignore[attr-defined]
    transformers_module.GlmAsrForConditionalGeneration = StubGLMModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = GLMASRBackend(language=None)
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert stub_processor.prompt is None
    assert segments[0].text == "hello"


@pytest.fixture()
def stub_glm_mlx(monkeypatch):
    """Stub out GLMASRMLXBackend model loading."""
    monkeypatch.setattr(GLMASRMLXBackend, "_load_model", lambda _self: None)


def test_glm_mlx_defaults(stub_glm_mlx):
    """Test that glm-asr-mlx initializes with the default model and limits."""
    transcriber = GLMASRMLXBackend(language=None)
    assert transcriber.model == GLM_ASR_MLX_DEFAULT_MODEL
    assert transcriber.hard_limit_seconds == GLM_ASR_HARD_LIMIT_SECONDS
    assert transcriber.soft_limit_seconds == GLM_ASR_SOFT_LIMIT_SECONDS


def test_glm_mlx_device_ignored(stub_glm_mlx, caplog):
    """Test that non-metal device is accepted but logged+ignored."""
    import logging
    caplog.set_level(logging.WARNING)
    _ = GLMASRMLXBackend(language="en", device="cuda")
    assert any("ignored" in r.message and "device" in r.message for r in caplog.records)


def test_glm_mlx_num_threads_ignored(stub_glm_mlx, caplog):
    """Test that num_threads is accepted but logged+ignored."""
    import logging
    caplog.set_level(logging.WARNING)
    _ = GLMASRMLXBackend(language="en", num_threads=4)
    assert any("ignored" in r.message and "num_threads" in r.message for r in caplog.records)


def test_glm_mlx_unsupported_language_warns(stub_glm_mlx, caplog):
    """Test that an out-of-support language emits a warning but does not raise."""
    import logging
    caplog.set_level(logging.WARNING)
    _ = GLMASRMLXBackend(language="fr")
    assert any("English/Chinese" in r.message for r in caplog.records)


def test_create_transcriber_glm_mlx_default_model(stub_glm_mlx):
    """Test that factory uses GLM_ASR_MLX_DEFAULT_MODEL when model is None."""
    transcriber = audio_transcriber.create_transcriber(
        language=None,
        backend="glm-asr-mlx",
    )
    assert isinstance(transcriber, GLMASRMLXBackend)
    assert transcriber.model == GLM_ASR_MLX_DEFAULT_MODEL


def test_create_transcriber_glm_mlx_condition_rejected(stub_glm_mlx):
    """Test that condition=True raises ValueError for glm-asr-mlx."""
    with pytest.raises(ValueError, match="condition=True is not supported"):
        _ = audio_transcriber.create_transcriber(
            language=None,
            backend="glm-asr-mlx",
            condition=True,
        )


def test_glm_mlx_transcribe_integration(monkeypatch):
    """Test glm-asr-mlx transcribe via a stub mlx_audio module."""

    class StubMLXModel:
        def __init__(self):
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, audio, *, max_tokens=128, verbose=False, **_kwargs):
            self.generate_calls.append(
                {"audio": audio, "max_tokens": max_tokens, "verbose": verbose}
            )
            return SimpleNamespace(text="hello glm mlx")

    stub = StubMLXModel()
    mlx_audio_module = ModuleType("mlx_audio")
    stt_module = ModuleType("mlx_audio.stt")
    utils_module = ModuleType("mlx_audio.stt.utils")
    utils_module.load_model = lambda _model_id: stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils_module)

    backend = GLMASRMLXBackend(language=None)
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32), start_offset=1.5)

    assert stub.generate_calls
    assert stub.generate_calls[0]["max_tokens"] == GLM_ASR_MLX_MAX_TOKENS
    assert stub.generate_calls[0]["verbose"] is False
    assert segments[0].text == "hello glm mlx"
    assert segments[0].start == 1.5
    assert segments[0].end == 2.5


def test_glm_mlx_empty_text_still_returns_segment(monkeypatch):
    """Empty/whitespace model output still yields a segment (matches qwen-asr-mlx)."""

    class StubMLXModel:
        def generate(self, _audio, **_kwargs):
            return SimpleNamespace(text="   ")

    stub = StubMLXModel()
    mlx_audio_module = ModuleType("mlx_audio")
    stt_module = ModuleType("mlx_audio.stt")
    utils_module = ModuleType("mlx_audio.stt.utils")
    utils_module.load_model = lambda _model_id: stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils_module)

    backend = GLMASRMLXBackend(language=None)
    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(segments) == 1
    assert segments[0].text == "   "


# ---------------------------------------------------------------------------
# Retry-without-prompt behavior
#
# Each conditioning-capable backend (whisper, qwen_rs, qwen_asr_mlx) re-runs
# inference with the conditioning prompt cleared if the first output near-
# duplicates the previous segment's text and a prompt was actually used.
# Whisper gates this on sub_timestamps=False (the no-stitching path) — the
# multi-chunk path keeps existing behavior.
# ---------------------------------------------------------------------------


class _StubPromptIds:
    def to(self, _device):
        return self


class _StubProcessor:
    def get_prompt_ids(self, _text, return_tensors=None):
        return _StubPromptIds()


def _whisper_no_subts_backend(prior_line: str, prompt_set: bool) -> WhisperBackend:
    """Build a stubbed whisper backend pre-loaded with prior-segment state."""
    backend = WhisperBackend(language="en", model="large-v3-turbo", sub_timestamps=False)
    backend._prior_line = prior_line
    backend._prompt_ids = "prompt-tensor" if prompt_set else None  # opaque sentinel; backend only checks for None
    backend._processor = _StubProcessor()  # post-transcribe conditioning update needs get_prompt_ids
    return backend


def test_whisper_retry_without_prompt_when_output_near_duplicates_prior():
    """Prompt was used and output near-duplicates prior → re-run without prompt_ids and mark prompt_retry=True."""
    backend = _whisper_no_subts_backend(prior_line="hello world", prompt_set=True)
    calls: list[dict[str, Any]] = []

    def fake_pipe(audio, *, return_timestamps, generate_kwargs):
        calls.append({"return_timestamps": return_timestamps, "generate_kwargs": dict(generate_kwargs)})
        # First call returns a near-duplicate; second call (no prompt_ids) returns different text.
        if "prompt_ids" in generate_kwargs:
            return {"text": "hello world"}
        return {"text": "something different"}

    backend.pipe = fake_pipe

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(calls) == 2
    assert "prompt_ids" in calls[0]["generate_kwargs"]
    assert "prompt_ids" not in calls[1]["generate_kwargs"]
    assert len(segments) == 1
    assert segments[0].text == "something different"
    assert segments[0].prompt_retry is True


def test_whisper_no_retry_when_prompt_was_not_used():
    """No prompt → no retry, even if output happens to near-duplicate prior."""
    backend = _whisper_no_subts_backend(prior_line="hello world", prompt_set=False)
    calls: list[dict[str, Any]] = []

    def fake_pipe(audio, *, return_timestamps, generate_kwargs):
        calls.append({"generate_kwargs": dict(generate_kwargs)})
        return {"text": "hello world"}

    backend.pipe = fake_pipe

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(calls) == 1
    assert "prompt_ids" not in calls[0]["generate_kwargs"]
    assert segments[0].prompt_retry is False


def test_whisper_no_retry_when_output_differs_from_prior():
    """Prompt was used but output is clearly different → single call only."""
    backend = _whisper_no_subts_backend(prior_line="hello world", prompt_set=True)
    calls: list[dict[str, Any]] = []

    def fake_pipe(audio, *, return_timestamps, generate_kwargs):
        calls.append({"generate_kwargs": dict(generate_kwargs)})
        return {"text": "completely unrelated transcription"}

    backend.pipe = fake_pipe

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(calls) == 1
    assert segments[0].prompt_retry is False


def _whisper_subts_backend(prior_last_chunk: str, prompt_set: bool) -> WhisperBackend:
    """Build a stubbed sub_timestamps=True whisper backend with prior-call state."""
    backend = WhisperBackend(language="en", model="large-v3-turbo", sub_timestamps=True)
    backend._prior_last_chunk = prior_last_chunk
    backend._prior_line = prior_last_chunk  # consistent with single-chunk-prior case
    backend._prompt_ids = "prompt-tensor" if prompt_set else None
    backend._processor = _StubProcessor()
    return backend


def _staged_pipe(staged_results, calls):
    """Return a fake pipe whose successive calls return the staged dicts in order."""
    it = iter(staged_results)

    def fake_pipe(audio, *, return_timestamps, generate_kwargs):
        calls.append({
            "audio_len": len(audio),
            "return_timestamps": return_timestamps,
            "generate_kwargs": dict(generate_kwargs),
        })
        return next(it)

    return fake_pipe


def test_whisper_subts_mid_segment_repetition_retries_without_prompt():
    """chunk[i] near-duplicates chunk[i-1] → trim from chunk[i].start, retry without prompt_ids.

    No initial prompt is set; mid-segment duplicates trigger a retry regardless.
    """
    backend = _whisper_subts_backend(prior_last_chunk="", prompt_set=False)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "hello", "timestamp": (0.0, 2.0)},
                {"text": "abcdefg", "timestamp": (2.0, 5.0)},
                {"text": "abcdefg", "timestamp": (5.0, 8.0)},
            ]},
            {"chunks": [
                {"text": "fresh tail", "timestamp": (0.0, 3.0)},
            ]},
        ],
        calls,
    )

    audio = np.zeros(int(8.0 * 16000), dtype=np.float32)
    segments = backend.transcribe(audio)

    assert len(calls) == 2
    assert calls[1]["audio_len"] == len(audio) - int(5.0 * 16000)  # trimmed at 5.0s
    assert "prompt_ids" not in calls[1]["generate_kwargs"]
    assert [s.text for s in segments] == ["hello", "abcdefg", "fresh tail"]
    assert [s.prompt_retry for s in segments] == [False, False, True]
    # Retried chunk's timestamps were re-based by +5.0s.
    assert segments[2].start == 5.0
    assert segments[2].end == 8.0


def test_whisper_subts_mid_segment_repetition_drops_prompt_ids_when_set():
    """Mid-segment retry also clears prompt_ids when one was passed in."""
    backend = _whisper_subts_backend(prior_last_chunk="", prompt_set=True)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "abc", "timestamp": (0.0, 2.0)},
                {"text": "abc", "timestamp": (2.0, 4.0)},
            ]},
            {"chunks": [{"text": "different", "timestamp": (0.0, 2.0)}]},
        ],
        calls,
    )

    backend.transcribe(np.zeros(int(4.0 * 16000), dtype=np.float32))

    assert len(calls) == 2
    assert "prompt_ids" in calls[0]["generate_kwargs"]
    assert "prompt_ids" not in calls[1]["generate_kwargs"]


def test_whisper_subts_cross_vad_repetition_with_prompt_triggers_retry():
    """chunk[0] near-duplicates _prior_last_chunk → retry whole call without prompt_ids."""
    backend = _whisper_subts_backend(prior_last_chunk="hello world", prompt_set=True)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "hello world", "timestamp": (0.0, 3.0)},
                {"text": "more text", "timestamp": (3.0, 6.0)},
            ]},
            {"chunks": [
                {"text": "actual transcription", "timestamp": (0.0, 6.0)},
            ]},
        ],
        calls,
    )

    audio = np.zeros(int(6.0 * 16000), dtype=np.float32)
    segments = backend.transcribe(audio)

    assert len(calls) == 2
    assert calls[1]["audio_len"] == len(audio)  # trim at 0.0 = whole audio
    assert "prompt_ids" not in calls[1]["generate_kwargs"]
    assert [s.text for s in segments] == ["actual transcription"]
    assert all(s.prompt_retry for s in segments)


def test_whisper_subts_cross_vad_repetition_skipped_without_prompt():
    """chunk[0] near-duplicates _prior_last_chunk but no prompt was used → no retry."""
    backend = _whisper_subts_backend(prior_last_chunk="hello world", prompt_set=False)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "hello world", "timestamp": (0.0, 3.0)},
                {"text": "more text", "timestamp": (3.0, 6.0)},
            ]},
        ],
        calls,
    )

    segments = backend.transcribe(np.zeros(int(6.0 * 16000), dtype=np.float32))

    assert len(calls) == 1
    assert all(not s.prompt_retry for s in segments)


def test_whisper_subts_no_repetition_no_retry():
    """All chunks distinct from each other and from prior_last_chunk → single call."""
    backend = _whisper_subts_backend(prior_last_chunk="prior text", prompt_set=True)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "alpha", "timestamp": (0.0, 2.0)},
                {"text": "beta", "timestamp": (2.0, 4.0)},
                {"text": "gamma", "timestamp": (4.0, 6.0)},
            ]},
        ],
        calls,
    )

    segments = backend.transcribe(np.zeros(int(6.0 * 16000), dtype=np.float32))

    assert len(calls) == 1
    assert [s.text for s in segments] == ["alpha", "beta", "gamma"]
    assert all(not s.prompt_retry for s in segments)


def test_whisper_subts_retry_offsets_timestamps_with_start_offset():
    """Retried-chunk start/end = start_offset + trim_start_sec + chunk_timestamp_in_trimmed_call."""
    backend = _whisper_subts_backend(prior_last_chunk="", prompt_set=False)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "intro", "timestamp": (0.0, 3.0)},
                {"text": "loop", "timestamp": (3.0, 5.0)},
                {"text": "loop", "timestamp": (5.0, 7.0)},
            ]},
            # Trimmed audio starts at 5.0s; retried chunk's local timestamps are (1.0, 4.0).
            {"chunks": [{"text": "rescued", "timestamp": (1.0, 4.0)}]},
        ],
        calls,
    )

    segments = backend.transcribe(np.zeros(int(7.0 * 16000), dtype=np.float32), start_offset=10.0)

    # First two chunks: just the start_offset added.
    assert segments[0].start == 10.0
    assert segments[0].end == 13.0
    assert segments[1].start == 13.0
    assert segments[1].end == 15.0
    # Retried chunk: start_offset + trim_start_sec + local_ts.
    assert segments[2].text == "rescued"
    assert segments[2].start == 10.0 + 5.0 + 1.0
    assert segments[2].end == 10.0 + 5.0 + 4.0
    assert segments[2].prompt_retry is True


def test_whisper_subts_prior_last_chunk_updated_after_retry():
    """After a retried call, _prior_last_chunk reflects the final (retried) chunk's text."""
    backend = _whisper_subts_backend(prior_last_chunk="", prompt_set=False)
    calls: list[dict[str, Any]] = []
    backend.pipe = _staged_pipe(
        [
            {"chunks": [
                {"text": "alpha", "timestamp": (0.0, 2.0)},
                {"text": "alpha", "timestamp": (2.0, 4.0)},
            ]},
            {"chunks": [{"text": "post-retry tail", "timestamp": (0.0, 2.0)}]},
        ],
        calls,
    )

    backend.transcribe(np.zeros(int(4.0 * 16000), dtype=np.float32))

    assert backend._prior_last_chunk == "post-retry tail"


def _install_stub_qwencandle(monkeypatch, side_effect):
    """Install a stub qwencandle module whose QwenAsr.transcribe yields each item from side_effect in turn."""

    class StubQwenAsr:
        def __init__(self, device, model_id=None):
            self.calls: list[dict[str, object]] = []
            self._iter = iter(side_effect)

        def transcribe(self, samples, *, language=None, context=None):
            self.calls.append({"language": language, "context": context})
            return next(self._iter)

    qwencandle_module = ModuleType("qwencandle")
    qwencandle_module.QwenAsr = StubQwenAsr  # type: ignore[attr-defined]
    qwencandle_module.DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"  # type: ignore[attr-defined]
    qwencandle_module.is_cuda_available = lambda: False  # type: ignore[attr-defined]
    qwencandle_module.is_metal_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qwencandle", qwencandle_module)


def test_qwen_rs_retry_without_context_when_output_near_duplicates_prior(monkeypatch):
    _install_stub_qwencandle(monkeypatch, ["hello world", "something different"])
    backend = QwenASRRsBackend(language="en")
    backend._previous_text = "hello world"
    backend._prior_line = "hello world"

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    calls = backend._model.calls  # type: ignore[attr-defined]
    assert len(calls) == 2
    assert calls[0]["context"] == "hello world"
    assert calls[1]["context"] is None
    assert segments[0].text == "something different"
    assert segments[0].prompt_retry is True


def test_qwen_rs_no_retry_when_no_context(monkeypatch):
    _install_stub_qwencandle(monkeypatch, ["hello world"])
    backend = QwenASRRsBackend(language="en")
    backend._previous_text = ""  # empty string → treated as "no prompt"
    backend._prior_line = "hello world"

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    calls = backend._model.calls  # type: ignore[attr-defined]
    assert len(calls) == 1
    assert segments[0].prompt_retry is False


def test_qwen_rs_no_retry_when_output_differs(monkeypatch):
    _install_stub_qwencandle(monkeypatch, ["totally fresh transcription"])
    backend = QwenASRRsBackend(language="en")
    backend._previous_text = "hello world"
    backend._prior_line = "hello world"

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(backend._model.calls) == 1  # type: ignore[attr-defined]
    assert segments[0].prompt_retry is False


def _install_stub_mlx_audio(monkeypatch, texts):
    """Install a stub mlx_audio module whose load_model returns a model yielding texts in turn."""

    class StubMLXModel:
        def __init__(self):
            self.calls: list[dict[str, object]] = []
            self._iter = iter(texts)

        def generate(self, audio, *, language=None, system_prompt=None, max_tokens=8192, verbose=False, **_kwargs):
            self.calls.append({"language": language, "system_prompt": system_prompt})
            return SimpleNamespace(text=next(self._iter))

    stub = StubMLXModel()
    mlx_audio_module = ModuleType("mlx_audio")
    stt_module = ModuleType("mlx_audio.stt")
    utils_module = ModuleType("mlx_audio.stt.utils")
    utils_module.load_model = lambda _model_id: stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt_module)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", utils_module)
    return stub


def test_qwen_mlx_retry_without_system_prompt_when_output_near_duplicates_prior(monkeypatch):
    stub = _install_stub_mlx_audio(monkeypatch, ["hello world", "something different"])
    backend = QwenASRMLXBackend(language="en")
    backend._previous_text = "hello world"
    backend._prior_line = "hello world"

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(stub.calls) == 2
    assert stub.calls[0]["system_prompt"] == "hello world"
    assert stub.calls[1]["system_prompt"] is None
    assert segments[0].text == "something different"
    assert segments[0].prompt_retry is True


def test_qwen_mlx_no_retry_when_no_system_prompt(monkeypatch):
    stub = _install_stub_mlx_audio(monkeypatch, ["hello world"])
    backend = QwenASRMLXBackend(language="en")
    backend._previous_text = ""
    backend._prior_line = "hello world"

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(stub.calls) == 1
    assert stub.calls[0]["system_prompt"] is None
    assert segments[0].prompt_retry is False


def test_qwen_mlx_no_retry_when_output_differs(monkeypatch):
    stub = _install_stub_mlx_audio(monkeypatch, ["totally fresh transcription"])
    backend = QwenASRMLXBackend(language="en")
    backend._previous_text = "hello world"
    backend._prior_line = "hello world"

    segments = backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert len(stub.calls) == 1
    assert segments[0].prompt_retry is False


# ---------------------------------------------------------------------------
# clip_repetitive_text — char-level periodic-run truncation
# ---------------------------------------------------------------------------


_FILLER = (
    "the quick brown fox jumps over the lazy dog and some more words to pad "
    "this out beyond one hundred characters easily"
)


def test_clip_repetitive_text_short_text_unchanged():
    """Lines below 100 chars are passed through verbatim (real-speech guard)."""
    from vad_transcribe_py._utils import clip_repetitive_text

    assert clip_repetitive_text("short") == "short"
    assert clip_repetitive_text("a" * 99) == "a" * 99


def test_clip_repetitive_text_empty_string_unchanged():
    """Empty input passes through."""
    from vad_transcribe_py._utils import clip_repetitive_text

    assert clip_repetitive_text("") == ""


def test_clip_repetitive_text_no_repeats_unchanged():
    """A long line with no periodic run stays as-is."""
    from vad_transcribe_py._utils import clip_repetitive_text

    assert clip_repetitive_text(_FILLER) == _FILLER


def test_clip_repetitive_text_simple_repeat_truncated_after_first_copy():
    """The repetitive tail is replaced; the meaningful prefix + one copy survives."""
    from vad_transcribe_py._utils import INDISTINGUISHABLE_PLACEHOLDER, clip_repetitive_text

    text = _FILLER + " " + "ab" * 15
    assert clip_repetitive_text(text) == _FILLER + " " + "ab" + INDISTINGUISHABLE_PLACEHOLDER


def test_clip_repetitive_text_with_patterns_returns_detected_pattern():
    """The metadata helper returns the exact repeat pattern used for clipping."""
    from vad_transcribe_py._utils import (
        INDISTINGUISHABLE_PLACEHOLDER,
        clip_repetitive_text_with_patterns,
    )

    text = _FILLER + " " + "ab" * 15
    clipped_text, patterns = clip_repetitive_text_with_patterns(text)

    assert clipped_text == _FILLER + " " + "ab" + INDISTINGUISHABLE_PLACEHOLDER
    assert patterns == ["ab"]


def test_clip_repetitive_text_with_patterns_no_clip_returns_empty_patterns():
    """When text is unchanged, no clipped patterns are reported."""
    from vad_transcribe_py._utils import clip_repetitive_text_with_patterns

    assert clip_repetitive_text_with_patterns(_FILLER) == (_FILLER, [])


def test_clip_repetitive_text_repeat_at_start():
    """Whole-line repetition collapses to one pattern copy + placeholder."""
    from vad_transcribe_py._utils import INDISTINGUISHABLE_PLACEHOLDER, clip_repetitive_text

    assert clip_repetitive_text("ab" * 60) == "ab" + INDISTINGUISHABLE_PLACEHOLDER


def test_clip_repetitive_text_longer_pattern():
    """Multi-char patterns are detected the same way."""
    from vad_transcribe_py._utils import INDISTINGUISHABLE_PLACEHOLDER, clip_repetitive_text

    text = _FILLER + " " + "hello" * 12
    assert clip_repetitive_text(text) == _FILLER + " " + "hello" + INDISTINGUISHABLE_PLACEHOLDER


def test_clip_repetitive_text_just_below_min_repeats():
    """9 repeats with default min_repeats=10 is not enough to trigger truncation."""
    from vad_transcribe_py._utils import clip_repetitive_text

    text = _FILLER + " " + "ab" * 9 + " end"
    assert clip_repetitive_text(text) == text


def test_clip_repetitive_text_respects_min_repeats_override():
    """Lowering min_repeats lets shorter runs trigger truncation."""
    from vad_transcribe_py._utils import INDISTINGUISHABLE_PLACEHOLDER, clip_repetitive_text

    assert clip_repetitive_text("ab" * 60, min_repeats=5) == "ab" + INDISTINGUISHABLE_PLACEHOLDER


def test_clip_repetitive_text_rejects_invalid_min_repeats():
    """min_repeats below two cannot define a repeated run."""
    from vad_transcribe_py._utils import clip_repetitive_text

    with pytest.raises(ValueError, match="min_repeats must be at least 2"):
        clip_repetitive_text("ab" * 60, min_repeats=1)


def test_clip_repetitive_text_real_world_whisper_loop():
    """Production case: noise prefix + dungu loop → prefix + first ' dungu' + placeholder.

    The period-6 alignment locks onto the recurring ``" dungu"`` (leading space),
    so the kept first copy carries the leading space and the placeholder follows
    immediately with no separating space.
    """
    from vad_transcribe_py._utils import INDISTINGUISHABLE_PLACEHOLDER, clip_repetitive_text

    prefix = "Kipan san jindin jindu nindu padded out past one hundred chars of plain ASR text"
    text = prefix + " " + "dungu " * 30
    assert clip_repetitive_text(text) == prefix + " dungu" + INDISTINGUISHABLE_PLACEHOLDER


def test_clip_repetitive_text_chinese_single_char_loop():
    """Single CJK char repeating → keep one pair + placeholder."""
    from vad_transcribe_py._utils import INDISTINGUISHABLE_PLACEHOLDER, clip_repetitive_text

    # 100+ chars of "好" — long enough to pass the short-line guard.
    text = "好" * 120
    assert clip_repetitive_text(text) == "好好" + INDISTINGUISHABLE_PLACEHOLDER
