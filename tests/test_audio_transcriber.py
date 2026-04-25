import sys
from types import ModuleType
from types import SimpleNamespace

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
from vad_transcribe_py.backends.mlx import QwenASRMLXBackend
from vad_transcribe_py.backends.qwen_rs import QwenASRRsBackend
from vad_transcribe_py.backends.whisper import WhisperBackend, _resolve_whisper_model_id
from vad_transcribe_py.vad_processor import (
    GLM_ASR_HARD_LIMIT_SECONDS,
    GLM_ASR_SOFT_LIMIT_SECONDS,
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

    class StubMLXModel:
        def __init__(self):
            self.generate_calls: list[dict[str, object]] = []

        def generate(self, audio, *, language=None, system_prompt=None, verbose=False, **_kwargs):
            self.generate_calls.append(
                {"audio": audio, "language": language, "system_prompt": system_prompt}
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


def test_glm_mlx_empty_text_returns_no_segments(monkeypatch):
    """Test that empty model output yields no segments (matches glm-asr behavior)."""

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

    assert segments == []
