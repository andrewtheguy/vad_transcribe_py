"""Model registry and constants for Moonshine ONNX models.

Trimmed to English (streaming) and Chinese (non-streaming) only.
"""

from enum import IntEnum

SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = "en"
DEFAULT_MODEL = "small-streaming"

# Hard limits by architecture type
NON_STREAMING_HARD_MAX_SPEECH_SECONDS = 9
STREAMING_HARD_MAX_SPEECH_SECONDS = 60


class ModelArch(IntEnum):
    TINY = 0
    BASE = 1
    TINY_STREAMING = 2
    BASE_STREAMING = 3
    SMALL_STREAMING = 4
    MEDIUM_STREAMING = 5


STREAMING_ARCHS = {
    ModelArch.TINY_STREAMING,
    ModelArch.BASE_STREAMING,
    ModelArch.SMALL_STREAMING,
    ModelArch.MEDIUM_STREAMING,
}


def _moonshine_model(
    model_name: str,
    model_arch: ModelArch,
    download_url: str,
) -> dict[str, object]:
    is_streaming = model_arch in STREAMING_ARCHS
    return {
        "model_name": model_name,
        "model_arch": model_arch,
        "download_url": download_url,
        "hard_max_speech_seconds": STREAMING_HARD_MAX_SPEECH_SECONDS if is_streaming else NON_STREAMING_HARD_MAX_SPEECH_SECONDS,
    }


MODEL_INFO = {
    "en": {
        "english_name": "English",
        "models": [
            _moonshine_model(
                "medium-streaming-en",
                ModelArch.MEDIUM_STREAMING,
                "https://download.moonshine.ai/model/medium-streaming-en/quantized",
            ),
            _moonshine_model(
                "small-streaming-en",
                ModelArch.SMALL_STREAMING,
                "https://download.moonshine.ai/model/small-streaming-en/quantized",
            ),
            _moonshine_model(
                "base-en",
                ModelArch.BASE,
                "https://download.moonshine.ai/model/base-en/quantized/base-en",
            ),
            _moonshine_model(
                "tiny-streaming-en",
                ModelArch.TINY_STREAMING,
                "https://download.moonshine.ai/model/tiny-streaming-en/quantized",
            ),
            _moonshine_model(
                "tiny-en",
                ModelArch.TINY,
                "https://download.moonshine.ai/model/tiny-en/quantized/tiny-en",
            ),
        ],
    },
    "zh": {
        "english_name": "Chinese",
        "models": [
            _moonshine_model(
                "base-zh",
                ModelArch.BASE,
                "https://download.moonshine.ai/model/base-zh/quantized/base-zh",
            ),
            _moonshine_model(
                "tiny-zh",
                ModelArch.TINY,
                "https://download.moonshine.ai/model/tiny-zh/quantized/tiny-zh",
            ),
        ],
    },
}

LANGUAGE_NAMES = tuple(MODEL_INFO.keys())


def _scoped_model_name(language: str, model_name: str) -> str:
    suffix = f"-{language}"
    if model_name.endswith(suffix):
        return model_name[: -len(suffix)]
    return model_name


MODEL_NAMES_BY_LANGUAGE = {
    language: tuple(
        _scoped_model_name(language, model["model_name"])
        for model in info["models"]
    )
    for language, info in MODEL_INFO.items()
}


def default_model_for_language(language: str) -> str:
    """Return the default CLI model name for a language."""
    if language == DEFAULT_LANGUAGE:
        return DEFAULT_MODEL
    return MODEL_NAMES_BY_LANGUAGE[language][0]


def resolve_model(
    language: str = DEFAULT_LANGUAGE, model: str | None = None
) -> tuple[str, str, ModelArch, bool, str, int]:
    """Resolve a language + scoped model name to model runtime settings.

    Returns:
        (model_name, language, model_arch, is_streaming, download_url,
         hard_max_speech_seconds)
    """
    language = language.lower()
    if language not in MODEL_INFO:
        available = ", ".join(LANGUAGE_NAMES)
        raise ValueError(f"Unknown language: {language}. Available: {available}")

    if model is None:
        model = default_model_for_language(language)
    model = model.lower()

    for model_info in MODEL_INFO[language]["models"]:
        if _scoped_model_name(language, model_info["model_name"]) != model:
            continue
        arch = model_info["model_arch"]
        is_streaming = arch in STREAMING_ARCHS
        return (
            model_info["model_name"],
            language,
            arch,
            is_streaming,
            model_info["download_url"],
            model_info["hard_max_speech_seconds"],
        )

    available = ", ".join(MODEL_NAMES_BY_LANGUAGE[language])
    raise ValueError(
        f"Unknown model for language {language}: {model}. Available: {available}"
    )
