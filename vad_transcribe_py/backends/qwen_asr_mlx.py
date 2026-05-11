"""Qwen3-ASR backend using the mlx-audio package (Apple Silicon / Metal)."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from vad_transcribe_py._types import (
    TranscribedSegment,
    TranscriberBase,
)
from vad_transcribe_py._utils import (
    TARGET_SAMPLE_RATE,
    ChineseConversion,
    conditioning_context,
    is_near_duplicate,
)
from vad_transcribe_py.vad_processor import (
    QWEN_ASR_HARD_LIMIT_SECONDS,
    QWEN_ASR_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

QWEN_ASR_MLX_DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-bf16"
# mlx-audio defaults to max_tokens=8192, which lets larger Qwen3-ASR variants
# (e.g. 1.7B-bf16) run away on music or other non-speech audio. 500 covers the
# longest 60s speech segment we ever feed in.
QWEN_ASR_MLX_MAX_TOKENS = 500

_LANGUAGE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


class QwenASRMLXBackend(TranscriberBase):
    """Transcriber backend using Qwen3-ASR via mlx-audio (Apple Silicon / Metal)."""

    def __init__(
        self,
        language: str | None,
        model: str = QWEN_ASR_MLX_DEFAULT_MODEL,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        condition: bool = True,
        device: str | None = None,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self.model = model
        self._condition = condition
        self._previous_text: str = ""
        self._prior_line: str = ""
        self._mlx_model: Any = None

        if device is not None and device != "metal":
            logger.warning("device=%r ignored; mlx-audio backend always uses Metal", device)
        if num_threads is not None:
            logger.warning("num_threads=%d ignored; mlx-audio has no threading knob", num_threads)

        logger.info("Loading %s model via mlx-audio...", self.model)
        self._load_model()

    @property
    def hard_limit_seconds(self) -> int:
        return QWEN_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return QWEN_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self) -> None:
        """Load Qwen3-ASR via mlx-audio. First run downloads the model from HuggingFace."""
        try:
            from mlx_audio.stt.utils import load_model  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "mlx-audio is not installed (Apple Silicon only). "
                "Install with: uv add mlx-audio"
            )

        self._mlx_model = load_model(self.model)
        logger.info("mlx-audio Qwen3-ASR model loaded: %s", self.model)

    def transcribe(
        self, audio: npt.NDArray[np.float32], start_offset: float = 0.0,
    ) -> list[TranscribedSegment]:
        """Transcribe a single audio segment and return one TranscribedSegment."""
        assert self._mlx_model is not None

        qwen_language = _LANGUAGE_MAP.get(self.language) if self.language else None
        if self.language and qwen_language is None:
            raise ValueError(
                f"Unrecognized language code '{self.language}'. "
                f"Available: {', '.join(_LANGUAGE_MAP)}"
            )

        sys_prompt = self._previous_text if (self._condition and self._previous_text) else None
        used_prompt = sys_prompt is not None

        output = self._mlx_model.generate(
            audio,
            language=qwen_language,
            system_prompt=sys_prompt,
            max_tokens=QWEN_ASR_MLX_MAX_TOKENS,
            verbose=False,
        )
        text: str = output.text

        prompt_retry = False
        if used_prompt and is_near_duplicate(text, self._prior_line):
            logger.info("Retrying segment without conditioning prompt (near-duplicate of prior)")
            output = self._mlx_model.generate(
                audio,
                language=qwen_language,
                system_prompt=None,
                max_tokens=QWEN_ASR_MLX_MAX_TOKENS,
                verbose=False,
            )
            text = output.text
            prompt_retry = True

        if self._condition:
            self._previous_text = conditioning_context(text, self._prior_line)
            self._prior_line = text.strip()

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time, prompt_retry=prompt_retry)]
