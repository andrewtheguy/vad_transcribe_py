"""Qwen3-ASR backend using the qwen_burn Rust package (PyO3 bindings)."""

import logging
import os

import numpy as np
import numpy.typing as npt

from vad_transcribe_py._types import (
    TARGET_SAMPLE_RATE,
    ChineseConversion,
    TranscribedSegment,
    TranscriberBase,
    is_repetitive,
)
from vad_transcribe_py.vad_processor import (
    QWEN_ASR_HARD_LIMIT_SECONDS,
    QWEN_ASR_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

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


class QwenASRRsBackend(TranscriberBase):
    """Transcriber backend using Qwen3-ASR via the qwen_burn Rust library."""

    def __init__(
        self,
        language: str | None,
        model: str | None = None,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        condition: bool = True,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        if num_threads is not None:
            os.environ["RAYON_NUM_THREADS"] = str(num_threads)
        self._condition = condition
        self._previous_text: str = ""
        self._model: object = None

        self._load_model(model)

    @property
    def hard_limit_seconds(self) -> int:
        return QWEN_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return QWEN_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self, model: str | None) -> None:
        """Load Qwen3-ASR model via the qwen_burn Rust bindings."""
        try:
            from qwen_burn import QwenAsr
        except ImportError:
            raise ImportError(
                "qwen-burn is not installed. "
                "Install with: uv pip install qwen-burn"
            )

        logger.info("Loading qwen_burn model...")
        self._model = QwenAsr(model_id=model)
        logger.info("qwen_burn model loaded")

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return a single segment."""
        assert self._model is not None
        qwen_language = _LANGUAGE_MAP.get(self.language) if self.language else None
        if self.language and qwen_language is None:
            raise ValueError(f"Unrecognized language code '{self.language}'. Available: {', '.join(_LANGUAGE_MAP)}")

        context = self._previous_text if self._condition else None

        text: str = self._model.transcribe(  # pyright: ignore[reportAttributeAccessIssue]
            audio,
            language=qwen_language,
            context=context,
        )

        if self._condition:
            stripped = text.strip()
            if stripped and not is_repetitive(stripped):
                self._previous_text = stripped
            elif is_repetitive(stripped):
                self._previous_text = ""

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time)]
