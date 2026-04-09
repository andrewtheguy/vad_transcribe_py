"""Qwen3-ASR backend using the qwencandle Rust package (PyO3 bindings)."""

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
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
    "es": "Spanish",
}


class QwenASRRsBackend(TranscriberBase):
    """Transcriber backend using Qwen3-ASR via the qwencandle Rust library."""

    def __init__(
        self,
        language: str,
        model: str | None = None,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        device: str = 'cpu',
        condition: bool = True,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        if num_threads is not None:
            os.environ["RAYON_NUM_THREADS"] = str(num_threads)
        self._device = device
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
        """Load Qwen3-ASR model via the qwencandle Rust bindings."""
        try:
            from qwencandle import QwenAsr
        except ImportError:
            raise ImportError(
                "qwencandle is not installed. "
                "Install with: uv pip install qwencandle"
            )

        logger.info("Loading qwencandle model (device=%s)...", self._device)
        self._model = QwenAsr(model_id=model, device=self._device)
        logger.info("qwencandle model loaded on %s", self._device)

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return a single segment."""
        assert self._model is not None
        qwen_language = _LANGUAGE_MAP.get(self.language)

        context = self._previous_text if self._condition else None

        text: str = self._model.transcribe(  # pyright: ignore[reportAttributeAccessIssue]
            audio,
            language=qwen_language,
            context=context,
        )

        if self._condition and text.strip() and not is_repetitive(text.strip()):
            self._previous_text = text.strip()
        elif self._condition and is_repetitive(text.strip()):
            self._previous_text = ""

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time)]
