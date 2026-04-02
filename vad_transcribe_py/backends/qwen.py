"""Qwen3-ASR backend using the qwen-asr package."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from vad_transcribe_py._types import (
    TARGET_SAMPLE_RATE,
    ChineseConversion,
    TranscribedSegment,
    TranscriberBase,
)

logger = logging.getLogger(__name__)

QWEN_ASR_DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
QWEN_ASR_HARD_LIMIT_SECONDS = 30
QWEN_ASR_SOFT_LIMIT_SECONDS = 6.0

_LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "yue": "Cantonese",
}


def _get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Auto-detect best device and dtype for Qwen3-ASR."""
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


class QwenASRBackend(TranscriberBase):
    """Transcriber backend using Qwen3-ASR."""

    def __init__(
        self,
        language: str,
        model: str = QWEN_ASR_DEFAULT_MODEL,
        chinese_conversion: ChineseConversion = 'none',
    ):
        super().__init__(language, chinese_conversion)
        self.model = model
        self._qwen_model: Any = None

        logger.info("Loading %s model...", self.model)
        self._load_model()

    @property
    def hard_limit_seconds(self) -> int:
        return QWEN_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return QWEN_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self) -> None:
        """Load Qwen3-ASR model via qwen-asr package."""
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "qwen-asr is not installed. "
                "Install with: uv pip install qwen-asr"
            )

        #import transformers
        #transformers.logging.set_verbosity_error()

        device, torch_dtype = _get_device_and_dtype()

        self._qwen_model = Qwen3ASRModel.from_pretrained(
            self.model,
            dtype=torch_dtype,
            device_map=device,
        )

        logger.info("Qwen3-ASR model loaded: %s on %s", self.model, device)

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return a single segment."""
        qwen_language = _LANGUAGE_MAP.get(self.language)

        results = self._qwen_model.transcribe(
            audio=(audio, TARGET_SAMPLE_RATE),
            language=qwen_language,
        )

        text: str = results[0].text
        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        result = [self._make_segment(text, start_offset, end_time)]

        # Clear retained rope_deltas to prevent memory growth across segments.
        # The model stores this tensor as instance state during generate() and
        # never clears it, which keeps old attention mask tensors alive.
        thinker = self._qwen_model.model.thinker
        if hasattr(thinker, 'rope_deltas'):
            thinker.rope_deltas = None

        return result
