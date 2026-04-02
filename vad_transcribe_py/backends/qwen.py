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

def _resolve_eos_token_ids(processor: Any) -> list[int]:
    """Resolve Qwen EOS token ids from the runtime tokenizer."""
    tokenizer = processor.tokenizer
    token_ids: list[int] = []

    for attr in ("eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int) and token_id not in token_ids:
            token_ids.append(token_id)

    if not token_ids:
        raise RuntimeError("Qwen3-ASR tokenizer does not expose EOS token ids")

    return token_ids


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
        self._parse_asr_output: Any = None
        self._eos_token_ids: list[int] = []

        logger.info("Loading %s model...", self.model)
        self._load_model()

    @property
    def hard_limit_seconds(self) -> int:
        return QWEN_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return QWEN_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self) -> None:
        """Load Qwen3-ASR model via the non-streaming Transformers backend."""
        try:
            from qwen_asr import Qwen3ASRModel
            from qwen_asr.inference.utils import parse_asr_output
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
        self._parse_asr_output = parse_asr_output

        backend = getattr(self._qwen_model, "backend", None)
        if backend != "transformers":
            raise RuntimeError(
                "Qwen3-ASR must use the non-streaming transformers backend; "
                f"got {backend!r}"
            )
        self._eos_token_ids = _resolve_eos_token_ids(self._qwen_model.processor)

        logger.info(
            "Qwen3-ASR model loaded: %s on %s (mode=non-streaming)",
            self.model,
            device,
        )

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return a single segment."""
        qwen_language = _LANGUAGE_MAP.get(self.language)

        try:
            results = self._qwen_model.transcribe(
                audio=(audio, TARGET_SAMPLE_RATE),
                language=qwen_language,
            )

            text: str = results[0].text
            end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
            return [self._make_segment(text, start_offset, end_time)]
        finally:
            # qwen-asr doesn't clean up after generate(), causing MPS memory leak:
            # - rope_deltas persists on the thinker module between calls
            # - KV-cache isn't released from the MPS allocator
            thinker = self._qwen_model.model.thinker
            if hasattr(thinker, "rope_deltas"):
                thinker.rope_deltas = None
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
