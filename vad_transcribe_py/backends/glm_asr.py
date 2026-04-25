"""GLM-ASR backend using HuggingFace Transformers."""

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
from vad_transcribe_py.vad_processor import (
    GLM_ASR_HARD_LIMIT_SECONDS,
    GLM_ASR_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

GLM_ASR_DEFAULT_MODEL = "zai-org/GLM-ASR-Nano-2512"
GLM_ASR_MAX_NEW_TOKENS = 512

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


def _detect_device(device: str | None = None) -> str:
    """Resolve a torch device for GLM-ASR."""
    if device is None:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        return "cuda:0"
    if device == "metal":
        return "mps"
    return device


class GLMASRBackend(TranscriberBase):
    """Transcriber backend using zai-org GLM-ASR via Transformers."""

    def __init__(
        self,
        language: str | None,
        model: str = GLM_ASR_DEFAULT_MODEL,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        condition: bool = True,
        device: str | None = None,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self.model = model
        self._device = _detect_device(device)
        self._processor: Any = None
        self._model: Any = None

        if num_threads is not None:
            logger.warning("num_threads=%d ignored; glm-asr backend has no threading knob", num_threads)
        if not condition:
            logger.warning("condition=False ignored; glm-asr backend does not support previous-segment conditioning")

        logger.info("Loading %s model via Transformers on %s...", self.model, self._device)
        self._load_model()

    @property
    def hard_limit_seconds(self) -> int:
        return GLM_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return GLM_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self) -> None:
        """Load GLM-ASR via HuggingFace Transformers."""
        try:
            from transformers import AutoProcessor, GlmAsrForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers with GLM-ASR support is not installed. "
                "Install with: uv add 'transformers>=5.5.4'"
            )

        self._processor = AutoProcessor.from_pretrained(self.model)
        self._model = GlmAsrForConditionalGeneration.from_pretrained(
            self.model,
            dtype="auto",
        )
        self._model.to(self._device)
        self._model.eval()
        logger.info("GLM-ASR model loaded: %s on %s", self.model, self._device)

    def _get_prompt(self) -> str | None:
        """Return a language-specific transcription prompt, or None for model default."""
        if self.language is None:
            return None

        language_name = _LANGUAGE_MAP.get(self.language)
        if language_name is None:
            raise ValueError(
                f"Unrecognized language code '{self.language}'. "
                f"Available: {', '.join(_LANGUAGE_MAP)}"
            )

        return f"Transcribe the input speech in {language_name}."

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        start_offset: float = 0.0,
    ) -> list[TranscribedSegment]:
        """Transcribe a single audio segment and return one TranscribedSegment."""
        assert self._processor is not None
        assert self._model is not None

        prompt = self._get_prompt()
        inputs = self._processor.apply_transcription_request(
            audio.copy(),
            prompt=prompt,
            return_tensors="pt",
        )
        model_dtype = getattr(self._model, "dtype", None)
        if isinstance(model_dtype, torch.dtype):
            inputs = inputs.to(device=self._device, dtype=model_dtype)
        else:
            inputs = inputs.to(self._device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=GLM_ASR_MAX_NEW_TOKENS,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_length:]
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time)] if text else []
