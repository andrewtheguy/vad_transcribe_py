"""Whisper backend using HuggingFace Transformers."""

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
    conditioning_context,
)
from vad_transcribe_py.vad_processor import (
    WHISPER_HARD_LIMIT_SECONDS,
    WHISPER_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

WHISPER_DEFAULT_MODEL = "large-v3-turbo"

# Models that need language code remapping (e.g. yue → zh)
_MODEL_LANGUAGE_OVERRIDES: dict[str, dict[str, str]] = {
    "alvanlii/whisper-small-cantonese": {"yue": "zh"},
}


def _get_device_and_dtype(device: str | None = None) -> tuple[str, torch.dtype]:
    """Resolve device and dtype for Whisper.

    When *device* is ``None``, auto-detect: cuda > mps > cpu.
    """
    if device is None:
        if torch.cuda.is_available():
            resolved = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"
    else:
        resolved = device

    if resolved == "cuda":
        return "cuda:0", torch.float16
    elif resolved == "mps":
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def _resolve_whisper_model_id(model: str) -> str:
    """Resolve short whisper model name to full HuggingFace model ID."""
    if '/' in model:
        return model
    return f"openai/whisper-{model}"


class WhisperBackend(TranscriberBase):
    """Transcriber backend using Whisper via HuggingFace Transformers."""

    def __init__(
        self,
        language: str | None,
        model: str,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        condition: bool = True,
        sub_timestamps: bool = True,
        device: str | None = None,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self.model = model
        self.pipe: Any = None
        self._condition = condition
        self._sub_timestamps = sub_timestamps
        self._prompt_ids: Any = None
        self._prior_line: str = ""
        self._processor: Any = None
        self._requested_device = device
        self._device: str = "cpu"

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        logger.info("Loading %s model...", self.model)
        self._load_whisper()

    @property
    def hard_limit_seconds(self) -> int:
        return WHISPER_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return WHISPER_SOFT_LIMIT_SECONDS

    def _load_whisper(self) -> None:
        """Load Whisper model via HuggingFace Transformers pipeline."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            raise ImportError(
                "transformers is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]'. "
                "For VAD-only mode without transcription, use the 'split' command instead."
            )

        model_id = _resolve_whisper_model_id(self.model)
        self._device, torch_dtype = _get_device_and_dtype(device=self._requested_device)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self._device)

        self._processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            dtype=torch_dtype,
            device=self._device,
        )

        logger.info("Whisper model loaded: %s on %s", model_id, self._device)

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return segments with sub-sentence timestamps."""
        generate_kwargs: dict[str, Any] = {
            "language": _MODEL_LANGUAGE_OVERRIDES.get(self.model, {}).get(self.language, self.language) if self.language else None,
        }
        if self._prompt_ids is not None:
            generate_kwargs["prompt_ids"] = self._prompt_ids

        result = self.pipe(
            audio.copy(),
            return_timestamps=self._sub_timestamps,
            generate_kwargs=generate_kwargs,
        )

        if self._sub_timestamps:
            segments = [
                self._make_segment(
                    chunk["text"],
                    start_offset + chunk["timestamp"][0],
                    start_offset + (chunk["timestamp"][1] if chunk["timestamp"][1] is not None else len(audio) / TARGET_SAMPLE_RATE),
                )
                for chunk in result["chunks"]
            ]
        else:
            text = result["text"]
            segments = [self._make_segment(text, start_offset, start_offset + len(audio) / TARGET_SAMPLE_RATE)]

        # Update prompt with this segment's output for next-segment conditioning
        if self._condition and segments:
            output_text = " ".join(seg.text for seg in segments)
            safe = conditioning_context(output_text, self._prior_line)
            if safe:
                self._prompt_ids = self._processor.get_prompt_ids(safe, return_tensors="pt").to(self._device)
            else:
                self._prompt_ids = None
            self._prior_line = output_text.strip()

        return segments
