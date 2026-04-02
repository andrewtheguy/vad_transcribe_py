"""Whisper backend using HuggingFace Transformers."""

import sys
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
    WHISPER_HARD_LIMIT_SECONDS,
    WHISPER_SOFT_LIMIT_SECONDS,
)

WHISPER_DEFAULT_MODEL = "large-v3-turbo"


def _get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Auto-detect best device and dtype."""
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
        language: str,
        model: str,
        chinese_conversion: ChineseConversion = 'none',
    ):
        super().__init__(language, chinese_conversion)
        self.model = model
        self.pipe: Any = None

        print(f"Loading {self.model} model...", file=sys.stderr)
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
        device, torch_dtype = _get_device_and_dtype()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch_dtype,
            device=device,
        )

        print(f"Whisper model loaded: {model_id} on {device}", file=sys.stderr)

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return segments with sub-sentence timestamps."""
        result = self.pipe(
            audio.copy(),
            return_timestamps=True,
            generate_kwargs={"language": self.language},
        )

        return [
            self._make_segment(
                chunk["text"],
                start_offset + chunk["timestamp"][0],
                start_offset + (chunk["timestamp"][1] if chunk["timestamp"][1] is not None else len(audio) / TARGET_SAMPLE_RATE),
            )
            for chunk in result["chunks"]
        ]
