"""Whisper backend using HuggingFace Transformers."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from vad_transcribe_py._types import (
    TranscribedSegment,
    TranscriberBase,
)
from vad_transcribe_py._utils import (
    TARGET_SAMPLE_RATE,
    ChineseConversion,
    conditioning_context,
)
from vad_transcribe_py.vad_processor import (
    WHISPER_HARD_LIMIT_NO_SUB_TIMESTAMPS_SECONDS,
    WHISPER_HARD_LIMIT_SECONDS,
    WHISPER_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

WHISPER_DEFAULT_MODEL = "large-v3-turbo"

# Audio tail prepended to the next contiguous call so Whisper has acoustic
# context for words cut off mid-sentence at a hard-limit boundary.
_OVERLAP_SECONDS = 2.0
_ADJACENT_TOLERANCE_SECONDS = 0.1

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
        self._tail_audio: npt.NDArray[np.float32] | None = None
        self._prior_end_time: float | None = None

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        logger.info("Loading %s model...", self.model)
        self._load_whisper()

    @property
    def hard_limit_seconds(self) -> int:
        # HF's pipeline only does long-form decoding when return_timestamps=True;
        # with --no-sub-timestamps it falls back to single-window inference and
        # would error on audio past Whisper's native 30s window.
        if self._sub_timestamps:
            return WHISPER_HARD_LIMIT_SECONDS
        return WHISPER_HARD_LIMIT_NO_SUB_TIMESTAMPS_SECONDS

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
        # Prepend the tail of the prior call when this segment is contiguous,
        # so Whisper has acoustic context to resume a sentence cut off at a
        # hard-limit boundary. Only safe in sub_timestamps mode where we can
        # dedupe on chunk timestamps.
        prepend_tail = (
            self._sub_timestamps
            and self._tail_audio is not None
            and self._prior_end_time is not None
            and abs(start_offset - self._prior_end_time) < _ADJACENT_TOLERANCE_SECONDS
        )
        if prepend_tail:
            assert self._tail_audio is not None
            pipeline_audio = np.concatenate([self._tail_audio, audio])
            overlap_seconds = len(self._tail_audio) / TARGET_SAMPLE_RATE
        else:
            pipeline_audio = audio
            overlap_seconds = 0.0

        generate_kwargs: dict[str, Any] = {
            "language": _MODEL_LANGUAGE_OVERRIDES.get(self.model, {}).get(self.language, self.language) if self.language else None,
            "condition_on_prev_tokens": True,
            # "compression_ratio_threshold": 1.35,
            # "logprob_threshold": -1.0,
            # "no_speech_threshold": 0.6,
            # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        }
        if self._prompt_ids is not None:
            generate_kwargs["prompt_ids"] = self._prompt_ids

        result = self.pipe(
            pipeline_audio.copy(),
            return_timestamps=self._sub_timestamps,
            generate_kwargs=generate_kwargs,
        )

        if self._sub_timestamps:
            pipeline_duration = len(pipeline_audio) / TARGET_SAMPLE_RATE
            segments: list[TranscribedSegment] = []
            for chunk in result["chunks"]:
                chunk_start = chunk["timestamp"][0]
                chunk_end = chunk["timestamp"][1] if chunk["timestamp"][1] is not None else pipeline_duration
                # Drop chunks whose midpoint lies in the prepended overlap region.
                if (chunk_start + chunk_end) / 2 < overlap_seconds:
                    continue
                adjusted_start = max(0.0, chunk_start - overlap_seconds)
                adjusted_end = max(adjusted_start, chunk_end - overlap_seconds)
                segments.append(
                    self._make_segment(
                        chunk["text"],
                        start_offset + adjusted_start,
                        start_offset + adjusted_end,
                    )
                )
        else:
            text = result["text"]
            segments = [self._make_segment(text, start_offset, start_offset + len(audio) / TARGET_SAMPLE_RATE)]

        if self._sub_timestamps:
            tail_samples = int(_OVERLAP_SECONDS * TARGET_SAMPLE_RATE)
            self._tail_audio = audio[-tail_samples:].copy() if len(audio) >= tail_samples else audio.copy()
            self._prior_end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE

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
