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

# Models that need language code remapping (e.g. yue → zh)
_MODEL_LANGUAGE_OVERRIDES: dict[str, dict[str, str]] = {
    "alvanlii/whisper-small-cantonese": {"yue": "zh"},
}


def _merge_transcript_text(left: str, right: str) -> str:
    """Join deferred incomplete text with the next overlapped decode."""
    if not left:
        return right
    if not right:
        return left

    max_overlap = min(len(left), len(right))
    left_folded = left.casefold()
    right_folded = right.casefold()
    for overlap in range(max_overlap, 0, -1):
        if left_folded[-overlap:] == right_folded[:overlap]:
            return left + right[overlap:]

    if left[-1:].isspace() and right[:1].isspace():
        return left + right.lstrip()
    return left + right


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
        self._overlap_audio: npt.NDArray[np.float32] | None = None
        self._overlap_start_offset: float | None = None
        self._pending_incomplete_segment: TranscribedSegment | None = None
        self.last_segment_debug: dict[str, object] = {}

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
        original_audio_duration = len(audio) / TARGET_SAMPLE_RATE
        original_end_offset = start_offset + original_audio_duration
        effective_audio: npt.NDArray[np.float32] = audio
        effective_start_offset: float = start_offset

        pending_overlap_audio = self._overlap_audio
        pending_overlap_start_offset = self._overlap_start_offset
        overlap_applied = False
        overlap_duration = 0.0
        if (
            self._sub_timestamps
            and pending_overlap_audio is not None
            and pending_overlap_start_offset is not None
            and len(pending_overlap_audio) > 0
        ):
            overlap_applied = True
            effective_audio = np.concatenate([pending_overlap_audio, audio]).astype(np.float32, copy=False)
            effective_start_offset = pending_overlap_start_offset
            overlap_duration = len(pending_overlap_audio) / TARGET_SAMPLE_RATE

        pending_incomplete_segment = self._pending_incomplete_segment if overlap_applied else None

        # A pending overlap is single-use. This call will decide whether to store
        # a fresh overlap for the next VAD segment.
        self._overlap_audio = None
        self._overlap_start_offset = None
        if overlap_applied:
            self._pending_incomplete_segment = None
        self.last_segment_debug = {}

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
            effective_audio.copy(),
            return_timestamps=self._sub_timestamps,
            generate_kwargs=generate_kwargs,
        )

        conditioning_segments: list[TranscribedSegment] = []

        if self._sub_timestamps:
            raw_chunks = result["chunks"]
            segments = []
            skipped_overlap_chunks = 0
            effective_audio_duration = len(effective_audio) / TARGET_SAMPLE_RATE
            held_incomplete_segment: TranscribedSegment | None = None

            for chunk_index, chunk in enumerate(raw_chunks):
                timestamp_start, timestamp_end = chunk["timestamp"]
                timestamp_missing_start = timestamp_start is None
                timestamp_missing_end = timestamp_end is None
                is_final_chunk = chunk_index == len(raw_chunks) - 1

                chunk_start_relative = float(timestamp_start) if timestamp_start is not None else 0.0
                chunk_end_relative = (
                    float(timestamp_end)
                    if timestamp_end is not None
                    else effective_audio_duration
                )
                chunk_start = effective_start_offset + chunk_start_relative
                chunk_end = effective_start_offset + chunk_end_relative

                overlap_only = overlap_applied and chunk_end <= start_offset
                if overlap_only:
                    skipped_overlap_chunks += 1
                    continue

                starts_in_overlap = overlap_applied and chunk_start < start_offset
                debug: dict[str, object] = {
                    "ended_mid_sentence": timestamp_missing_end,
                    "whisper_timestamp_missing_start": timestamp_missing_start,
                    "whisper_timestamp_missing_end": timestamp_missing_end,
                    "overlap_applied": overlap_applied,
                    "starts_in_overlap": starts_in_overlap,
                    "vad_start_ms": round(start_offset * 1000),
                    "vad_end_ms": round(original_end_offset * 1000),
                }
                if overlap_applied:
                    debug.update(
                        {
                            "overlap_start_ms": round(effective_start_offset * 1000),
                            "overlap_cutoff_ms": round(start_offset * 1000),
                            "overlap_duration_ms": round(overlap_duration * 1000),
                        }
                    )

                segments.append(
                    self._make_segment(
                        chunk["text"],
                        chunk_start,
                        chunk_end,
                        debug=debug,
                    )
                )
                if is_final_chunk and timestamp_missing_end:
                    held_incomplete_segment = segments.pop()

            final_chunk = raw_chunks[-1] if raw_chunks else None
            final_timestamp_start = None
            final_timestamp_end = None
            if final_chunk is not None:
                final_timestamp_start, final_timestamp_end = final_chunk["timestamp"]

            final_missing_end = final_timestamp_end is None if final_chunk is not None else False
            pending_incomplete_merged = False
            if pending_incomplete_segment is not None:
                pending_debug: dict[str, object] = {
                    "merged_incomplete": True,
                    "pending_incomplete_start_ms": round(pending_incomplete_segment.start * 1000),
                    "pending_incomplete_end_ms": round(pending_incomplete_segment.end * 1000),
                    "pending_incomplete_text_chars": len(pending_incomplete_segment.text),
                }
                if segments:
                    first_segment = segments[0]
                    merged_debug: dict[str, object] = {**first_segment.debug, **pending_debug}
                    segments[0] = self._make_segment(
                        _merge_transcript_text(pending_incomplete_segment.text, first_segment.text),
                        pending_incomplete_segment.start,
                        first_segment.end,
                        debug=merged_debug,
                    )
                    pending_incomplete_merged = True
                elif held_incomplete_segment is not None:
                    merged_debug = {**held_incomplete_segment.debug, **pending_debug}
                    held_incomplete_segment = self._make_segment(
                        _merge_transcript_text(
                            pending_incomplete_segment.text,
                            held_incomplete_segment.text,
                        ),
                        pending_incomplete_segment.start,
                        held_incomplete_segment.end,
                        debug=merged_debug,
                    )
                    pending_incomplete_merged = True
                else:
                    self._pending_incomplete_segment = pending_incomplete_segment

            next_overlap_stored = False
            next_overlap_start = None
            next_overlap_duration = 0.0
            overlap_skipped_reason = None
            if final_missing_end:
                if final_timestamp_start is None:
                    overlap_skipped_reason = "missing_start_timestamp"
                else:
                    overlap_start_relative = max(
                        0.0,
                        min(float(final_timestamp_start), effective_audio_duration),
                    )
                    overlap_start_sample = int(round(overlap_start_relative * TARGET_SAMPLE_RATE))
                    overlap_for_next = effective_audio[overlap_start_sample:]
                    if len(overlap_for_next) > 0:
                        self._overlap_audio = np.copy(overlap_for_next)
                        next_overlap_start = effective_start_offset + overlap_start_relative
                        self._overlap_start_offset = next_overlap_start
                        next_overlap_duration = len(overlap_for_next) / TARGET_SAMPLE_RATE
                        next_overlap_stored = True
                    else:
                        overlap_skipped_reason = "empty_overlap"
                if held_incomplete_segment is not None:
                    self._pending_incomplete_segment = held_incomplete_segment

            self.last_segment_debug = {
                "overlap_applied": overlap_applied,
                "overlap_start_ms": round(effective_start_offset * 1000) if overlap_applied else None,
                "overlap_cutoff_ms": round(start_offset * 1000) if overlap_applied else None,
                "overlap_duration_ms": round(overlap_duration * 1000),
                "skipped_overlap_chunk_count": skipped_overlap_chunks,
                "raw_chunk_count": len(raw_chunks),
                "emitted_chunk_count": len(segments),
                "ended_mid_sentence": final_missing_end,
                "whisper_final_timestamp_missing_end": final_missing_end,
                "held_incomplete": held_incomplete_segment is not None,
                "pending_incomplete_merged": pending_incomplete_merged,
                "next_overlap_stored": next_overlap_stored,
                "next_overlap_start_ms": round(next_overlap_start * 1000) if next_overlap_start is not None else None,
                "next_overlap_duration_ms": round(next_overlap_duration * 1000),
            }
            if held_incomplete_segment is not None:
                self.last_segment_debug.update(
                    {
                        "held_incomplete_start_ms": round(held_incomplete_segment.start * 1000),
                        "held_incomplete_end_ms": round(held_incomplete_segment.end * 1000),
                    }
                )
            if overlap_skipped_reason is not None:
                self.last_segment_debug["next_overlap_skipped_reason"] = overlap_skipped_reason
            conditioning_segments = [*segments]
            if held_incomplete_segment is not None:
                conditioning_segments.append(held_incomplete_segment)
        else:
            text = result["text"]
            segments = [self._make_segment(text, start_offset, original_end_offset)]
            self.last_segment_debug = {
                "overlap_applied": False,
                "ended_mid_sentence": False,
            }
            conditioning_segments = segments

        # Update prompt with this segment's output for next-segment conditioning
        if self._condition and conditioning_segments:
            output_text = " ".join(seg.text for seg in conditioning_segments)
            safe = conditioning_context(output_text, self._prior_line)
            if safe:
                self._prompt_ids = self._processor.get_prompt_ids(safe, return_tensors="pt").to(self._device)
            else:
                self._prompt_ids = None
            self._prior_line = output_text.strip()

        return segments
