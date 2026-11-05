"""NVIDIA Nemotron ASR (cache-aware FastConformer RNNT) backend via HuggingFace Transformers.

The model is transcribed offline (whole VAD segment in one pass). Because it is a
non-autoregressive RNNT model it does not hallucinate or loop on long input, which
makes it suitable for multi-minute VAD segments. Sentence-level sub-timestamps are
derived from the RNNT token durations (each blank advances one encoder frame).
"""

import logging
from dataclasses import dataclass
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
)
from vad_transcribe_py.vad_processor import (
    NEMOTRON_ASR_HARD_LIMIT_SECONDS,
    NEMOTRON_ASR_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

NEMOTRON_ASR_DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
# Encoder frame duration: 10ms mel hop x 8x subsampling.
NEMOTRON_FRAME_SECONDS = 0.08
_AUTO_LANGUAGE = "auto"
_SENTENCE_END_CHARS = (".", "?", "!", "。", "？", "！")


def _detect_device(device: str | None = None) -> str:
    """Resolve a torch device for Nemotron ASR."""
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


def _dtype_for_device(device: str) -> torch.dtype:
    """fp16 on GPU (halves activation memory on long segments); fp32 on CPU."""
    return torch.float32 if device == "cpu" else torch.float16


def resolve_language(language: str | None, prompt_dictionary: dict[str, int]) -> str:
    """Map a language code to a key the processor accepts.

    None means automatic detection. Exact keys (``en``, ``de-DE``) pass through;
    bare codes without an exact entry (``zh``, ``ja``) map to the first
    ``<code>-XX`` locale in the prompt dictionary.
    """
    if language is None:
        return _AUTO_LANGUAGE
    if language in prompt_dictionary:
        return language
    prefix = f"{language}-"
    for key in prompt_dictionary:
        if key.startswith(prefix):
            return key
    raise ValueError(
        f"Unrecognized language code '{language}' for Nemotron ASR. "
        f"Available: {', '.join(sorted(prompt_dictionary))}"
    )


@dataclass
class _TimedToken:
    token_id: int
    piece: str
    frame: int


def timed_tokens(
    sequence: list[int],
    durations: list[int],
    blank_token_id: int,
    pieces: list[str],
) -> list[_TimedToken]:
    """Drop blanks and attach an encoder-frame index to each emitted token.

    ``durations`` can carry one extra trailing step beyond ``sequence``; it only
    affects frames after the last token, so it is ignored.
    """
    tokens: list[_TimedToken] = []
    frame = 0
    for token_id, duration, piece in zip(sequence, durations[: len(sequence)], pieces, strict=True):
        if token_id != blank_token_id:
            tokens.append(_TimedToken(token_id, piece, frame))
        frame += duration
    return tokens


def split_sentences(tokens: list[_TimedToken]) -> list[list[_TimedToken]]:
    """Group tokens into sentences ending at terminal punctuation.

    A sentence closes only when the punctuation token is followed by a new word
    (SentencePiece ``▁`` prefix) or the end of output, so "3.5" is not split.
    """
    sentences: list[list[_TimedToken]] = []
    current: list[_TimedToken] = []
    for idx, token in enumerate(tokens):
        current.append(token)
        if not token.piece.endswith(_SENTENCE_END_CHARS):
            continue
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if nxt is None or nxt.piece.startswith("▁"):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


class NemotronASRBackend(TranscriberBase):
    """Transcriber backend using NVIDIA Nemotron ASR via Transformers."""

    def __init__(
        self,
        language: str | None,
        model: str = NEMOTRON_ASR_DEFAULT_MODEL,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        device: str | None = None,
        sub_timestamps: bool = True,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self.model = model
        self._device = _detect_device(device)
        self._sub_timestamps = sub_timestamps
        self._processor: Any = None
        self._model: Any = None
        self._language_key: str | None = None

        if num_threads is not None:
            torch.set_num_threads(num_threads)

        logger.info("Loading %s model via Transformers on %s...", self.model, self._device)
        self._load_model()

    @property
    def hard_limit_seconds(self) -> int:
        return NEMOTRON_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return NEMOTRON_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self) -> None:
        """Load Nemotron ASR via HuggingFace Transformers."""
        try:
            from transformers import AutoModelForRNNT, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers with Nemotron ASR support is not installed. "
                "Install with: uv add 'transformers>=5.13.0'"
            )

        self._processor = AutoProcessor.from_pretrained(self.model)
        self._model = AutoModelForRNNT.from_pretrained(
            self.model,
            dtype=_dtype_for_device(self._device),
        )
        self._model.to(self._device)
        self._model.eval()
        logger.info("Nemotron ASR model loaded: %s on %s", self.model, self._device)
        # Fail on an invalid language now, before any audio is streamed.
        self._get_language_key()

    def _get_language_key(self) -> str:
        if self._language_key is None:
            self._language_key = resolve_language(self.language, self._processor.prompt_dictionary)
        return self._language_key

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        start_offset: float = 0.0,
    ) -> list[TranscribedSegment]:
        """Transcribe one VAD segment; split into sentences when sub_timestamps is on."""
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(
            audio.copy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            language=self._get_language_key(),
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)

        with torch.inference_mode():
            output = self._model.generate(**inputs, return_dict_in_generate=True)

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        tokenizer = self._processor.tokenizer
        if not self._sub_timestamps:
            text = self._processor.decode(output.sequences, skip_special_tokens=True)[0].strip()
            if not text:
                return []
            return [self._make_segment(text, start_offset, end_time)]

        sequence: list[int] = output.sequences[0].tolist()
        durations: list[int] = output.durations[0].tolist()
        tokens = timed_tokens(
            sequence,
            durations,
            self._processor.blank_token_id,
            tokenizer.convert_ids_to_tokens(sequence),
        )

        segments: list[TranscribedSegment] = []
        sentences = split_sentences(tokens)
        for idx, sentence in enumerate(sentences):
            text = tokenizer.decode([t.token_id for t in sentence], skip_special_tokens=True).strip()
            if not text:
                continue
            start = start_offset if idx == 0 else start_offset + sentence[0].frame * NEMOTRON_FRAME_SECONDS
            if idx + 1 < len(sentences):
                end = start_offset + sentences[idx + 1][0].frame * NEMOTRON_FRAME_SECONDS
            else:
                end = end_time
            segments.append(self._make_segment(text, min(start, end_time), min(end, end_time)))
        return segments
