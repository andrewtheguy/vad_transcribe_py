"""NVIDIA Nemotron ASR (cache-aware FastConformer RNNT) backend via HuggingFace Transformers.

Each VAD segment is decoded offline in Whisper-style seek windows: a window that
does not reach the end of the segment keeps only its finished sentences, and the
next window re-decodes from shortly before the unfinished one began, so long
segments are never cut mid-sentence. Being a non-autoregressive RNNT model, it does not hallucinate or
loop on long input. Token timing comes from the RNNT durations (each blank
advances one encoder frame).
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
_FRAME_SAMPLES = round(NEMOTRON_FRAME_SECONDS * TARGET_SAMPLE_RATE)
# Seek window length. The encoder caps offline input at 5000 frames (~400s);
# 300s keeps headroom and peaks at ~4.5GB on MPS in fp16.
NEMOTRON_ASR_WINDOW_SECONDS = 300.0
# The next window starts this far before the cut: a window that starts right at a
# word drops or garbles it, and token emission lags the audio by a few frames.
NEMOTRON_ASR_OVERLAP_SECONDS = 4.0
# Cuts stay this far from the window end: without right context the model adds
# spurious terminal punctuation there and its token timing drifts.
NEMOTRON_ASR_EDGE_MARGIN_SECONDS = 8.0
# How far a sentence boundary in the next window may sit from the cut to match it.
_MATCH_TOLERANCE_FRAMES = 6
# Preferred minimum pause at a cut (0.4s): token timing differs by a few frames
# between two decodes, so a cut inside a real pause is matched unambiguously.
_MIN_CUT_GAP_FRAMES = 5
_AUTO_LANGUAGE = "auto"
# ASCII marks close a sentence only before a new word ("3.5", "U.S." stay whole);
# other scripts' marks (CJK, Arabic, Urdu, Devanagari, Armenian) close unconditionally.
_ASCII_SENTENCE_END = (".", "?", "!")
_SCRIPT_SENTENCE_END = ("。", "？", "！", "؟", "۔", "।", "॥", "։")
_SENTENCE_END_CHARS = _ASCII_SENTENCE_END + _SCRIPT_SENTENCE_END


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
    skip_token_ids: set[int],
    pieces: list[str],
) -> list[_TimedToken]:
    """Drop blanks and special tokens; attach an encoder-frame index to the rest.

    ``durations`` can carry one extra trailing step beyond ``sequence``; it only
    affects frames after the last token, so it is ignored. A bare word-boundary
    piece (``▁``) is often emitted right after the previous sentence ends, so it
    takes the frame of the token it precedes; otherwise pauses before a sentence
    look like zero gaps.
    """
    tokens: list[_TimedToken] = []
    frame = 0
    for token_id, duration, piece in zip(sequence, durations[: len(sequence)], pieces, strict=True):
        if token_id not in skip_token_ids:
            tokens.append(_TimedToken(token_id, piece, frame))
        frame += duration
    for i in range(len(tokens) - 2, -1, -1):
        if tokens[i].piece == "▁":
            tokens[i].frame = tokens[i + 1].frame
    return tokens


def split_sentences(tokens: list[_TimedToken]) -> list[list[_TimedToken]]:
    """Group tokens into sentences ending at terminal punctuation.

    ASCII punctuation closes a sentence only when followed by a new word
    (SentencePiece ``▁`` prefix) or the end of output, so "3.5" is not split.
    Other scripts' terminal marks close it unless more terminal punctuation follows.
    """
    sentences: list[list[_TimedToken]] = []
    current: list[_TimedToken] = []
    for idx, token in enumerate(tokens):
        current.append(token)
        if not token.piece.endswith(_SENTENCE_END_CHARS):
            continue
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        script_mark = token.piece.endswith(_SCRIPT_SENTENCE_END)
        if (
            nxt is None
            or nxt.piece.startswith("▁")
            or (script_mark and not nxt.piece.startswith(_SENTENCE_END_CHARS))
        ):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _sentence_starts(tokens: list[_TimedToken]) -> list[int]:
    """Indices of the first token of every sentence after the first."""
    starts: list[int] = []
    idx = 0
    for sentence in split_sentences(tokens)[:-1]:
        idx += len(sentence)
        starts.append(idx)
    return starts


def _gap_middle(tokens: list[_TimedToken], idx: int, end_frame: int) -> int:
    """Frame halfway between tokens[idx - 1] and tokens[idx] (or end_frame)."""
    nxt = tokens[idx].frame if idx < len(tokens) else end_frame
    return (tokens[idx - 1].frame + nxt + 1) // 2


def choose_cut(tokens: list[_TimedToken], window_frames: int, margin_frames: int) -> tuple[int, int]:
    """Pick where to end a window that does not reach the end of the audio.

    Returns ``(tokens to keep, cut frame)``. Cuts fall between the window midpoint
    and ``margin_frames`` before its end, before a sentence start: the latest one
    after a pause of at least ``_MIN_CUT_GAP_FRAMES``, else the one after the
    longest pause. Earlier sentences are kept and the rest is re-decoded by the
    next window. Without a sentence start in range, cuts in the middle of the
    longest gap between tokens (a silent tail counts), or at the end of the range.
    """
    lo, hi = window_frames // 2, window_frames - margin_frames

    def gap(i: int) -> int:
        nxt = tokens[i].frame if i < len(tokens) else window_frames
        return nxt - tokens[i - 1].frame

    sentence_cuts = [i for i in _sentence_starts(tokens) if lo <= tokens[i].frame <= hi]
    if sentence_cuts:
        paused = [i for i in sentence_cuts if gap(i) >= _MIN_CUT_GAP_FRAMES]
        best = paused[-1] if paused else max(reversed(sentence_cuts), key=gap)
        # Clamped to the midpoint so the next window always advances by a quarter window.
        return best, max(lo, _gap_middle(tokens, best, window_frames))

    word_cuts = [
        i for i in range(1, len(tokens) + 1)
        if tokens[i - 1].frame <= hi and (i == len(tokens) or tokens[i].frame >= lo)
    ]
    if not word_cuts:
        return sum(1 for t in tokens if t.frame < hi), hi
    best = max(reversed(word_cuts), key=gap)
    return best, max(lo, _gap_middle(tokens, best, window_frames))


def _is_word_start(piece: str) -> bool:
    """SentencePiece word start; non-ASCII tokens (CJK characters) also count."""
    return piece.startswith("▁") or not piece[:1].isascii()


def resume_index(tokens: list[_TimedToken], cut: int) -> int:
    """First token of an overlapping window that was not kept from the previous one.

    ``tokens`` and ``cut`` share a frame origin. Matches the sentence boundary
    nearest the cut (within a small tolerance), so timing jitter between the two
    decodes cannot duplicate or drop a word. Otherwise picks whichever word start
    around the cut sits nearest it, so a word is never split.
    """
    best: int | None = None
    best_dist = _MATCH_TOLERANCE_FRAMES + 1
    for i in _sentence_starts(tokens):
        dist = abs(_gap_middle(tokens, i, cut) - cut)
        if dist < best_dist:
            best, best_dist = i, dist
    if best is not None:
        return best

    first = next((i for i, t in enumerate(tokens) if t.frame >= cut), len(tokens))
    if first == len(tokens):
        return first
    candidates = [next((i for i in range(first, len(tokens)) if _is_word_start(tokens[i].piece)), len(tokens))]
    before = next((i for i in range(first - 1, -1, -1) if _is_word_start(tokens[i].piece)), None)
    if before is not None:
        candidates.append(before)

    def position(i: int) -> int:
        if i == 0:
            return tokens[0].frame
        return _gap_middle(tokens, i, tokens[-1].frame + 1)

    return min(candidates, key=lambda i: abs(position(i) - cut))


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
        window_seconds: float = NEMOTRON_ASR_WINDOW_SECONDS,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self.model = model
        self._window_frames = max(4, int(window_seconds / NEMOTRON_FRAME_SECONDS))
        # Overlap and margin stay under a quarter window so every window makes progress.
        self._overlap_frames = min(
            int(NEMOTRON_ASR_OVERLAP_SECONDS / NEMOTRON_FRAME_SECONDS), self._window_frames // 4
        )
        self._margin_frames = min(
            int(NEMOTRON_ASR_EDGE_MARGIN_SECONDS / NEMOTRON_FRAME_SECONDS), self._window_frames // 4
        )
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

    def _decode(self, audio: npt.NDArray[np.float32]) -> list[_TimedToken]:
        """Decode one window (at most ~400s) into timed tokens."""
        inputs = self._processor(
            audio.copy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            language=self._get_language_key(),
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)

        with torch.inference_mode():
            output = self._model.generate(**inputs, return_dict_in_generate=True)

        tokenizer = self._processor.tokenizer
        sequence: list[int] = output.sequences[0].tolist()
        durations: list[int] = output.durations[0].tolist()
        # The model emits config.blank_token_id (== pad), which differs from the
        # processor's blank_token_id; drop all of them, plus special tokens such
        # as the language tags emitted at sentence starts.
        candidates = (
            self._processor.blank_token_id,
            getattr(getattr(self._model, "config", None), "blank_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            *getattr(tokenizer, "all_special_ids", ()),
        )
        skip_ids = {int(c) for c in candidates if c is not None}
        return timed_tokens(sequence, durations, skip_ids, tokenizer.convert_ids_to_tokens(sequence))

    def _decode_windows(self, audio: npt.NDArray[np.float32]) -> list[_TimedToken]:
        """Decode a segment of any length in overlapping seek windows.

        Returned frames are relative to the segment start.
        """
        window_samples = self._window_frames * _FRAME_SAMPLES
        tokens: list[_TimedToken] = []
        offset = 0  # window start, in frames
        cut: int | None = None  # end of the previous window's kept tokens, in frames
        while True:
            start = offset * _FRAME_SAMPLES
            window_tokens = self._decode(audio[start:start + window_samples])
            begin = 0 if cut is None else resume_index(window_tokens, cut - offset)
            last = start + window_samples >= len(audio)
            keep, window_cut = (
                (len(window_tokens), 0) if last else choose_cut(window_tokens, self._window_frames, self._margin_frames)
            )
            tokens.extend(
                _TimedToken(t.token_id, t.piece, t.frame + offset) for t in window_tokens[begin:keep]
            )
            if last:
                return tokens
            cut = offset + window_cut
            offset = max(offset + 1, cut - self._overlap_frames)
            logger.debug("Nemotron window cut at %.2fs", cut * NEMOTRON_FRAME_SECONDS)

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        start_offset: float = 0.0,
    ) -> list[TranscribedSegment]:
        """Transcribe one VAD segment; split into sentences when sub_timestamps is on."""
        assert self._processor is not None
        assert self._model is not None

        tokens = self._decode_windows(audio)
        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        tokenizer = self._processor.tokenizer
        if not self._sub_timestamps:
            text = tokenizer.decode([t.token_id for t in tokens], skip_special_tokens=True).strip()
            if not text:
                return []
            return [self._make_segment(text, start_offset, end_time)]

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
