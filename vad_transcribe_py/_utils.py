"""Shared constants and utility functions for transcriber backends."""

from difflib import SequenceMatcher
from typing import Literal

from zhconv_rs import zhconv

TARGET_SAMPLE_RATE = 16000
ChineseConversion = Literal['none', 'simplified', 'traditional']

_REPETITION_THRESHOLD = 0.3
_NEAR_DUPLICATE_THRESHOLD = 0.9

INDISTINGUISHABLE_PLACEHOLDER = "(indistinguishable speech)"
_CLIP_MIN_TEXT_LEN = 100
_CLIP_MIN_PATTERN_LEN = 2


def format_timestamp(seconds: float) -> str:
    """Format seconds to hh:mm:ss.ms format."""
    if seconds < 0:
        raise ValueError(f"format_timestamp requires non-negative seconds, got {seconds}")
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def is_repetitive(text: str) -> bool:
    """Detect repetitive text that would degrade conditioning quality."""
    if not text:
        return False

    # Word-level repetition (covers space-separated languages)
    words = text.lower().split()
    if len(words) >= 4 and len(set(words)) / len(words) < _REPETITION_THRESHOLD:
        return True

    # CJK character-level repetition
    cjk_chars = [c for c in text if '一' <= c <= '鿿' or '㐀' <= c <= '䶿']
    if len(cjk_chars) >= 4 and len(set(cjk_chars)) / len(cjk_chars) < _REPETITION_THRESHOLD:
        return True

    return False


def is_near_duplicate(current: str, prior: str) -> bool:
    """Detect when *current* is almost exactly the same as *prior*.

    Used to avoid feeding back near-identical lines as conditioning context,
    which tends to push the model into a repeat-the-prompt loop.
    """
    if not current or not prior:
        return False
    a = current.strip().lower()
    b = prior.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= _NEAR_DUPLICATE_THRESHOLD


def conditioning_context(current: str, prior_line: str) -> str:
    """Return *current* as conditioning context for the next segment, or "" if unsafe.

    Filters out empty, internally-repetitive, and near-duplicate-of-prior text —
    all cases where feeding the line back as a prompt tends to nudge the model
    into a repeat-the-prompt loop.
    """
    stripped = current.strip()
    if not stripped or is_repetitive(stripped) or is_near_duplicate(stripped, prior_line):
        return ""
    return stripped


def clip_repetitive_text(
    text: str,
    *,
    min_repeats: int = 10,
    max_pattern_len: int = 30,
) -> str:
    """Truncate consecutively repeating patterns caused by ASR hallucination."""
    return clip_repetitive_text_with_patterns(
        text,
        min_repeats=min_repeats,
        max_pattern_len=max_pattern_len,
    )[0]


def clip_repetitive_text_with_patterns(
    text: str,
    *,
    min_repeats: int = 10,
    max_pattern_len: int = 30,
) -> tuple[str, list[str]]:
    """Truncate consecutively repeating patterns caused by ASR hallucination.

    Scans for a char-level periodic run: for each feasible candidate
    ``pat_len`` in ``2..max_pattern_len``, walks the string looking for
    ``text[i] == text[i - pat_len]`` over a continuous run of at least
    ``pat_len * (min_repeats - 1)`` chars. When found, returns the prefix up to
    and including the first copy of the pattern, followed by
    ``(indistinguishable speech)``.

    With the default ``max_pattern_len=30``, this is linear in the text length
    with a small constant factor and no substring allocations before a match.
    Candidate scans stop as soon as the remaining suffix cannot satisfy
    ``min_repeats``.

    Short inputs (< ``_CLIP_MIN_TEXT_LEN`` chars) are passed through unchanged —
    short utterances like "yes yes yes" or "好好好" are plausible real speech,
    not model hallucination.

    Returns ``(text, patterns)`` where ``patterns`` is empty when no clipping
    happened. The current algorithm clips the first repetitive run, so
    ``patterns`` contains at most one string.
    """
    if min_repeats < 2:
        raise ValueError(f"min_repeats must be at least 2, got {min_repeats}")

    n = len(text)
    if n < _CLIP_MIN_TEXT_LEN:
        return text, []

    max_feasible_pattern_len = min(max_pattern_len, n // min_repeats)
    if max_feasible_pattern_len < _CLIP_MIN_PATTERN_LEN:
        return text, []

    for pat_len in range(_CLIP_MIN_PATTERN_LEN, max_feasible_pattern_len + 1):
        run = 0
        threshold = pat_len * (min_repeats - 1)
        for i in range(pat_len, n):
            if n - i < threshold - run:
                break
            if text[i] == text[i - pat_len]:
                run += 1
                if run >= threshold:
                    start = i - run - pat_len + 1
                    pattern = text[start : start + pat_len]
                    return text[: start + pat_len] + INDISTINGUISHABLE_PLACEHOLDER, [pattern]
            else:
                run = 0
    return text, []


def process_text(text: str, chinese_conversion: ChineseConversion) -> str:
    """Process text for storage (e.g., convert Chinese variants)."""
    if chinese_conversion != 'none':
        if chinese_conversion == 'traditional':
            return zhconv(text, 'zh-Hant')
        elif chinese_conversion == 'simplified':
            return zhconv(text, 'zh-Hans')
    return text
