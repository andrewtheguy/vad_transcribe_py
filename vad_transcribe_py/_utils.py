"""Shared constants and utility functions for transcriber backends."""

from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Literal

from zhconv_rs import zhconv

TARGET_SAMPLE_RATE = 16000
ChineseConversion = Literal['none', 'simplified', 'traditional']

_REPETITION_THRESHOLD = 0.3
_NEAR_DUPLICATE_THRESHOLD = 0.9

INDISTINGUISHABLE_PLACEHOLDER = "(indistinguishable speech)"
_CLIP_COVERAGE_THRESHOLD = 0.6
_CLIP_MAX_NGRAM = 4
_CLIP_DEFAULT_MIN_TOTAL_TOKENS = 8


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


def _has_dominant_repetition(tokens: Sequence[str], min_repetitions: int, min_total_tokens: int) -> bool:
    """Return True if some n-gram (n in 1.._CLIP_MAX_NGRAM) appears at least
    ``min_repetitions`` times AND its occurrences cover ≥ 60% of ``tokens``.

    Returns False unconditionally when there are fewer than ``min_total_tokens``
    tokens — short utterances like "yes yes yes" or "好好好" are plausible real
    speech and shouldn't be clipped even if technically repetitive.
    """
    n = len(tokens)
    if n < min_total_tokens or n < min_repetitions:
        return False
    max_ngram = min(_CLIP_MAX_NGRAM, n // min_repetitions)
    for ngram_size in range(1, max_ngram + 1):
        counter: Counter[tuple[str, ...]] = Counter(
            tuple(tokens[i:i + ngram_size]) for i in range(n - ngram_size + 1)
        )
        _, count = counter.most_common(1)[0]
        if count >= min_repetitions and (count * ngram_size) / n >= _CLIP_COVERAGE_THRESHOLD:
            return True
    return False


def clip_repetitive_text(
    text: str,
    *,
    min_repetitions: int = 3,
    min_total_tokens: int = _CLIP_DEFAULT_MIN_TOTAL_TOKENS,
) -> str:
    """Replace heavily repetitive transcripts with a placeholder, else return as-is.

    Catches model degenerate outputs like ``"dungu dungu dungu …"`` (word-level)
    or ``"好好好好"`` / ``"૨૨૨"`` (character-level) by checking n-gram coverage.
    A pattern is "heavy" when an n-gram of size 1..4 repeats ``min_repetitions``+
    times and accounts for ≥ 60% of the tokens.

    Lines below ``min_total_tokens`` (default 8) are passed through unchanged
    even if repetitive — short utterances are likely real speech, not model
    hallucination.
    """
    if not text:
        return text
    words = text.lower().split()
    if _has_dominant_repetition(words, min_repetitions, min_total_tokens):
        return INDISTINGUISHABLE_PLACEHOLDER
    chars = [c for c in text if not c.isspace() and not c.isascii()]
    if _has_dominant_repetition(chars, min_repetitions, min_total_tokens):
        return INDISTINGUISHABLE_PLACEHOLDER
    return text


def process_text(text: str, chinese_conversion: ChineseConversion) -> str:
    """Process text for storage (e.g., convert Chinese variants)."""
    if chinese_conversion != 'none':
        if chinese_conversion == 'traditional':
            return zhconv(text, 'zh-Hant')
        elif chinese_conversion == 'simplified':
            return zhconv(text, 'zh-Hans')
    return text
