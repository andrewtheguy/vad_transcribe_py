"""Shared types and utilities for transcriber backends."""

import logging
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from zhconv_rs import zhconv

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
ChineseConversion = Literal['none', 'simplified', 'traditional']


def format_timestamp(seconds: float) -> str:
    """Format seconds to hh:mm:ss.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


@dataclass
class TranscribedSegment:
    """Represents a transcribed audio segment."""
    text: str
    start: float
    end: float


_REPETITION_THRESHOLD = 0.3


def is_repetitive(text: str) -> bool:
    """Detect repetitive text that would degrade conditioning quality."""
    if not text:
        return False

    # Word-level repetition (covers space-separated languages)
    words = text.lower().split()
    if len(words) >= 4 and len(set(words)) / len(words) < _REPETITION_THRESHOLD:
        return True

    # CJK character-level repetition
    cjk_chars = [c for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf']
    if len(cjk_chars) >= 4 and len(set(cjk_chars)) / len(cjk_chars) < _REPETITION_THRESHOLD:
        return True

    return False


def process_text(text: str, language: str | None, chinese_conversion: ChineseConversion) -> str:
    """Process text for storage (e.g., convert Chinese variants)."""
    if language in ['yue', 'zh'] and chinese_conversion != 'none':
        if chinese_conversion == 'traditional':
            return zhconv(text, 'zh-Hant')
        elif chinese_conversion == 'simplified':
            return zhconv(text, 'zh-Hans')
    return text


@runtime_checkable
class AudioTranscriber(Protocol):
    """Protocol that all transcriber backends must satisfy."""

    @property
    def hard_limit_seconds(self) -> int: ...

    @property
    def soft_limit_seconds(self) -> float | None: ...

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]: ...


class TranscriberBase:
    """Shared functionality for transcriber backends."""

    def __init__(self, language: str | None, chinese_conversion: ChineseConversion = 'none', num_threads: int | None = None):
        self.language = language
        self.chinese_conversion: ChineseConversion = chinese_conversion
        self.num_threads = num_threads

    def _make_segment(self, text: str, start: float, end: float) -> TranscribedSegment:
        """Format timestamps, print to stderr, process text, and return a TranscribedSegment."""
        start_fmt = format_timestamp(start)
        end_fmt = format_timestamp(end)
        logger.info("[%s -> %s] %s", start_fmt, end_fmt, text)
        text = process_text(text, self.language, self.chinese_conversion)
        return TranscribedSegment(text=text, start=start, end=end)
