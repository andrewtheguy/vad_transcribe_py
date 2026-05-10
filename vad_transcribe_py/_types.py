"""Shared types for transcriber backends."""

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from vad_transcribe_py._utils import (
    ChineseConversion,
    format_timestamp,
    process_text,
)

logger = logging.getLogger(__name__)


@dataclass
class TranscribedSegment:
    """Represents a transcribed audio segment."""
    text: str
    start: float
    end: float
    debug: dict[str, object] = field(default_factory=dict)


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

    def _make_segment(
        self,
        text: str,
        start: float,
        end: float,
        debug: dict[str, object] | None = None,
    ) -> TranscribedSegment:
        """Format timestamps, print to stderr, process text, and return a TranscribedSegment."""
        start_fmt = format_timestamp(start)
        end_fmt = format_timestamp(end)
        logger.info("[%s -> %s] %s", start_fmt, end_fmt, text)
        text = process_text(text, self.chinese_conversion)
        return TranscribedSegment(text=text, start=start, end=end, debug=debug or {})
