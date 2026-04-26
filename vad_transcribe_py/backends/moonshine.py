"""Moonshine backend using ONNX Runtime."""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from vad_transcribe_py._types import (
    TranscribedSegment,
    TranscriberBase,
)
from vad_transcribe_py._utils import (
    TARGET_SAMPLE_RATE,
    ChineseConversion,
)


logger = logging.getLogger(__name__)


class MoonshineBackend(TranscriberBase):
    """Transcriber backend using Moonshine via ONNX Runtime."""

    def __init__(
        self,
        language: str,
        model: str | None = None,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self._moonshine_transcriber: Any = None

        self._hard_limit_seconds: int = 0
        self._soft_limit_seconds: float | None = None

        self._load_moonshine(model, num_threads)

    @property
    def hard_limit_seconds(self) -> int:
        return self._hard_limit_seconds

    @property
    def soft_limit_seconds(self) -> float | None:
        return self._soft_limit_seconds

    def _load_moonshine(self, model: str | None, num_threads: int | None) -> None:
        """Load Moonshine model via ONNX runtime."""
        from vad_transcribe_py.moonshine import resolve_model, download_model, Transcriber, SAMPLE_RATE

        assert self.language is not None
        name, language, arch, is_streaming, url, hard_limit, soft_limit = resolve_model(
            self.language, model
        )
        self.model = name
        self._hard_limit_seconds = hard_limit
        self._soft_limit_seconds = soft_limit

        logger.info("Loading %s model...", name)
        model_dir = download_model(language, arch, url)

        # Max tokens = audio_samples * token_limit_factor. Streaming models produce
        # ~6.5 tokens/sec, non-streaming ~13 tokens/sec. Dividing by SAMPLE_RATE
        # converts from per-second to per-sample. (Source: moonshine-ai/moonshine)
        token_limit_factor = 6.5 / SAMPLE_RATE if is_streaming else 13 / SAMPLE_RATE
        strip_cjk_spaces = self.language in ('zh', 'ja', 'ko')

        self._moonshine_transcriber = Transcriber(
            model_dir=model_dir,
            model_arch=arch,
            is_streaming=is_streaming,
            strip_cjk_spaces=strip_cjk_spaces,
            token_limit_factor=token_limit_factor,
            num_threads=num_threads,
        )

        logger.info("Moonshine model loaded: %s", name)

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """Transcribe audio and return a single segment."""
        text = self._moonshine_transcriber.transcribe_chunk(audio)
        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time)]
