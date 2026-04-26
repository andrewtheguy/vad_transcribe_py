"""GLM-ASR backend using the mlx-audio package (Apple Silicon / Metal)."""

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
from vad_transcribe_py.vad_processor import (
    GLM_ASR_HARD_LIMIT_SECONDS,
    GLM_ASR_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

GLM_ASR_MLX_DEFAULT_MODEL = "mlx-community/GLM-ASR-Nano-2512-8bit"
GLM_ASR_MLX_MAX_TOKENS = 500

# mlx-audio's GLM-ASR auto-detects language and only ships English + Chinese
# weights, so we don't pass --language down to the model. Warn if the user
# supplies something the model can't handle.
_SUPPORTED_LANGUAGES = {"zh", "en", "yue"}


class GLMASRMLXBackend(TranscriberBase):
    """Transcriber backend using GLM-ASR via mlx-audio (Apple Silicon / Metal)."""

    def __init__(
        self,
        language: str | None,
        model: str = GLM_ASR_MLX_DEFAULT_MODEL,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        device: str | None = None,
    ):
        super().__init__(language, chinese_conversion, num_threads)
        self.model = model
        self._mlx_model: Any = None

        if device is not None and device != "metal":
            logger.warning("device=%r ignored; mlx-audio backend always uses Metal", device)
        if num_threads is not None:
            logger.warning("num_threads=%d ignored; mlx-audio has no threading knob", num_threads)
        if language is not None and language not in _SUPPORTED_LANGUAGES:
            logger.warning(
                "language=%r is outside GLM-ASR's English/Chinese support; "
                "the model will still auto-detect, but accuracy may suffer",
                language,
            )

        logger.info("Loading %s model via mlx-audio...", self.model)
        self._load_model()

    @property
    def hard_limit_seconds(self) -> int:
        return GLM_ASR_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return GLM_ASR_SOFT_LIMIT_SECONDS

    def _load_model(self) -> None:
        """Load GLM-ASR via mlx-audio. First run downloads the model from HuggingFace."""
        try:
            from mlx_audio.stt.utils import load_model  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "mlx-audio is not installed (Apple Silicon only). "
                "Install with: uv add mlx-audio"
            )

        self._mlx_model = load_model(self.model)
        logger.info("mlx-audio GLM-ASR model loaded: %s", self.model)

    def transcribe(
        self, audio: npt.NDArray[np.float32], start_offset: float = 0.0,
    ) -> list[TranscribedSegment]:
        """Transcribe a single audio segment and return one TranscribedSegment."""
        assert self._mlx_model is not None

        output = self._mlx_model.generate(
            audio,
            max_tokens=GLM_ASR_MLX_MAX_TOKENS,
            verbose=False,
        )
        text: str = output.text

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time)]
