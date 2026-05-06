"""NVIDIA Whisper Large V3 backend via Riva gRPC (build.nvidia.com)."""

import io
import logging
import os
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from vad_transcribe_py._types import (
    TranscribedSegment,
    TranscriberBase,
)
from vad_transcribe_py._utils import (
    TARGET_SAMPLE_RATE,
    ChineseConversion,
)
from vad_transcribe_py.vad_processor import (
    NVIDIA_WHISPER_HARD_LIMIT_SECONDS,
    NVIDIA_WHISPER_SOFT_LIMIT_SECONDS,
)

logger = logging.getLogger(__name__)

NVIDIA_WHISPER_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"
NVIDIA_RIVA_URI = "grpc.nvcf.nvidia.com:443"


class NvidiaWhisperBackend(TranscriberBase):
    """Transcriber backend using NVIDIA-hosted whisper-large-v3 via Riva gRPC."""

    def __init__(
        self,
        language: str | None,
        model: str | None = None,
        chinese_conversion: ChineseConversion = 'none',
        num_threads: int | None = None,
        device: str | None = None,
    ):
        super().__init__(language, chinese_conversion, num_threads)

        if model is not None:
            logger.warning("model=%r ignored; nvidia-whisper is fixed to whisper-large-v3", model)
        if num_threads is not None:
            logger.warning("num_threads=%d ignored; nvidia-whisper runs server-side", num_threads)
        if device is not None:
            logger.warning("device=%r ignored; nvidia-whisper runs server-side", device)

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Add it to .env "
                "(get a key from https://build.nvidia.com/openai/whisper-large-v3)."
            )

        self._service: Any = None
        self._recognition_config_cls: Any = None
        self._init_service(api_key)

    @property
    def hard_limit_seconds(self) -> int:
        return NVIDIA_WHISPER_HARD_LIMIT_SECONDS

    @property
    def soft_limit_seconds(self) -> float | None:
        return NVIDIA_WHISPER_SOFT_LIMIT_SECONDS

    def _init_service(self, api_key: str) -> None:
        """Build the Riva ASR service with NVCF function-id + bearer auth metadata."""
        try:
            import riva.client  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError(
                "nvidia-riva-client is not installed. "
                "Install with: uv sync --extra nvidia-whisper"
            )

        auth = riva.client.Auth(
            uri=NVIDIA_RIVA_URI,
            use_ssl=True,
            metadata_args=[
                ["function-id", NVIDIA_WHISPER_FUNCTION_ID],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        self._service = riva.client.ASRService(auth)
        self._recognition_config_cls = riva.client.RecognitionConfig
        logger.info("NVIDIA whisper-large-v3 backend ready (Riva gRPC)")

    def transcribe(
        self, audio: npt.NDArray[np.float32], start_offset: float = 0.0,
    ) -> list[TranscribedSegment]:
        """Encode audio as 16-bit PCM WAV and POST it to the Riva offline endpoint."""
        assert self._service is not None
        assert self._recognition_config_cls is not None

        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, TARGET_SAMPLE_RATE, pcm16)
        wav_bytes = buf.getvalue()

        config = self._recognition_config_cls(
            language_code=self.language or "multi",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=False,
        )

        response = self._service.offline_recognize(wav_bytes, config)

        text_parts: list[str] = []
        for result in response.results:
            if result.alternatives:
                text_parts.append(result.alternatives[0].transcript)
        text = "".join(text_parts).strip()

        end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
        return [self._make_segment(text, start_offset, end_time)]
