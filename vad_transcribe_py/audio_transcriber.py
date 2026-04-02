import struct
import subprocess
import sys
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, BinaryIO, Literal

import numpy as np
import numpy.typing as npt
import torch

from zhconv_rs import zhconv

from vad_transcribe_py.vad_processor import (
    WHISPER_HARD_LIMIT_SECONDS,
    WHISPER_SOFT_LIMIT_SECONDS,
)

TARGET_SAMPLE_RATE = 16000
ChineseConversion = Literal['none', 'simplified', 'traditional']


def format_timestamp(seconds: float) -> str:
    """Format seconds to hh:mm:ss.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _validate_wav_header(stream: BinaryIO) -> tuple[int, int, int, int, int]:
    """Read and validate a WAV header from a binary stream.

    Accepts mono WAV at target sample rate in 16-bit PCM, 32-bit PCM,
    or 32-bit IEEE float. Returns (audio_format, bits_per_sample, sample_rate,
    channels, data_size). Raises ValueError on invalid format.
    """
    riff = stream.read(4)
    if riff != b'RIFF':
        raise ValueError(f"Not a WAV file: expected RIFF, got {riff!r}")

    stream.read(4)  # file size (ignore)

    wave = stream.read(4)
    if wave != b'WAVE':
        raise ValueError(f"Not a WAV file: expected WAVE, got {wave!r}")

    # Find fmt chunk
    while True:
        chunk_id = stream.read(4)
        if len(chunk_id) < 4:
            raise ValueError("WAV file missing fmt chunk")
        chunk_size = struct.unpack('<I', stream.read(4))[0]

        if chunk_id == b'fmt ':
            break
        skipped = stream.read(chunk_size)
        if len(skipped) != chunk_size:
            raise ValueError("WAV file truncated while skipping chunk")

    fmt_data = stream.read(chunk_size)
    if len(fmt_data) < 16:
        raise ValueError("WAV fmt chunk too short")

    audio_format, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
        '<HHIIHH', fmt_data[:16]
    )

    if audio_format == 1:  # PCM integer
        if bits_per_sample not in (16, 32):
            raise ValueError(f"Expected 16-bit or 32-bit PCM, got {bits_per_sample}")
    elif audio_format == 3:  # IEEE float
        if bits_per_sample != 32:
            raise ValueError(f"Expected 32-bit float, got {bits_per_sample}")
    else:
        raise ValueError(f"Expected PCM (1) or IEEE float (3) format, got {audio_format}")

    if channels != 1:
        raise ValueError(f"Expected mono (1 channel), got {channels}")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate}")

    # Find data chunk
    data_size = 0
    while True:
        chunk_id = stream.read(4)
        if len(chunk_id) < 4:
            raise ValueError("WAV file missing data chunk")
        chunk_size_bytes = stream.read(4)
        if len(chunk_size_bytes) < 4:
            raise ValueError("WAV file truncated")

        if chunk_id == b'data':
            data_size = struct.unpack('<I', chunk_size_bytes)[0]
            break
        chunk_size = struct.unpack('<I', chunk_size_bytes)[0]
        skipped = stream.read(chunk_size)
        if len(skipped) != chunk_size:
            raise ValueError("WAV file truncated while skipping chunk")

    return audio_format, bits_per_sample, sample_rate, channels, data_size


def _stream_wav_as_float32(stream: BinaryIO, audio_format: int, bits_per_sample: int, chunk_bytes: int = 4096) -> Generator[npt.NDArray[np.float32], None, None]:
    """Stream WAV data as float32 arrays, converting from PCM if needed."""
    if audio_format == 3:
        dtype = np.float32
    elif bits_per_sample == 16:
        dtype = np.int16
    else:
        dtype = np.int32

    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            break
        raw = np.frombuffer(chunk, dtype=dtype)
        if dtype == np.int16:
            yield (raw.astype(np.float32) / np.float32(32768.0))
        elif dtype == np.int32:
            yield (raw.astype(np.float32) / np.float32(2147483648.0))
        else:
            yield raw.astype(np.float32)



def stream_stdin_wav(
    chunk_bytes: int = 4096,
) -> Generator[npt.NDArray[np.float32], None, None]:
    """Read WAV from stdin and yield float32 audio chunks.

    Validates the WAV header, then streams PCM data as float32 arrays.
    Accepts 16-bit PCM, 32-bit PCM, or 32-bit float WAV (mono, 16kHz).
    """
    audio_format, bits_per_sample, *_ = _validate_wav_header(sys.stdin.buffer)
    print("Reading WAV from stdin", file=sys.stderr)
    yield from _stream_wav_as_float32(sys.stdin.buffer, audio_format, bits_per_sample, chunk_bytes)


def ffmpeg_stream_float32(
    full_audio_path: str,
    target_sample_rate: int | None = None,
    ac: int | None = None,
    chunk_bytes: int = 4096,
) -> Generator[npt.NDArray[np.float32], None, None]:
    """Stream audio as float32 arrays from a file, using direct WAV read or ffmpeg.

    Args:
        full_audio_path: Path to audio file or URL
        target_sample_rate: Target sample rate for output
        ac: Number of audio channels
        chunk_bytes: Bytes to read per chunk (default 4096 = 1024 float32 samples)

    Yields:
        Float32 numpy arrays of audio samples
    """
    # Try direct WAV reading for files (not URLs)
    if not full_audio_path.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
        try:
            with open(full_audio_path, 'rb') as f:
                audio_format, bits_per_sample, *_ = _validate_wav_header(f)
                print(f"Direct WAV read: {full_audio_path}", file=sys.stderr)
                yield from _stream_wav_as_float32(f, audio_format, bits_per_sample, chunk_bytes)
                return
        except (OSError, ValueError):
            pass

    # Fall back to ffmpeg
    command = [
        "ffmpeg",
        "-i", full_audio_path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
    ]

    if ac is not None:
        command.extend(["-ac", str(ac)])

    if target_sample_rate is not None:
        command.extend(["-ar", str(target_sample_rate)])

    command.extend([
        "-loglevel", "error",
        "pipe:",
    ])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
    )

    assert process.stdout is not None
    try:
        while True:
            chunk = process.stdout.read(chunk_bytes)
            if not chunk:
                break
            yield np.frombuffer(chunk, dtype=np.float32)

        returncode = process.wait()
        if returncode != 0:
            raise ValueError(f"ffmpeg command failed with return code {returncode}")
    finally:
        process.stdout.close()


def get_window_size_samples() -> int:
    """Get window size for VAD processing."""
    return 512 if TARGET_SAMPLE_RATE == 16000 else 256


@dataclass
class TranscribedSegment:
    """Represents a transcribed audio segment."""
    text: str
    start: float
    end: float


def _get_device_and_dtype():
    """Auto-detect best device and dtype."""
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def _resolve_whisper_model_id(model: str) -> str:
    """Resolve short whisper model name to full HuggingFace model ID."""
    if '/' in model:
        return model
    return f"openai/whisper-{model}"


WHISPER_DEFAULT_MODEL = "large-v3-turbo"


def create_transcriber(
    language: str,
    model: str | None = None,
    backend: Literal['whisper', 'moonshine'] = 'whisper',
    chinese_conversion: ChineseConversion = 'none',
) -> 'WhisperTranscriber':
    """Factory function to create a WhisperTranscriber instance."""
    return WhisperTranscriber(
        language=language,
        model=model,
        backend=backend,
        chinese_conversion=chinese_conversion,
    )


class WhisperTranscriber:
    """Transcriber supporting Whisper (HF Transformers) and Moonshine (ONNX) backends."""

    def __init__(
        self,
        language: str,
        model: str | None = None,
        backend: Literal['whisper', 'moonshine'] = 'whisper',
        chinese_conversion: ChineseConversion = 'none',
    ):
        self.language = language
        self.model = model
        self.backend = backend
        self.chinese_conversion = chinese_conversion
        self._hard_limit_seconds = WHISPER_HARD_LIMIT_SECONDS
        self._soft_limit_seconds: float | None = WHISPER_SOFT_LIMIT_SECONDS
        self.pipe: Any = None
        self._moonshine_transcriber: Any = None

        # Load backend-specific model
        if self.backend == 'whisper':
            if self.model is None:
                self.model = WHISPER_DEFAULT_MODEL
            print(f"Loading {self.model} model...", file=sys.stderr)
            self._load_whisper()
        elif self.backend == 'moonshine':
            self._load_moonshine()  # handles model=None via resolve_model
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @property
    def hard_limit_seconds(self) -> int:
        return self._hard_limit_seconds

    @property
    def soft_limit_seconds(self) -> float | None:
        return self._soft_limit_seconds

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

        assert self.model is not None
        model_id = _resolve_whisper_model_id(self.model)
        device, torch_dtype = _get_device_and_dtype()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch_dtype,
            device=device,
        )

        print(f"Whisper model loaded: {model_id} on {device}", file=sys.stderr)

    def _load_moonshine(self) -> None:
        """Load Moonshine model via ONNX runtime."""
        from vad_transcribe_py.moonshine import resolve_model, download_model, Transcriber, SAMPLE_RATE

        # resolve_model handles default model selection per language when model is None
        name, language, arch, is_streaming, url, hard_limit, soft_limit = resolve_model(
            self.language, self.model
        )
        self.model = name
        self._hard_limit_seconds = hard_limit
        self._soft_limit_seconds = soft_limit

        print(f"Loading {name} model...", file=sys.stderr)
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
        )

        print(f"Moonshine model loaded: {name}", file=sys.stderr)

    def transcribe(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> list[TranscribedSegment]:
        """
        Transcribe audio and return segments.

        Args:
            audio: Float32 numpy array normalized to [-1.0, 1.0]
            start_offset: Start time offset for the audio segment

        Returns:
            List of TranscribedSegment objects
        """
        results: list[TranscribedSegment] = []

        if self.backend == 'whisper':
            result = self.pipe(
                audio.copy(),
                return_timestamps=True,
                generate_kwargs={"language": self.language},
            )

            for chunk in result["chunks"]:
                chunk_start = start_offset + chunk["timestamp"][0]
                chunk_end = start_offset + (chunk["timestamp"][1] if chunk["timestamp"][1] is not None else len(audio) / TARGET_SAMPLE_RATE)
                start_fmt = format_timestamp(chunk_start)
                end_fmt = format_timestamp(chunk_end)
                print("[%s -> %s] %s" % (start_fmt, end_fmt, chunk["text"]), file=sys.stderr)
                text = self._process_text(chunk["text"])
                results.append(TranscribedSegment(
                    text=text,
                    start=chunk_start,
                    end=chunk_end,
                ))

        elif self.backend == 'moonshine':
            text = self._moonshine_transcriber.transcribe_chunk(audio)

            end_time = start_offset + len(audio) / TARGET_SAMPLE_RATE
            start_fmt = format_timestamp(start_offset)
            end_fmt = format_timestamp(end_time)
            print("[%s -> %s] %s" % (start_fmt, end_fmt, text), file=sys.stderr)
            text = self._process_text(text)
            results.append(TranscribedSegment(
                text=text,
                start=start_offset,
                end=end_time,
            ))

        return results

    def _process_text(self, text: str) -> str:
        """Process text for storage (e.g., convert Chinese variants)."""
        if self.language in ['yue', 'zh'] and self.chinese_conversion != 'none':
            if self.chinese_conversion == 'traditional':
                return zhconv(text, 'zh-Hant')
            elif self.chinese_conversion == 'simplified':
                return zhconv(text, 'zh-Hans')
        return text
