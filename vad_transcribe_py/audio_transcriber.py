import struct
import subprocess
import sys
from collections.abc import Generator
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

from vad_transcribe_py._types import (
    TARGET_SAMPLE_RATE as TARGET_SAMPLE_RATE,
    AudioTranscriber as AudioTranscriber,
    ChineseConversion as ChineseConversion,
    TranscribedSegment as TranscribedSegment,
    TranscriberBase as TranscriberBase,
    format_timestamp as format_timestamp,
    process_text as process_text,
)


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


def create_transcriber(
    language: str,
    model: str | None = None,
    backend: str = 'whisper',
    chinese_conversion: ChineseConversion = 'none',
) -> AudioTranscriber:
    """Factory function to create a transcriber backend instance."""
    if backend == 'whisper':
        from vad_transcribe_py.backends.whisper import WHISPER_DEFAULT_MODEL, WhisperBackend

        return WhisperBackend(
            language=language,
            model=model if model is not None else WHISPER_DEFAULT_MODEL,
            chinese_conversion=chinese_conversion,
        )
    elif backend == 'moonshine':
        from vad_transcribe_py.backends.moonshine import MoonshineBackend

        return MoonshineBackend(
            language=language,
            model=model,
            chinese_conversion=chinese_conversion,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
