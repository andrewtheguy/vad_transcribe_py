import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np
import numpy.typing as npt
import torch

from zhconv_rs import zhconv

TARGET_SAMPLE_RATE = 16000
ChineseConversion = Literal['none', 'simplified', 'traditional']

WHISPER_HARD_LIMIT_SECONDS = 30
MOONSHINE_HARD_LIMIT_SECONDS = 14


def format_timestamp(seconds: float) -> str:
    """Format seconds to hh:mm:ss.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _parse_wav_header(f):
    """Parse WAV header from file-like object.

    Returns:
        (header_bytes, format_info) where format_info is a dict with
        audio_format, channels, sample_rate, bits_per_sample, data_size,
        or None if not a valid WAV.
        header_bytes contains all bytes consumed from the stream.
    """
    header_bytes = b''

    riff = f.read(12)
    header_bytes += riff
    if len(riff) < 12 or riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
        return header_bytes, None

    fmt_info = None

    while True:
        chunk_hdr = f.read(8)
        header_bytes += chunk_hdr
        if len(chunk_hdr) < 8:
            return header_bytes, None

        chunk_id = chunk_hdr[:4]
        chunk_size = struct.unpack_from('<I', chunk_hdr, 4)[0]

        if chunk_id == b'fmt ':
            fmt_data = f.read(chunk_size)
            header_bytes += fmt_data
            if len(fmt_data) < 16:
                return header_bytes, None
            fmt_info = {
                'audio_format': struct.unpack_from('<H', fmt_data, 0)[0],
                'channels': struct.unpack_from('<H', fmt_data, 2)[0],
                'sample_rate': struct.unpack_from('<I', fmt_data, 4)[0],
                'bits_per_sample': struct.unpack_from('<H', fmt_data, 14)[0],
            }
        elif chunk_id == b'data':
            if fmt_info is not None:
                fmt_info['data_size'] = chunk_size
                return header_bytes, fmt_info
            return header_bytes, None
        else:
            skip = f.read(chunk_size)
            header_bytes += skip
            if len(skip) < chunk_size:
                return header_bytes, None


def _wav_format_matches(fmt_info, target_sample_rate, ac):
    """Check if WAV format is float32 and matches requested params."""
    if fmt_info is None:
        return False
    if fmt_info['audio_format'] != 3 or fmt_info['bits_per_sample'] != 32:
        return False
    if ac is not None and fmt_info['channels'] != ac:
        return False
    if target_sample_rate is not None and fmt_info['sample_rate'] != target_sample_rate:
        return False
    return True


def _stream_raw_float32(f, data_size, chunk_bytes=4096):
    """Stream raw float32 data from file-like object."""
    if data_size == 0 or data_size == 0xFFFFFFFF:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            yield np.frombuffer(chunk, dtype=np.float32)
    else:
        remaining = data_size
        while remaining > 0:
            to_read = min(chunk_bytes, remaining)
            chunk = f.read(to_read)
            if not chunk:
                break
            remaining -= len(chunk)
            yield np.frombuffer(chunk, dtype=np.float32)



def ffmpeg_stream_float32(
    full_audio_path: str | None = None,
    target_sample_rate: int | None = None,
    ac: int | None = None,
    from_stdin: bool = False,
    chunk_bytes: int = 4096,
) -> Generator[npt.NDArray[np.float32], None, None]:
    """Stream audio as float32 arrays, skipping ffmpeg if already float32 WAV.

    Args:
        full_audio_path: Path to audio file or URL (ignored if from_stdin=True)
        target_sample_rate: Target sample rate for output
        ac: Number of audio channels
        from_stdin: If True, read WAV audio from stdin instead of file
        chunk_bytes: Bytes to read per chunk (default 4096 = 1024 float32 samples)

    Yields:
        Float32 numpy arrays of audio samples
    """
    # Try direct WAV reading for files (not URLs)
    if not from_stdin and full_audio_path and not full_audio_path.startswith(('http://', 'https://', 'rtmp://', 'rtsp://')):
        try:
            with open(full_audio_path, 'rb') as f:
                _, fmt_info = _parse_wav_header(f)
                if _wav_format_matches(fmt_info, target_sample_rate, ac):
                    print(f"Direct float32 WAV read: {full_audio_path}", file=sys.stderr)
                    yield from _stream_raw_float32(f, fmt_info['data_size'], chunk_bytes)
                    return
        except OSError:
            pass

    # Parse WAV header from stdin — must be float32 mono at target rate
    if from_stdin:
        _, fmt_info = _parse_wav_header(sys.stdin.buffer)
        if not _wav_format_matches(fmt_info, target_sample_rate, ac):
            raise ValueError(
                f"stdin must be float32 WAV (mono, {target_sample_rate}Hz). "
                f"Got: {fmt_info}"
            )
        print("Direct float32 WAV read from stdin", file=sys.stderr)
        yield from _stream_raw_float32(sys.stdin.buffer, fmt_info['data_size'], chunk_bytes)
        return

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


def get_window_size_samples():
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


def resolve_model_id(model: str, backend: str) -> str:
    """Resolve short model name to full HuggingFace model ID."""
    if '/' in model:
        return model
    if backend == 'whisper':
        return f"openai/whisper-{model}"
    elif backend == 'moonshine':
        return f"UsefulSensors/{model}"
    return model


def create_transcriber(
    language: str,
    model: str = "large-v3-turbo",
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
    """Transcriber using HuggingFace Transformers backends."""

    def __init__(
        self,
        language: str,
        model: str = "large-v3-turbo",
        backend: Literal['whisper', 'moonshine'] = 'whisper',
        chinese_conversion: ChineseConversion = 'none',
    ):
        self.language = language
        self.model = model
        self.backend = backend
        self.chinese_conversion = chinese_conversion

        # Load backend-specific model
        if self.backend == 'whisper':
            self._load_whisper()
        elif self.backend == 'moonshine':
            self._load_moonshine()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @property
    def hard_limit_seconds(self) -> int:
        if self.backend == 'whisper':
            return WHISPER_HARD_LIMIT_SECONDS
        elif self.backend == 'moonshine':
            return MOONSHINE_HARD_LIMIT_SECONDS
        return WHISPER_HARD_LIMIT_SECONDS

    def _load_whisper(self):
        """Load Whisper model via HuggingFace Transformers pipeline."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError:
            raise ImportError(
                "transformers is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]'. "
                "For VAD-only mode without transcription, use the 'split' command instead."
            )

        model_id = resolve_model_id(self.model, self.backend)
        device, torch_dtype = _get_device_and_dtype()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        print(f"Whisper model loaded: {model_id} on {device}", file=sys.stderr)

    def _load_moonshine(self):
        """Load Moonshine model via HuggingFace Transformers."""
        try:
            from transformers import MoonshineForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]'. "
                "For VAD-only mode without transcription, use the 'split' command instead."
            )

        model_id = resolve_model_id(self.model, self.backend)
        device, torch_dtype = _get_device_and_dtype()

        self.moonshine_model = MoonshineForConditionalGeneration.from_pretrained(
            model_id
        ).to(device).to(torch_dtype)

        self.moonshine_processor = AutoProcessor.from_pretrained(model_id)
        self._moonshine_device = device
        self._moonshine_dtype = torch_dtype

        print(f"Moonshine model loaded: {model_id} on {device}", file=sys.stderr)

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
            inputs = self.moonshine_processor(
                audio,
                return_tensors="pt",
                sampling_rate=self.moonshine_processor.feature_extractor.sampling_rate,
            )
            inputs = inputs.to(self._moonshine_device, self._moonshine_dtype)

            # Limit max tokens to prevent hallucination loops
            token_limit_factor = 13 / self.moonshine_processor.feature_extractor.sampling_rate
            seq_lens = inputs.attention_mask.sum(dim=-1)
            max_length = int((seq_lens * token_limit_factor).max().item())

            generated_ids = self.moonshine_model.generate(**inputs, max_length=max_length)
            text = self.moonshine_processor.decode(generated_ids[0], skip_special_tokens=True)

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
