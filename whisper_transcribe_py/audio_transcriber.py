import subprocess
import sys
from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np
import numpy.typing as npt

from zhconv_rs import zhconv

TARGET_SAMPLE_RATE = 16000
ChineseConversion = Literal['none', 'simplified', 'traditional']


def format_timestamp(seconds: float) -> str:
    """Format seconds to hh:mm:ss.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def ffmpeg_stream_float32(
    full_audio_path: str | None = None,
    target_sample_rate: int | None = None,
    ac: int | None = None,
    from_stdin: bool = False,
    chunk_bytes: int = 4096,
) -> Generator[npt.NDArray[np.float32], None, None]:
    """Stream audio as float32 arrays from ffmpeg.

    Args:
        full_audio_path: Path to audio file or URL (ignored if from_stdin=True)
        target_sample_rate: Target sample rate for output
        ac: Number of audio channels
        from_stdin: If True, read WAV audio from stdin instead of file
        chunk_bytes: Bytes to read per chunk (default 4096 = 1024 float32 samples)

    Yields:
        Float32 numpy arrays of audio samples
    """
    if from_stdin:
        command = [
            "ffmpeg",
            "-i", "pipe:0",  # Read from stdin
        ]
    else:
        command = [
            "ffmpeg",
            "-i", full_audio_path,
        ]

    command.extend([
        "-f", "f32le",  # Output format
        "-acodec", "pcm_f32le",  # Audio codec
    ])

    if ac is not None:
        command.extend(["-ac", str(ac)])

    if target_sample_rate is not None:
        command.extend(["-ar", str(target_sample_rate)])

    command.extend([
        "-loglevel", "error",  # Suppress extra logs
        "pipe:"  # Output to stdout
    ])

    process = subprocess.Popen(
        command,
        stdin=sys.stdin.buffer if from_stdin else None,
        stdout=subprocess.PIPE,
        stderr=None,  # Inherit parent's stderr
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


def create_transcriber(
    language: str,
    model: str = "large-v3-turbo",
    backend: Literal['whisper_cpp', 'faster_whisper'] = 'whisper_cpp',
    n_threads: int = 1,
    chinese_conversion: ChineseConversion = 'none',
) -> 'WhisperTranscriber':
    """Factory function to create a WhisperTranscriber instance."""
    return WhisperTranscriber(
        language=language,
        model=model,
        backend=backend,
        n_threads=n_threads,
        chinese_conversion=chinese_conversion,
    )


class WhisperTranscriber:
    """Simple transcriber without VAD or queues."""

    def __init__(
        self,
        language: str,
        model: str = "large-v3-turbo",
        backend: Literal['whisper_cpp', 'faster_whisper'] = 'whisper_cpp',
        n_threads: int = 1,
        chinese_conversion: ChineseConversion = 'none',
    ):
        self.language = language
        self.model = model
        self.backend = backend
        self.n_threads = n_threads
        self.chinese_conversion = chinese_conversion

        # Load backend-specific model
        if self.backend == 'whisper_cpp':
            self._load_whisper_cpp()
        elif self.backend == 'faster_whisper':
            self._load_faster_whisper()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_whisper_cpp(self):
        """Load whisper.cpp model."""
        try:
            from pywhispercpp.model import Model
        except ImportError:
            raise ImportError(
                "pywhispercpp is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]'. "
                "For VAD-only mode without transcription, use the 'split' command instead."
            )

        self.whisper_cpp_model = Model(
            self.model,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            n_threads=self.n_threads,
            # Anti-looping settings
            single_segment=False,  # Allow multiple segments (was True, causing loops)
            #no_context=True,  # Don't use past transcription as prompt (prevents loops)
            # Use beam search for better accuracy
            params_sampling_strategy=1,  # 1 = BEAM_SEARCH
            beam_search={"beam_size": 5, "patience": -1.0},
            # anti-looping settings
            n_max_text_ctx=64,  # Use max context length
        )

        print("Whisper.cpp model loaded:", file=sys.stderr)
        print(self.whisper_cpp_model.get_params(), file=sys.stderr)
        print(self.whisper_cpp_model.system_info(), file=sys.stderr)

    def _load_faster_whisper(self):
        """Load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]'. "
                "For VAD-only mode without transcription, use the 'split' command instead."
            )

        self.faster_whisper_model = WhisperModel(self.model)

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

        if self.backend == 'whisper_cpp':
            def print_segment(segment):
                # whisper.cpp timestamps are in centiseconds (10ms units)
                start = start_offset + segment.t0 / 100
                end = start_offset + segment.t1 / 100
                start_fmt = format_timestamp(start)
                end_fmt = format_timestamp(end)
                print("[%s -> %s] %s" % (start_fmt, end_fmt, segment.text), file=sys.stderr)

            whispercpp_results = self.whisper_cpp_model.transcribe(
                audio, new_segment_callback=print_segment, language=self.language
            )
            for segment in whispercpp_results:
                text = self._process_text(segment.text)
                # whisper.cpp timestamps are in centiseconds (10ms units), not milliseconds
                results.append(TranscribedSegment(
                    text=text,
                    start=start_offset + segment.t0 / 100,
                    end=start_offset + segment.t1 / 100,
                ))

        elif self.backend == 'faster_whisper':
            segments, _ = self.faster_whisper_model.transcribe(
                audio,
                beam_size=5,
                language=self.language,
                vad_filter=False,
                # Anti-looping settings
                condition_on_previous_text=False,  # Don't use past output as prompt
                repetition_penalty=1.2,  # Penalize repeated tokens
                no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                compression_ratio_threshold=2.4,  # Default, but explicit
                log_prob_threshold=-1.0,  # Default, but explicit
            )

            for segment in segments:
                start = start_offset + segment.start
                end = start_offset + segment.end
                start_fmt = format_timestamp(start)
                end_fmt = format_timestamp(end)
                print("[%s -> %s] %s" % (start_fmt, end_fmt, segment.text), file=sys.stderr)
                text = self._process_text(segment.text)
                results.append(TranscribedSegment(
                    text=text,
                    start=start,
                    end=end,
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
