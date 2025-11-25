import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from zhconv_rs import zhconv

TARGET_SAMPLE_RATE = 16000
DEFAULT_CHINESE_LOCALE = 'zh-Hant'


@contextmanager
def ffmpeg_get_16bit_pcm(full_audio_path, target_sample_rate=None, ac=None):
    """Convert audio file to 16-bit PCM using ffmpeg with streaming output."""
    command = [
        "ffmpeg",
        "-i", full_audio_path,
        "-f", "s16le",  # Output format
        "-acodec", "pcm_s16le",  # Audio codec
    ]

    if ac is not None:
        command.extend(["-ac", str(ac)])

    if target_sample_rate is not None:
        command.extend(["-ar", str(target_sample_rate)])

    command.extend([
        "-loglevel", "error",  # Suppress extra logs
        "pipe:"  # Output to stdout
    ])

    process = None

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        yield process.stdout
        returncode = process.wait()
        if returncode != 0:
            stderr_output = process.stderr.read().decode('utf-8', errors='replace')
            raise ValueError(f"ffmpeg command failed with return code {returncode}. Error: {stderr_output}")
    finally:
        if process is not None:
            process.stdout.close()
            if process.stderr:
                process.stderr.close()


def pcm_int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """Convert int16 PCM audio data to float32 format."""
    max_int16 = np.iinfo(np.int16).max
    audio_float32 = audio_int16.astype(np.float32) / (max_int16 + 1)
    return audio_float32


def pcm_s16le_to_float32(pcm_bytes: bytes) -> npt.NDArray[np.float32]:
    """Convert raw PCM S16LE (Signed 16-bit Little Endian) bytes to NumPy float32 array."""
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    max_int16 = np.iinfo(np.int16).max
    audio_float32 = audio_int16.astype(np.float32) / (max_int16 + 1)
    return audio_float32


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
) -> 'WhisperTranscriber':
    """Factory function to create a WhisperTranscriber instance."""
    return WhisperTranscriber(
        language=language,
        model=model,
        backend=backend,
        n_threads=n_threads,
    )


class WhisperTranscriber:
    """Simple transcriber without VAD or queues."""

    def __init__(
        self,
        language: str,
        model: str = "large-v3-turbo",
        backend: Literal['whisper_cpp', 'faster_whisper'] = 'whisper_cpp',
        n_threads: int = 1,
    ):
        self.language = language
        self.model = model
        self.backend = backend
        self.n_threads = n_threads

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
                "To use transcription, install with: uv pip install -e '.[transcribe]' "
                "or use --no-transcribe flag for VAD-only mode."
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
                "To use transcription, install with: uv pip install -e '.[transcribe]' "
                "or use --no-transcribe flag for VAD-only mode."
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
                start = start_offset + segment.t0 / 1000
                end = start_offset + segment.t1 / 1000
                print("[%.2f -> %.2f] %s" % (start, end, segment.text))

            whispercpp_results = self.whisper_cpp_model.transcribe(
                audio, new_segment_callback=print_segment, language=self.language
            )
            for segment in whispercpp_results:
                text = self._process_text(segment.text)
                results.append(TranscribedSegment(
                    text=text,
                    start=start_offset + segment.t0 / 1000,
                    end=start_offset + segment.t1 / 1000,
                ))

        elif self.backend == 'faster_whisper':
            segments, _ = self.faster_whisper_model.transcribe(
                audio,
                beam_size=5,
                language=self.language,
                vad_filter=False,
                without_timestamps=True,
            )

            for segment in segments:
                start = start_offset + segment.start
                end = start_offset + segment.end
                print("[%.2f -> %.2f] %s" % (start, end, segment.text))
                text = self._process_text(segment.text)
                results.append(TranscribedSegment(
                    text=text,
                    start=start,
                    end=end,
                ))

        return results

    def _process_text(self, text: str) -> str:
        """Process text for storage (e.g., convert Chinese variants)."""
        if self.language in ['yue', 'zh']:
            return zhconv(text, DEFAULT_CHINESE_LOCALE)
        return text
