import logging
import queue
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt
import scipy.signal

from zhconv_rs import zhconv
from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment

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
    show_name: str
    language: str
    text: str
    start: float
    end: float


TranscriptionCallback = Callable[[list[TranscribedSegment]], None]


class AudioTranscriber:
    """Transcribes audio from a queue using VAD and Whisper."""

    def __init__(
            self,
            audio_input_queue: queue.Queue[AudioSegment],
            language: str,
            show_name="unknown",
            model="large-v3-turbo",
            audio_segment_callback: Optional[Callable[[AudioSegment], None]] = None,
            transcription_callback: Optional[TranscriptionCallback] = None,
            n_threads: int = 1,
            stop_event: Optional[threading.Event] = None,
            backend: Literal['whisper_cpp', 'faster_whisper'] = 'whisper_cpp',
            vad_min_speech_seconds: Optional[float] = None,
            vad_max_speech_seconds: Optional[float] = None,
    ):
        self.backend = backend
        self.transcribe_queue = queue.Queue()
        self.audio_input_queue = audio_input_queue
        self.language = language
        self.audio_segment_callback = audio_segment_callback
        self.transcription_callback = transcription_callback
        self.ts_transcribe_start = None
        self.show_name = show_name
        self.model = model
        self.current_audio_offset = 0.0
        self.n_threads = n_threads
        self.stop_event = stop_event

        # VAD defaults
        default_min_speech = 3.0
        default_max_speech = 60.0

        actual_max_speech = vad_max_speech_seconds if vad_max_speech_seconds is not None else default_max_speech

        # Initialize speech detector with callback
        self.speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            min_speech_seconds=vad_min_speech_seconds if vad_min_speech_seconds is not None else default_min_speech,
            max_speech_seconds=actual_max_speech,
            on_segment_complete=self._handle_vad_segment,
        )

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
            single_segment=True  # for prerecorded mode
        )

        print("Whisper.cpp model loaded:")
        print(self.whisper_cpp_model.get_params())
        print(self.whisper_cpp_model.system_info())

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

    def _transcribe(self):
        """Transcription worker thread."""
        while True:
            queued_item = self.transcribe_queue.get(block=True)
            if queued_item is None:
                break
            if isinstance(queued_item, AudioSegment):
                audio = queued_item.audio
                segment_offset = queued_item.start
            else:
                raise TypeError(
                    f"Expected AudioSegment but got {type(queued_item).__name__}. "
                    f"All audio items in transcribe_queue must be AudioSegment instances."
                )

            self.current_audio_offset = segment_offset
            self.ts_transcribe_start = None
            self._backend_transcribe(audio)

    def _backend_transcribe(self, audio: npt.NDArray[np.float32]) -> list[TranscribedSegment]:
        """Backend-specific transcription method."""
        class SegmentWrapper:
            def __init__(self, t0_ms: int, t1_ms: int, text: str):
                self.t0 = t0_ms
                self.t1 = t1_ms
                self.text = text

        raw_segments: list[SegmentWrapper] = []

        if self.backend == 'whisper_cpp':
            def print_segment(segment):
                relative_start = self.current_audio_offset + segment.t0 / 1000
                relative_end = self.current_audio_offset + segment.t1 / 1000
                print("[%.2f -> %.2f] %s" % (relative_start, relative_end, segment.text))

            whispercpp_results = self.whisper_cpp_model.transcribe(
                audio, new_segment_callback=print_segment, language=self.language
            )
            for segment in whispercpp_results:
                raw_segments.append(SegmentWrapper(segment.t0, segment.t1, segment.text))

        elif self.backend == 'faster_whisper':
            segments, info = self.faster_whisper_model.transcribe(
                audio,
                beam_size=5,
                language=self.language,
                vad_filter=False,
                without_timestamps=True,
            )

            for segment in segments:
                relative_start = self.current_audio_offset + segment.start
                relative_end = self.current_audio_offset + segment.end
                print("[%.2f -> %.2f] %s" % (relative_start, relative_end, segment.text))
                raw_segments.append(SegmentWrapper(
                    int(segment.start * 1000),
                    int(segment.end * 1000),
                    segment.text
                ))
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Transform all raw segments to TranscribedSegment objects
        transcribed_segments = [self._transform_segment(seg) for seg in raw_segments]

        # Call batch callback with all segments
        if self.transcription_callback is not None and transcribed_segments:
            self.transcription_callback(transcribed_segments)

        return transcribed_segments

    def _transform_segment(self, segment) -> TranscribedSegment:
        """Transform a raw segment to TranscribedSegment object."""
        start = self.current_audio_offset + segment.t0 / 1000
        end = self.current_audio_offset + segment.t1 / 1000

        text_for_storage = zhconv(segment.text, DEFAULT_CHINESE_LOCALE) if self.language in ['yue', 'zh'] else segment.text

        return TranscribedSegment(
            show_name=self.show_name,
            language=self.language,
            text=text_for_storage,
            start=start,
            end=end,
        )

    def _handle_vad_segment(self, segment: AudioSegment) -> None:
        """Callback from SpeechDetector when speech segment completes."""
        if self.audio_segment_callback is not None:
            self.audio_segment_callback(segment)

        # Queue the segment for transcription
        self.transcribe_queue.put(segment)

    def _process_input_prerecorded(self, input_sample_rate: int) -> None:
        """
        Process audio input in file/prerecorded mode.

        Simple flow: VAD speech detection → transcribe → output to JSON
        """
        window_size_samples = get_window_size_samples()
        window_seconds = window_size_samples / TARGET_SAMPLE_RATE

        transcribing_thread = threading.Thread(target=self._transcribe)
        transcribing_thread.start()

        ts = None  # Relative timestamp only
        buffer = []
        stop_requested = False

        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                stop_requested = True
                break
            segment = self.audio_input_queue.get()  # blocking
            if segment is None:
                print(f"end of audio, ts={ts}", file=sys.stderr)
                break

            # Initialize timestamp from first segment
            if ts is None:
                ts = segment.start
            elif len(buffer) == 0:
                logging.debug(f"queue is empty, reset ts {ts}, to new_ts {segment.start}")
                ts = segment.start

            # Resample if needed
            if input_sample_rate != TARGET_SAMPLE_RATE:
                data_q = scipy.signal.resample(
                    segment.audio,
                    int(len(segment.audio) * TARGET_SAMPLE_RATE / input_sample_rate)
                )
            else:
                data_q = segment.audio
            buffer.extend(data_q)

            # Process windows through VAD
            while len(buffer) >= window_size_samples:
                arr = buffer[:window_size_samples]
                buffer = buffer[window_size_samples:]
                data_slice = np.asarray(arr)

                # Process window through speech detector (no wall_clock_timestamp for prerecorded mode)
                has_speech = self.speech_detector.process_window(data_slice, ts, wall_clock_timestamp=None)

                # Advance timestamp
                if ts is not None:
                    ts += window_seconds

        print("finished processing audio", ts, file=sys.stderr)

        # Flush any incomplete speech segment
        if not stop_requested:
            self.speech_detector.flush()

        if stop_requested:
            # Drop any queued-but-unprocessed transcribe work so shutdown is fast
            while True:
                try:
                    item = self.transcribe_queue.get_nowait()
                    if item is None:
                        continue
                except queue.Empty:
                    break

        self.transcribe_queue.put(None)
        transcribing_thread.join()

    def transcribe_audio_segment(self, audio: npt.NDArray[np.float32], start_offset: float = 0.0) -> None:
        """
        Transcribe an already-isolated float32 audio buffer directly without running VAD.

        Intended for pre-segmented audio (e.g., database rows).

        Args:
            audio: Float32 numpy array normalized to [-1.0, 1.0]
            start_offset: Relative start timestamp used for callbacks
        """
        if audio is None or len(audio) == 0:
            return

        # Reset timestamps for direct transcription path
        self.current_audio_offset = start_offset
        self.ts_transcribe_start = None
        self._backend_transcribe(audio)
