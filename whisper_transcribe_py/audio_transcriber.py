import logging
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from time import sleep
from typing import Callable, Literal, Optional

import numpy.typing as npt
import scipy

from zhconv_rs import zhconv
from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment

TARGET_SAMPLE_RATE = 16000

DEFAULT_CHINESE_LOCALE = 'zh-Hant'

# Queue backlog limit must be at least 2x the max speech segment duration (default 60s)
# to ensure a single segment doesn't exceed the queue capacity
QUEUE_TIME_LIMIT_SECONDS = 120.0
# don't make it too low otherwise it will be stuck in drop mode
QUEUE_RESUME_LIMIT_SECONDS = 15.0

# use ffmpeg to stream audio from url
@contextmanager
def stream_url(url):

    command = [
        "ffmpeg",
        "-i", url,
        #"-attempt_recovery", "1",
        "-hide_banner",
        "-loglevel", "error",
        #"-recovery_wait_time", "1",
        "-f", "s16le",  # Output format: raw PCM, signed 16-bit little-endian
        "-acodec", "pcm_s16le",  # Audio codec: PCM 16-bit signed little-endian
        "-ac", "1",  # Number of audio channels (1 = mono)
        "-ar", str(TARGET_SAMPLE_RATE),  # Sample rate: 16 kHz
        "pipe:"  # Output to stdout
    ]
    process = None
    try:
        # Run the command, capturing only stdout
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE  # Pipe stdout
        )
        yield process.stdout
        if process.wait() != 0:
            raise ValueError(f"ffmpeg command failed with return code {process.returncode}")
    finally:
        if process is not None:
            process.stdout.close()


# convert audio to 16 bit pcm with streaming output
# not used by web streaming api
@contextmanager
def ffmpeg_get_16bit_pcm(full_audio_path,target_sample_rate=None,ac=None):
    # Construct the ffmpeg command
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
        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,  # Pipe stdout
            stderr=subprocess.PIPE   # Capture stderr for error reporting
        )
        yield process.stdout
        returncode = process.wait()
        if returncode != 0:
            # Read stderr to get error message
            stderr_output = process.stderr.read().decode('utf-8', errors='replace')
            raise ValueError(f"ffmpeg command failed with return code {returncode}. Error: {stderr_output}")
    finally:
        if process is not None:
            process.stdout.close()
            if process.stderr:
                process.stderr.close()


import numpy as np


def pcm_int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """
    Convert int16 PCM audio data to float32 format.

    Parameters:
    -----------
    audio_int16 : numpy.ndarray
        Input audio data in int16 format

    Returns:
    --------
    numpy.ndarray
        Audio data converted to float32 format, scaled between -1.0 and 1.0
    """
    # Use numpy's iinfo to get the max value for int16
    # This is more robust and explicit than hardcoding 32768.0
    max_int16 = np.iinfo(np.int16).max

    # Normalize int16 audio to float32 range between -1.0 and 1.0
    audio_float32 = audio_int16.astype(np.float32) / (max_int16 + 1)

    return audio_float32


def pcm_s16le_to_float32(pcm_bytes: bytes) -> npt.NDArray[np.float32]:
    """
    Convert raw PCM S16LE (Signed 16-bit Little Endian) bytes to NumPy float32 array.

    Parameters:
    -----------
    pcm_bytes : bytes
        Raw PCM bytes in signed 16-bit little-endian format

    Returns:
    --------
    numpy.ndarray
        Audio data converted to float32 format, scaled between -1.0 and 1.0
    """
    # Convert bytes to int16 NumPy array
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Normalize to float32 range between -1.0 and 1.0
    max_int16 = np.iinfo(np.int16).max
    audio_float32 = audio_int16.astype(np.float32) / (max_int16 + 1)

    return audio_float32

# def record():
#     fs = TARGET_SAMPLE_RATE
#     duration = 10  # seconds
#     myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
#     sd.wait()  # Wait until recording is finished
#     return myrecording.squeeze()

def get_window_size_samples():
    return 512 if TARGET_SAMPLE_RATE == 16000 else 256

# def process_silero(model,audio):
#     #print(audio)
#     window_size_samples = get_window_size_samples()
#
#     # print(audio)
#
#     # Convert to PyTorch tensor and reshape to (1, num_samples)
#     # Silero typically expects a single-channel tensor with shape (1, samples)
#     audio_tensor = torch.from_numpy(audio)
#
#     speech_probs = []
#
#     wav = audio_tensor
#
#     for i in range(0, len(wav), window_size_samples):
#         chunk = wav[i: i + window_size_samples]
#         if len(chunk) < window_size_samples:
#             break
#         speech_prob = model(chunk, TARGET_SAMPLE_RATE).item()
#         speech_probs.append((i, speech_prob), )
#     model.reset_states()  # reset model states after each audio
#
#     # print indexes of speech_probs where probability is greater than 0.5
#     seconds = [i / TARGET_SAMPLE_RATE for i, prob in speech_probs if prob > 0.5]
#     print(seconds)

@dataclass
class TranscriptionNotice:
    text: str
    timestamp: Optional[float] = None


@dataclass
class TranscribedSegment:
    show_name: str
    language: str
    text: str
    relative_start: float
    relative_end: float
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]


AudioSegmentCallback = Callable[[AudioSegment], None]
TranscriptPersistenceCallback = Callable[[TranscribedSegment], None]


def create_audio_file_saver(show_name: str, directory: str = "./tmp/speech") -> AudioSegmentCallback:
    # Base directory for this show
    base_show_directory = os.path.join(directory, show_name)

    def _save(segment: AudioSegment):
        audio = segment.audio

        # Use wall clock timestamps for livestream mode, relative timestamps for file mode
        if segment.wall_clock_start is not None:
            # Livestream mode: use yyyymmddhhmmss.microseconds UTC format
            start_dt = datetime.fromtimestamp(segment.wall_clock_start, timezone.utc)
            start_timestamp = start_dt.strftime("%Y%m%d%H%M%S.%f")

            end_ts = segment.wall_clock_start + len(audio) / TARGET_SAMPLE_RATE
            end_dt = datetime.fromtimestamp(end_ts, timezone.utc)
            end_timestamp = end_dt.strftime("%Y%m%d%H%M%S.%f")

            # Organize by date: tmp/speech/showname/yyyy/mm/dd/
            date_path = start_dt.strftime("%Y/%m/%d")
            target_directory = os.path.join(base_show_directory, date_path)
        else:
            # File mode: use relative timestamps, no date subdirectories
            start_timestamp = f"{segment.start:08.3f}"
            end_timestamp = f"{(segment.start + len(audio) / TARGET_SAMPLE_RATE):08.3f}"
            target_directory = base_show_directory

        # Create target directory if it doesn't exist
        os.makedirs(target_directory, exist_ok=True)

        # Convert float32 audio to int16 for opus encoding
        max_int16 = np.iinfo(np.int16).max
        audio_int16 = (audio * max_int16).astype(np.int16)

        # Create final opus file path
        final_path = os.path.join(target_directory, f"{start_timestamp}-{end_timestamp}.opus")
        
        # Create temporary file path with proper extension for atomic save
        temp_path = f"{final_path}.tmp"
        
        # Use ffmpeg to encode to opus 8kbps with 8kHz sample rate
        command = [
            "ffmpeg",
            "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
            "-acodec", "pcm_s16le",  # Input codec
            "-ac", "1",  # Number of channels (1 = mono)
            "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
            "-i", "pipe:",  # Input from stdin
            "-c:a", "libopus",  # Audio codec: Opus
            "-b:a", "8k",  # Audio bitrate: 8kbps
            #"-ar", "8000",  # Output sample rate: 8kHz
            "-f", "ogg",  # Explicitly specify container format
            "-y",  # Overwrite output file if it exists
            temp_path
        ]

        # # Create final mp4 file path
        # final_path = os.path.join(directory, f"{start_timestamp}-{end_timestamp}.m4a")
        
        # # Create temporary file path with proper extension for atomic save
        # temp_path = f"{final_path}.tmp"
        
        # # Use ffmpeg to encode to opus 8kbps
        # command = [
        #     "ffmpeg",
        #     "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
        #     "-acodec", "pcm_s16le",  # Input codec
        #     "-ac", "1",  # Number of channels (1 = mono)
        #     "-ar", str(TARGET_SAMPLE_RATE),  # Sample rate
        #     "-i", "pipe:",  # Input from stdin
        #     "-c:a", "aac",  # Audio codec: AAC
        #     "-b:a", "8k",  # Audio bitrate: 8kbps
        #     "-f", "ipod",  # Explicitly specify container format
        #     "-y",  # Overwrite output file if it exists
        #     temp_path
        # ]
        
        process = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Write audio data to ffmpeg stdin
            process.stdin.write(audio_int16.tobytes())
            process.stdin.close()
            
            # Wait for ffmpeg to finish
            returncode = process.wait()
            if returncode != 0:
                stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                raise ValueError(f"ffmpeg opus encoding failed with return code {returncode}. Error: {stderr_output}")
            
            # Atomically rename temp file to final file
            os.rename(temp_path, final_path)
                
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
            raise RuntimeError(f"Failed to save opus audio: {e}")
        finally:
            if process and process.stderr:
                process.stderr.close()

    return _save


class QueueBacklogLimiter:
    """
    Tracks how much unprocessed audio (in seconds) is currently buffered so that
    producers can shed load instead of letting queues grow without bound.

    A limiter instance is typically shared between the code that places audio
    chunks on a queue (see `stream_url_thread` or `MicRecorder`) and the
    consumer (`AudioTranscriber`). Producers call `try_add` before enqueuing a
    chunk; if it returns `False` the chunk is dropped. Consumers call
    `consume` after audio is processed to release the accounted time.
    """

    def __init__(
            self,
            max_seconds: Optional[float],
            source_label: str = "audio input",
            resume_seconds: Optional[float] = None,
    ):
        self.max_seconds = max_seconds
        self.source_label = source_label
        self.current_seconds = 0.0
        self._total_accounted_seconds = 0.0
        self._dropped_seconds = 0.0
        self._lock = threading.Lock()
        self._last_logged_backlog_bucket: Optional[int] = None
        self._last_backlog_log_time = 0.0
        self._timestamp_consumed_seconds = 0.0
        self._drop_callback: Optional[Callable[[float], None]] = None
        self._stuck_callback: Optional[Callable[[], None]] = None
        self._drop_notice_active = False
        if resume_seconds is not None:
            self.resume_seconds = max(0.0, resume_seconds)
        elif self.max_seconds:
            self.resume_seconds = min(self.max_seconds, QUEUE_RESUME_LIMIT_SECONDS)
        else:
            self.resume_seconds = 0.0
        self._drop_mode = False
        # Deadlock detection: track backlog progress
        self._stuck_check_interval = 70.0  # Check every 70 seconds
        self._stuck_progress_threshold = 0.5  # Must make at least 0.5s progress
        self._last_stuck_check_time = time.time()
        self._last_stuck_check_backlog = 0.0

    def _log_backlog_state_locked(self) -> None:
        if self.max_seconds is None or self.max_seconds <= 0:
            return
        backlog_seconds = self.current_seconds
        now = time.time()
        if backlog_seconds < 0:
            backlog_seconds = 0.0
        bucket = int(min(backlog_seconds / self.max_seconds * 10, 10))
        if (
                self._last_logged_backlog_bucket == bucket
                and now - self._last_backlog_log_time < 5.0
        ):
            return
        print(
            f"[QueueBacklogLimiter:{self.source_label}] backlog {backlog_seconds:.1f}s / "
            f"{self.max_seconds:.0f}s",
            file=sys.stderr,
        )
        self._last_logged_backlog_bucket = bucket
        self._last_backlog_log_time = now

    def _log_drop(self, duration_seconds: float) -> None:
        backlog_seconds = self.current_seconds
        print(
            f"Warning: dropping newest audio chunk from {self.source_label} queue "
            f"to keep backlog under {int(self.max_seconds)} seconds "
            f"(current backlog {backlog_seconds:.1f}s, chunk {duration_seconds:.2f}s).",
            file=sys.stderr,
        )


    def register_drop_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback to be invoked when dropping audio due to backlog.

        The callback will be called with a real wall clock timestamp (float) indicating
        when the drop occurred. The callback must accept a float timestamp.
        """
        with self._lock:
            self._drop_callback = callback

    def register_stuck_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback to be invoked when backlog is stuck (deadlock detected).

        The callback will be called with a timestamp when the backlog has not made
        sufficient progress (< 0.5s in 70 seconds) while in drop mode, indicating
        a stuck state that requires clearing the queue and resetting.
        """
        with self._lock:
            self._stuck_callback = callback

    def _check_stuck_state_locked(self) -> tuple[bool, Optional[float]]:
        """Check if backlog is stuck and hasn't made progress. Must be called with lock held.

        Returns (is_stuck, timestamp_for_drop_notice) tuple.
        If stuck, also resets the limiter state completely.
        """
        if not self._drop_mode or self._stuck_callback is None:
            return (False, None)

        now = time.time()
        elapsed = now - self._last_stuck_check_time

        if elapsed < self._stuck_check_interval:
            return (False, None)

        # Check if backlog has decreased by at least the threshold
        progress = self._last_stuck_check_backlog - self.current_seconds

        if progress < self._stuck_progress_threshold:
            # Stuck detected - backlog hasn't decreased enough
            print(
                f"ERROR: Backlog stuck for {self.source_label}! "
                f"Only {progress:.2f}s progress in {elapsed:.1f}s (threshold: {self._stuck_progress_threshold}s). "
                f"Current backlog: {self.current_seconds:.1f}s. Resetting everything...",
                file=sys.stderr,
            )

            # Completely reset the limiter state
            old_backlog = self.current_seconds
            self.current_seconds = 0.0
            self._drop_mode = False
            self._drop_notice_active = False
            self._last_stuck_check_time = now
            self._last_stuck_check_backlog = 0.0

            # Return timestamp for drop notice (use current time)
            return (True, time.time())

        # Update checkpoint for next check
        self._last_stuck_check_time = now
        self._last_stuck_check_backlog = self.current_seconds
        return (False, None)

    def try_add(self, duration_seconds: float, chunk_wall_clock: Optional[float] = None) -> bool:
        """Return True and account for the chunk if it fits under the cap.

        Args:
            duration_seconds: Duration of the audio chunk in seconds
            chunk_wall_clock: Wall clock timestamp of the audio chunk being checked (required).
                            All callers must provide this timestamp.
        """
        if duration_seconds <= 0:
            return True
        callback_to_notify: Optional[Callable[[float], None]] = None
        stuck_callback_to_notify: Optional[Callable[[float], None]] = None
        drop_notice_timestamp: float | None = None
        with self._lock:
            resume_threshold = self.resume_seconds if self.resume_seconds else 0.0
            if resume_threshold > 0 and self._drop_mode and self.current_seconds > resume_threshold:
                self._dropped_seconds += duration_seconds
                self._log_drop(duration_seconds)
                # Check for stuck state while in drop mode
                is_stuck, stuck_timestamp = self._check_stuck_state_locked()
                if is_stuck:
                    # Stuck detected - limiter state already reset
                    # Invoke stuck callback to clear queue and emit stuck-specific notice
                    stuck_callback_to_notify = self._stuck_callback
                    drop_notice_timestamp = stuck_timestamp
                    # Don't return here - fall through to invoke callback outside lock
                else:
                    # Not stuck, just normal drop
                    return False
            if self._drop_mode and self.current_seconds <= resume_threshold:
                self._drop_mode = False
                self._drop_notice_active = False
            if self.max_seconds is None or self.max_seconds <= 0:
                self.current_seconds += duration_seconds
                self._total_accounted_seconds += duration_seconds
                self._log_backlog_state_locked()
                return True
            if self.current_seconds + duration_seconds > self.max_seconds:
                was_in_drop_mode = self._drop_mode
                if not self._drop_mode or not self._drop_notice_active:
                    callback_to_notify = self._drop_callback
                    # Fail fast if chunk_wall_clock is missing when drop callback is registered
                    if callback_to_notify is not None and chunk_wall_clock is None:
                        raise ValueError(
                            f"chunk_wall_clock is required when QueueBacklogLimiter has drop callback. "
                            f"All callers must provide real wall clock timestamps for audio chunks. "
                            f"Source: {self.source_label}"
                        )
                    drop_notice_timestamp = chunk_wall_clock if callback_to_notify else None
                    self._drop_notice_active = True
                self._drop_mode = True
                # Initialize stuck detection checkpoint when first entering drop mode
                if not was_in_drop_mode:
                    self._last_stuck_check_time = time.time()
                    self._last_stuck_check_backlog = self.current_seconds
                self._dropped_seconds += duration_seconds
                self._log_drop(duration_seconds)
            else:
                # Only accept chunk if no callbacks are pending (not stuck, not dropping)
                if stuck_callback_to_notify is None and callback_to_notify is None:
                    self.current_seconds += duration_seconds
                    self._total_accounted_seconds += duration_seconds
                    self._log_backlog_state_locked()
                    return True
        # Invoke callbacks outside of lock if needed
        if stuck_callback_to_notify is not None:
            assert drop_notice_timestamp is not None, "drop_notice_timestamp must be set when stuck callback is invoked"
            try:
                stuck_callback_to_notify(drop_notice_timestamp)
            except Exception:
                logging.exception("Stuck callback failed for %s", self.source_label)
        elif callback_to_notify is not None:
            assert drop_notice_timestamp is not None, "drop_notice_timestamp must be set when callback is invoked"
            try:
                callback_to_notify(drop_notice_timestamp)
            except Exception:
                logging.exception("Drop notice callback failed for %s", self.source_label)
        return False

    def consume(self, duration_seconds: float) -> None:
        """Reduce the tracked backlog after a chunk has been processed."""
        if duration_seconds <= 0:
            return
        with self._lock:
            self.current_seconds = max(0.0, self.current_seconds - duration_seconds)
            if (
                self._drop_mode
                and self.resume_seconds
                and self.current_seconds <= self.resume_seconds
            ):
                self._drop_mode = False
                self._drop_notice_active = False
            self._log_backlog_state_locked()


    @property
    def dropped_seconds(self) -> float:
        with self._lock:
            return self._dropped_seconds

    def note_timestamp_progress(self, duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        with self._lock:
            if self._total_accounted_seconds <= 0:
                return
            remaining = self._total_accounted_seconds - self._timestamp_consumed_seconds
            if remaining <= 0:
                return
            advance = min(duration_seconds, remaining)
            self._timestamp_consumed_seconds += advance


def create_default_queue_limiter(
        source_label: str,
        *,
        resume_seconds: Optional[float] = None,
) -> "QueueBacklogLimiter":
    """
    Convenience wrapper that instantiates a limiter using the shared default cap.
    """
    return QueueBacklogLimiter(
        QUEUE_TIME_LIMIT_SECONDS,
        source_label=source_label,
        resume_seconds=resume_seconds,
    )

class AudioTranscriber:
    def __init__(
            self,
            audio_input_queue: queue.Queue[AudioSegment],
            language: str,
            mode: Literal['file', 'livestream'] = 'livestream',
            show_name="unknown",
            model="large-v3-turbo",
            audio_segment_callback: Optional[AudioSegmentCallback] = None,
            transcript_persistence_callback: Optional[TranscriptPersistenceCallback] = None,
            segment_callback: Optional[Callable[..., None]] = None,
            n_threads: int = 1,
            stop_event: Optional[threading.Event] = None,
            wall_clock_reference: Optional[float] = None,
            queue_backlog_limiter: Optional["QueueBacklogLimiter"] = None,
            backend: Literal['whisper_cpp', 'faster_whisper'] = 'whisper_cpp',
            vad_min_speech_seconds: Optional[float] = None,
            vad_max_speech_seconds: Optional[float] = None,
    ):
        self.mode = mode
        self.backend = backend
        self.transcribe_queue = queue.Queue()
        self.audio_input_queue = audio_input_queue
        self.language = language
        self.audio_segment_callback = audio_segment_callback
        self.transcript_persistence_callback = transcript_persistence_callback
        self.ts_transcribe_start = None
        self.show_name = show_name
        self.model = model
        self.segment_callback = segment_callback
        self.current_audio_offset = 0.0
        self.n_threads = n_threads
        self.stop_event = stop_event

        # Mode-specific initialization
        if mode == 'livestream':
            self.wall_clock_reference = wall_clock_reference
            self.queue_backlog_limiter = queue_backlog_limiter
            self._last_transcript_wall_clock: Optional[float] = None
        else:  # file mode
            self.wall_clock_reference = None
            self.queue_backlog_limiter = None
            self._last_transcript_wall_clock = None

        # VAD defaults (same for both modes)
        default_min_speech = 3.0
        default_max_speech = 60.0

        # Determine actual max_speech_seconds value
        actual_max_speech = vad_max_speech_seconds if vad_max_speech_seconds is not None else default_max_speech

        # Initialize speech detector with callback
        self.speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            min_speech_seconds=vad_min_speech_seconds if vad_min_speech_seconds is not None else default_min_speech,
            max_speech_seconds=actual_max_speech,
            on_segment_complete=self._handle_vad_segment,
        )

        if mode == 'livestream' and self.queue_backlog_limiter is not None:
            # Validate that queue limit is at least twice the max speech duration
            if hasattr(self.queue_backlog_limiter, 'max_seconds') and self.queue_backlog_limiter.max_seconds is not None:
                min_required_queue_seconds = 2 * actual_max_speech
                if self.queue_backlog_limiter.max_seconds < min_required_queue_seconds:
                    raise ValueError(
                        f"QUEUE_TIME_LIMIT_SECONDS ({self.queue_backlog_limiter.max_seconds}s) must be at least "
                        f"twice the max speech duration ({actual_max_speech}s). "
                        f"Required minimum: {min_required_queue_seconds}s. "
                        f"Update QUEUE_TIME_LIMIT_SECONDS in audio_transcriber.py or pass a custom limiter."
                    )

            self.queue_backlog_limiter.register_drop_callback(self._handle_drop_notice)
            # Register stuck callback if available (not all limiters may have this)
            if hasattr(self.queue_backlog_limiter, 'register_stuck_callback'):
                self.queue_backlog_limiter.register_stuck_callback(self._handle_stuck_queue)

        # Load backend-specific model
        if self.backend == 'whisper_cpp':
            self._load_whisper_cpp()
        elif self.backend == 'faster_whisper':
            self._load_faster_whisper()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_whisper_cpp(self):
        try:
            from pywhispercpp.model import Model
        except ImportError:
            raise ImportError(
                "pywhispercpp is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]' "
                "or use --no-transcribe flag for VAD-only mode."
            )

        self.whisper_cpp_model = Model(self.model,

                                       print_realtime=False,
                                       print_progress=False,
                                       print_timestamps=False,
                                       n_threads=self.n_threads,

                                       )

        print("Whisper.cpp model loaded:")
        print(self.whisper_cpp_model.get_params())
        print(self.whisper_cpp_model.system_info())

    def _load_faster_whisper(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "To use transcription, install with: uv pip install -e '.[transcribe]' "
                "or use --no-transcribe flag for VAD-only mode."
            )
        #import torch

        # # Determine device and compute type intelligently
        # if torch.cuda.is_available():
        #     device = "cuda"
        #     compute_type = "float16"
        #     print(f"CUDA available, using GPU with {compute_type}")
        # else:
        #     device = "cpu"
        #     compute_type = "int8"
        #     print(f"CUDA not available, using CPU with {compute_type}")

        self.faster_whisper_model = WhisperModel(
            self.model,
            #device=device,
            #compute_type=compute_type
        )

        #print(f"Faster-whisper model loaded: {self.model} on {device} with {compute_type}")

    def _transcribe(self):
        while True:
            queued_item = self.transcribe_queue.get(block=True)
            if queued_item is None:
                break
            if isinstance(queued_item, TranscriptionNotice):
                # Livestream mode only
                if self.mode == 'livestream':
                    self._emit_notice(queued_item.text, queued_item.timestamp)
                    # Reset full transcription context after processing TranscriptionNotice
                    self._reset_transcription_context()
                continue
            if isinstance(queued_item, AudioSegment):
                audio = queued_item.audio
                segment_offset = queued_item.start
            else:
                # All audio items must be AudioSegment
                raise TypeError(
                    f"Expected AudioSegment but got {type(queued_item).__name__}. "
                    f"All audio items in transcribe_queue must be AudioSegment instances."
                )

            self.current_audio_offset = segment_offset

            # File mode: wall_clock_start will be None
            # Livestream mode: wall_clock_start is required
            if self.mode == 'livestream':
                if queued_item.wall_clock_start is None:
                    raise ValueError(
                        "wall_clock_start is required for AudioSegment in livestream mode. "
                        "All input sources must provide wall_clock_timestamp."
                    )
                self.ts_transcribe_start = queued_item.wall_clock_start
            else:
                # File mode doesn't use wall_clock_start
                self.ts_transcribe_start = None

            self._backend_transcribe(audio)
            segment_duration_seconds = None
            if isinstance(queued_item, AudioSegment):
                segment_duration_seconds = queued_item.duration_seconds
            if segment_duration_seconds is None and audio is not None:
                segment_duration_seconds = len(audio) / TARGET_SAMPLE_RATE
            if (
                    self.queue_backlog_limiter is not None
                    and segment_duration_seconds is not None
                    and segment_duration_seconds > 0
            ):
                self.queue_backlog_limiter.consume(segment_duration_seconds)

    def _backend_transcribe(self, audio: npt.NDArray[np.float32]) -> None:
        """
        Backend-specific transcription method.

        For whisper.cpp: Uses callback-based transcription
        For faster-whisper: Iterates through returned segments and calls callback manually
        """
        if self.backend == 'whisper_cpp':
            # whisper.cpp backend (callback-based)
            self.whisper_cpp_model.transcribe(audio, new_segment_callback=self._new_segment_callback, language=self.language)
        elif self.backend == 'faster_whisper':
            # faster-whisper backend (iterator-based)
            # faster-whisper returns an iterator of segments
            # vad_filter=False to disable built-in VAD since we use custom VAD logic
            segments, info = self.faster_whisper_model.transcribe(audio, beam_size=5, language=self.language, vad_filter=False)

            # Iterate through segments and manually call the callback
            for segment in segments:
                # Create a compatible segment object that matches whisper.cpp's format
                # faster-whisper segments have: start (float), end (float), text (str)
                # whisper.cpp segments have: t0 (int ms), t1 (int ms), text (str)
                class SegmentWrapper:
                    def __init__(self, start, end, text):
                        self.t0 = int(start * 1000)  # Convert seconds to milliseconds
                        self.t1 = int(end * 1000)
                        self.text = text

                wrapped_segment = SegmentWrapper(segment.start, segment.end, segment.text)
                self._new_segment_callback(wrapped_segment)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _new_segment_callback(self, segment):
        relative_start = self.current_audio_offset + segment.t0 / 1000
        relative_end = self.current_audio_offset + segment.t1 / 1000

        text_for_storage = zhconv(segment.text, DEFAULT_CHINESE_LOCALE) if self.language in ['yue', 'zh'] else segment.text

        # File mode: only relative timestamps, no wall_clock
        if self.mode == 'file':
            print("[%.2f -> %.2f] %s" % (relative_start, relative_end, segment.text))

            if self.transcript_persistence_callback is not None:
                segment_payload = TranscribedSegment(
                    show_name=self.show_name,
                    language=self.language,
                    text=text_for_storage,
                    relative_start=relative_start,
                    relative_end=relative_end,
                    start_timestamp=None,
                    end_timestamp=None,
                )
                self.transcript_persistence_callback(segment_payload)

            if self.segment_callback is not None:
                self.segment_callback(start=relative_start, end=relative_end, text=text_for_storage)

        # Livestream mode: use wall clock timestamps
        else:
            ts_start_dt = datetime.fromtimestamp(self.ts_transcribe_start + segment.t0 / 1000, timezone.utc)
            ts_end_dt = datetime.fromtimestamp(self.ts_transcribe_start + segment.t1 / 1000, timezone.utc)
            print("[%s -> %s] %s" % (ts_start_dt, ts_end_dt, segment.text))

            if self.transcript_persistence_callback is not None:
                segment_payload = TranscribedSegment(
                    show_name=self.show_name,
                    language=self.language,
                    text=text_for_storage,
                    relative_start=relative_start,
                    relative_end=relative_end,
                    start_timestamp=ts_start_dt,
                    end_timestamp=ts_end_dt,
                )
                self.transcript_persistence_callback(segment_payload)

            if self.segment_callback is not None:
                self.segment_callback(start=relative_start, end=relative_end, text=text_for_storage)

            # Update last transcript wall clock time
            self._last_transcript_wall_clock = ts_end_dt.timestamp()

    def _handle_vad_segment(self, segment: AudioSegment) -> None:
        """Callback from SpeechDetector when speech segment completes."""
        if self.audio_segment_callback is not None:
            self.audio_segment_callback(segment)

        # Queue the segment for transcription
        self.transcribe_queue.put(segment)

        # Consume non-speech gaps from backlog limiter
        if self.queue_backlog_limiter:
            non_speech = self.speech_detector.consume_non_speech()
            if non_speech > 0:
                self.queue_backlog_limiter.consume(non_speech)

    def _handle_drop_notice(self, timestamp: float, message: str = "(transcript temporarily dropped)") -> None:
        """
        Put a drop notice directly into the audio input queue.

        Args:
            timestamp: Real wall clock timestamp when the drop occurred (required).
                      This must be a real timestamp from the source, not estimated.
            message: Custom message for the notice. Defaults to "(transcript temporarily dropped)".

        The timestamp will be adjusted if necessary to maintain monotonicity with
        previously emitted segments (never goes backwards).
        """
        # Ensure timestamps are monotonic - use max of drop timestamp and last transcript
        if self._last_transcript_wall_clock is not None:
            ts_seconds = max(timestamp, self._last_transcript_wall_clock)
        else:
            ts_seconds = timestamp

        notice = TranscriptionNotice(message, ts_seconds)
        # Put notice directly into audio input queue
        self.audio_input_queue.put(notice)
        # Fast-forward downstream timestamps so future segments align with the drop point
        self._last_transcript_wall_clock = ts_seconds

    def _handle_stuck_queue(self, timestamp: float) -> None:
        """
        Handle stuck queue situation by clearing both queues and emitting a stuck notice.

        This is called when the backlog has not made sufficient progress while in
        drop mode, indicating a deadlock or stuck state that requires recovery.
        The limiter state is already reset by the time this is called.

        Args:
            timestamp: Wall clock timestamp when stuck was detected.
        """
        # Clear the audio input queue - all pending items are lost
        audio_cleared = 0
        try:
            while True:
                self.audio_input_queue.get_nowait()
                audio_cleared += 1
        except queue.Empty:
            pass

        # Also clear the transcribe queue - this is where the actual backlog likely is
        transcribe_cleared = 0
        try:
            while True:
                self.transcribe_queue.get_nowait()
                transcribe_cleared += 1
        except queue.Empty:
            pass

        print(
            f"Cleared {audio_cleared} items from audio_input_queue and {transcribe_cleared} items from transcribe_queue for {self.show_name}",
            file=sys.stderr,
        )

        # Put TranscriptionNotice directly into transcribe_queue for immediate processing
        # Bypassing audio_input_queue since processing thread might be blocked
        if self._last_transcript_wall_clock is not None:
            ts_seconds = max(timestamp, self._last_transcript_wall_clock)
        else:
            ts_seconds = timestamp

        notice = TranscriptionNotice("(transcript reset due to stuck queue)", ts_seconds)
        self.transcribe_queue.put(notice)
        self._last_transcript_wall_clock = ts_seconds

    def _emit_notice(self, text: str, wall_clock_ts: Optional[float]) -> None:
        ts_seconds = wall_clock_ts if wall_clock_ts is not None else self._last_transcript_wall_clock
        if ts_seconds is None:
            raise ValueError(
                f"Cannot emit notice '{text}' without a timestamp. "
                f"wall_clock_ts is None and no previous transcript timestamp available. "
                f"All TranscriptionNotice instances must have valid wall clock timestamps."
            )
        ts_dt = datetime.fromtimestamp(ts_seconds, timezone.utc)
        relative_time = 0.0
        if self.wall_clock_reference is not None:
            relative_time = max(0.0, ts_seconds - self.wall_clock_reference)
        print(f"[{ts_dt} -> {ts_dt}] {text}")
        if self.transcript_persistence_callback is not None:
            segment_payload = TranscribedSegment(
                show_name=self.show_name,
                language=self.language,
                text=text,
                relative_start=relative_time,
                relative_end=relative_time,
                start_timestamp=ts_dt,
                end_timestamp=ts_dt,
            )
            self.transcript_persistence_callback(segment_payload)
        if self.segment_callback is not None:
            self.segment_callback(start=relative_time, end=relative_time, text=text)
        self._last_transcript_wall_clock = ts_seconds
        return ts_seconds

    def _emit_drop_notice(self, wall_clock_ts: Optional[float]) -> None:
        self._emit_notice("(transcript temporarily dropped)", wall_clock_ts)

    def _reset_transcription_context(self) -> None:
        """
        Reset full transcription context (livestream mode only).

        Called when TranscriptionNotice is processed in the transcription pipeline.
        Resets everything as if starting over: SpeechDetector state, timestamps, and context.
        """
        if self.mode != 'livestream':
            return

        # Reset SpeechDetector internal state
        self.speech_detector.reset()

        # Reset timestamp tracking
        self.current_audio_offset = 0.0
        self.ts_transcribe_start = None

        # Note: _last_transcript_wall_clock is intentionally NOT reset here
        # because it's used for maintaining monotonicity of future timestamps

    def _reset_processing_state(self) -> None:
        """
        Reset processing state when TranscriptionNotice is encountered in input queue.

        Called in the processing loop when a TranscriptionNotice is dequeued.
        Clears buffer, resets SpeechDetector, as if starting fresh.
        """
        # Reset SpeechDetector state
        self.speech_detector.reset()

        # Note: buffer will be cleared by returning from the TranscriptionNotice handler
        # Note: timestamps will be re-initialized from next segment

    def _process_input_file(self, input_sample_rate: int) -> None:
        """
        Process audio input in file mode (no wall_clock timestamps, no backlog limiting).

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

            # File mode doesn't handle TranscriptionNotice
            if isinstance(segment, TranscriptionNotice):
                continue

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

                # Process window through speech detector (no wall_clock_timestamp for file mode)
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

    def _process_input_livestream(self, input_sample_rate: int) -> None:
        """
        Process audio input in livestream mode (with wall_clock timestamps and backlog limiting).

        Flow: VAD speech detection → transcribe → emit TranscriptionNotice immediately after segment_complete
        """
        window_size_samples = get_window_size_samples()
        window_seconds = window_size_samples / TARGET_SAMPLE_RATE

        transcribing_thread = threading.Thread(target=self._transcribe)
        transcribing_thread.start()

        ts_wall_clock = None  # Wall clock timestamp only (no dual tracking)
        buffer = []
        stop_requested = False

        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                stop_requested = True
                break
            segment = self.audio_input_queue.get()  # blocking
            if segment is None:
                print("end of audio", ts_wall_clock, file=sys.stderr)
                break
            if isinstance(segment, TranscriptionNotice):
                # Reset processing state: clear buffer, reset SpeechDetector
                self._reset_processing_state()
                buffer.clear()
                ts_wall_clock = None  # Reset timestamp tracking

                # Emit the notice to transcribe queue
                self.transcribe_queue.put(segment)
                continue
            segment_duration = segment.duration_seconds
            if segment_duration is None and segment.audio is not None:
                segment_duration = len(segment.audio) / input_sample_rate

            if self.queue_backlog_limiter and segment_duration is not None:
                self.queue_backlog_limiter.note_timestamp_progress(segment_duration)

            # Use wall clock timestamp from input source directly
            if ts_wall_clock is None or len(buffer) == 0:
                ts_wall_clock = segment.wall_clock_start
                if ts_wall_clock is None:
                    raise ValueError(
                        "wall_clock_start is required in livestream mode. "
                        "All input sources must provide wall_clock_timestamp."
                    )

            # Resample if needed
            if input_sample_rate != TARGET_SAMPLE_RATE:
                data_q = scipy.signal.resample(
                    segment.audio,
                    int(len(segment.audio) * TARGET_SAMPLE_RATE / input_sample_rate)
                )
            else:
                data_q = segment.audio
            buffer.extend(data_q)

            while len(buffer) >= window_size_samples:
                arr = buffer[:window_size_samples]
                buffer = buffer[window_size_samples:]
                data_slice = np.asarray(arr)

                # Process window through speech detector
                # Use relative timestamp calculated from wall_clock
                ts_relative = ts_wall_clock - (segment.wall_clock_start - segment.start) if segment.wall_clock_start and segment.start is not None else 0.0
                has_speech = self.speech_detector.process_window(data_slice, ts_relative, ts_wall_clock)

                # Handle backlog for non-speech (when not in speech and no accumulated segment)
                if not has_speech and not self.speech_detector.is_in_speech:
                    # Consume pending non-speech from backlog
                    if self.queue_backlog_limiter is not None:
                        pending_non_speech = self.speech_detector.pending_non_speech_duration
                        if pending_non_speech > 0:
                            self.queue_backlog_limiter.consume(pending_non_speech)

                # Advance wall clock timestamp
                if ts_wall_clock is not None:
                    ts_wall_clock += window_seconds

        print("finished processing audio", ts_wall_clock, file=sys.stderr)

        # Flush any incomplete speech segment
        if not stop_requested:
            self.speech_detector.flush()

        # Consume any remaining non-speech backlog
        if self.queue_backlog_limiter:
            remaining_non_speech = self.speech_detector.consume_non_speech()
            if remaining_non_speech > 0:
                self.queue_backlog_limiter.consume(remaining_non_speech)

        if stop_requested:
            # Drop any queued-but-unprocessed transcribe work so shutdown is fast
            while True:
                try:
                    item = self.transcribe_queue.get_nowait()
                    if item is None:
                        # do not requeue sentinel; loop will add a fresh one
                        continue
                except queue.Empty:
                    break

        self.transcribe_queue.put(None)
        transcribing_thread.join()

    def process_input(self, input_sample_rate: int) -> None:
        """
        Dispatch to mode-specific process_input method.
        """
        if self.mode == 'file':
            self._process_input_file(input_sample_rate)
        else:  # livestream
            self._process_input_livestream(input_sample_rate)


def stream_url_thread(
        url,
        audio_input_queue,
        stop_event=None,
        queue_limiter: Optional["QueueBacklogLimiter"] = None,
):
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        # Reset timestamp tracking for each stream attempt (including retries)
        ts = 0
        base_wall_clock = time.time()
        try:
            with stream_url(url) as stdout:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break
                    chunk = stdout.read(TARGET_SAMPLE_RATE*2) # 1 second
                    if not chunk:
                        break
                    audio = pcm_s16le_to_float32(chunk)
                    # put audio into queue one by one
                    if stop_event is not None and stop_event.is_set():
                        break
                    duration_seconds = len(audio) / TARGET_SAMPLE_RATE
                    # Calculate wall clock timestamp for this chunk
                    chunk_wall_clock = base_wall_clock + ts
                    if queue_limiter and not queue_limiter.try_add(duration_seconds, chunk_wall_clock=chunk_wall_clock):
                        ts += duration_seconds
                        continue
                    # Use stream start time + relative position as wall clock timestamp
                    audio_input_queue.put(
                        AudioSegment(
                            audio=audio,
                            start=ts,
                            wall_clock_start=chunk_wall_clock,
                            duration_seconds=duration_seconds
                        )
                    )
                    #print("audio_input_queue size", audio_input_queue.qsize())
                    ts += duration_seconds
        except ValueError as exc:
            if stop_event is not None and stop_event.is_set():
                break
            logging.warning("ffmpeg stream exited unexpectedly (%s); retrying shortly.", exc)
        if stop_event is not None and stop_event.is_set():
            break
        # Use the wall clock timestamp of where the stream ended (last audio position)
        # not the current time when the error was detected
        error_wall_clock = base_wall_clock + ts
        audio_input_queue.put(TranscriptionNotice("(transcript source temporary error)", error_wall_clock))
        print("stream_stopped, restarting", file=sys.stderr)
        sleep(0.5)
    print("stream_url_thread exiting", file=sys.stderr)

