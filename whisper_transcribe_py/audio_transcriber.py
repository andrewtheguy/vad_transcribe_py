
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

import soundfile as sf

from zhconv_rs import zhconv
from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment

TARGET_SAMPLE_RATE = 16000

DEFAULT_CHINESE_LOCALE = 'zh-Hant'

QUEUE_TIME_LIMIT_SECONDS = 60.0
# don't make it too low otherwise it will be stuck in drop mode
QUEUE_RESUME_LIMIT_SECONDS = 15.0

# use ffmpeg to stream audio from url
@contextmanager
def stream_url(url):

    command = [
        "ffmpeg",
        "-i", url,
        "-attempt_recovery", "1",
        "-hide_banner",
        "-loglevel", "error",
        "-recovery_wait_time", "1",
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


AudioSegmentCallback = Callable[[npt.NDArray[np.float32], float], None]
TranscriptPersistenceCallback = Callable[[TranscribedSegment], None]


def create_audio_file_saver(directory: str = "./tmp/speech") -> AudioSegmentCallback:
    os.makedirs(directory, exist_ok=True)

    def _save(audio: npt.NDArray[np.float32], start_timestamp: float):
        sf.write(os.path.join(directory, f"{start_timestamp}.wav"), audio, TARGET_SAMPLE_RATE)

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
        self._drop_callbacks: list[Callable[[Optional[float]], None]] = []
        self._drop_notice_active = False
        if resume_seconds is not None:
            self.resume_seconds = max(0.0, resume_seconds)
        elif self.max_seconds:
            self.resume_seconds = min(self.max_seconds, QUEUE_RESUME_LIMIT_SECONDS)
        else:
            self.resume_seconds = 0.0
        self._drop_mode = False

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
        when the drop occurred. All callbacks must accept a float timestamp.
        """
        if callback is None:
            return
        with self._lock:
            self._drop_callbacks.append(callback)

    def _notify_drop_start(self, timestamp: float, callbacks: list[Callable[[float], None]]) -> None:
        """Notify all registered callbacks of a drop event with the real timestamp.

        Args:
            timestamp: Real wall clock timestamp when the drop occurred (required)
            callbacks: List of callbacks to invoke with the timestamp
        """
        for callback in callbacks:
            try:
                callback(timestamp)
            except Exception:
                logging.exception("Drop notice callback failed for %s", self.source_label)

    def try_add(self, duration_seconds: float, chunk_wall_clock: Optional[float] = None) -> bool:
        """Return True and account for the chunk if it fits under the cap.

        Args:
            duration_seconds: Duration of the audio chunk in seconds
            chunk_wall_clock: Wall clock timestamp of the audio chunk being checked (required).
                            All callers must provide this timestamp.
        """
        if duration_seconds <= 0:
            return True
        callbacks_to_notify: list[Callable[[float], None]] = []
        drop_notice_timestamp: float | None = None
        with self._lock:
            resume_threshold = self.resume_seconds if self.resume_seconds else 0.0
            if resume_threshold > 0 and self._drop_mode and self.current_seconds > resume_threshold:
                self._dropped_seconds += duration_seconds
                self._log_drop(duration_seconds)
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
                if not self._drop_mode or not self._drop_notice_active:
                    callbacks_to_notify = list(self._drop_callbacks)
                    # Fail fast if chunk_wall_clock is missing when drop callbacks are registered
                    if callbacks_to_notify and chunk_wall_clock is None:
                        raise ValueError(
                            f"chunk_wall_clock is required when QueueBacklogLimiter has drop callbacks. "
                            f"All callers must provide real wall clock timestamps for audio chunks. "
                            f"Source: {self.source_label}"
                        )
                    drop_notice_timestamp = chunk_wall_clock if callbacks_to_notify else None
                    self._drop_notice_active = True
                self._drop_mode = True
                self._dropped_seconds += duration_seconds
                self._log_drop(duration_seconds)
            else:
                self.current_seconds += duration_seconds
                self._total_accounted_seconds += duration_seconds
                self._log_backlog_state_locked()
                return True
        # Callbacks are only populated when drop_notice_timestamp is set (both happen together)
        if callbacks_to_notify:
            assert drop_notice_timestamp is not None, "drop_notice_timestamp must be set when callbacks are populated"
            self._notify_drop_start(drop_notice_timestamp, callbacks_to_notify)
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
    ):
        self.mode = mode
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
            self._pending_drop_notice: Optional[TranscriptionNotice] = None
            self._drop_notice_lock = threading.Lock()
            self.wall_clock_reference = wall_clock_reference
            self.queue_backlog_limiter = queue_backlog_limiter
            self._last_transcript_wall_clock: Optional[float] = None
        else:  # file mode
            self._pending_drop_notice = None
            self._drop_notice_lock = None
            self.wall_clock_reference = None
            self.queue_backlog_limiter = None
            self._last_transcript_wall_clock = None

        # Initialize speech detector with callback
        self.speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            on_segment_complete=self._handle_vad_segment,
        )

        if mode == 'livestream' and self.queue_backlog_limiter is not None:
            self.queue_backlog_limiter.register_drop_callback(self._handle_drop_notice)

        self._load_whisper_cpp()

    def _load_whisper_cpp(self):
        from pywhispercpp.model import Model

        self.whisper_cpp_model = Model(self.model,

                                       print_realtime=False,
                                       print_progress=False,
                                       print_timestamps=False,
                                       n_threads=self.n_threads,

                                       )

        print("Whisper.cpp model loaded:")
        print(self.whisper_cpp_model.get_params())
        print(self.whisper_cpp_model.system_info())


    def _transcribe(self):
        self._transcribe_whisper_cpp()

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

    def _transcribe_whisper_cpp(self):
        while True:
            queued_item = self.transcribe_queue.get(block=True)
            if queued_item is None:
                #print("finished transcribing audio",file=sys.stderr)
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

            self.whisper_cpp_model.transcribe(audio, new_segment_callback=self._new_segment_callback, language=self.language)
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

    def _handle_vad_segment(self, segment: AudioSegment) -> None:
        """Callback from SpeechDetector when speech segment completes."""
        if self.audio_segment_callback is not None:
            self.audio_segment_callback(segment.audio, segment.start)

        # Queue the segment for transcription
        self.transcribe_queue.put(segment)

        # Emit any pending drop notice immediately after segment completion (livestream mode only)
        if self.mode == 'livestream':
            self._emit_pending_drop_notice()

        # Consume silence from backlog limiter
        if self.queue_backlog_limiter:
            silence = self.speech_detector.consume_silence()
            if silence > 0:
                self.queue_backlog_limiter.consume(silence)

    def _handle_drop_notice(self, timestamp: float) -> None:
        """
        Store a drop notice to be emitted after the next segment completes.

        Args:
            timestamp: Real wall clock timestamp when the drop occurred (required).
                      This must be a real timestamp from the source, not estimated.

        The timestamp will be adjusted if necessary to maintain monotonicity with
        previously emitted segments (never goes backwards).
        """
        # Ensure timestamps are monotonic - use max of drop timestamp and last transcript
        if self._last_transcript_wall_clock is not None:
            ts_seconds = max(timestamp, self._last_transcript_wall_clock)
        else:
            ts_seconds = timestamp

        notice = TranscriptionNotice("(transcript temporarily dropped)", ts_seconds)
        with self._drop_notice_lock:
            # Store single pending notice (overwrites previous if not yet emitted)
            self._pending_drop_notice = notice
        # Fast-forward downstream timestamps so future segments align with the drop point
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

    def _emit_pending_drop_notice(self) -> None:
        """Emit the pending drop notice immediately (livestream mode only)."""
        if self.mode != 'livestream':
            return

        notice_to_emit: Optional[TranscriptionNotice] = None
        with self._drop_notice_lock:
            if self._pending_drop_notice is not None:
                notice_to_emit = self._pending_drop_notice
                self._pending_drop_notice = None

        if notice_to_emit is not None:
            self.transcribe_queue.put(notice_to_emit)

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
            try:
                segment = self.audio_input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if segment is None:
                print("end of audio", ts, file=sys.stderr)
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
                self.speech_detector.process_window(data_slice, ts, wall_clock_timestamp=None)

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
            try:
                segment = self.audio_input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if segment is None:
                print("end of audio", ts_wall_clock, file=sys.stderr)
                break
            if isinstance(segment, TranscriptionNotice):
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

                # Handle backlog for silence (when not in speech and no accumulated segment)
                if not has_speech and not self.speech_detector.is_in_speech:
                    # Consume pending silence from backlog
                    if self.queue_backlog_limiter is not None:
                        pending_silence = self.speech_detector.pending_silence_duration
                        if pending_silence > 0:
                            self.queue_backlog_limiter.consume(pending_silence)

                # Advance wall clock timestamp
                if ts_wall_clock is not None:
                    ts_wall_clock += window_seconds

        print("finished processing audio", ts_wall_clock, file=sys.stderr)

        # Flush any incomplete speech segment
        if not stop_requested:
            self.speech_detector.flush()

        # Consume any remaining silence
        if self.queue_backlog_limiter:
            remaining_silence = self.speech_detector.consume_silence()
            if remaining_silence > 0:
                self.queue_backlog_limiter.consume(remaining_silence)

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

        # Emit any final pending drop notice
        self._emit_pending_drop_notice()

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
