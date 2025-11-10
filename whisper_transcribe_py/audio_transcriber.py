
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from time import sleep
from typing import Callable, Optional

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
            initial_timestamp: Optional[float] = None,
            resume_seconds: Optional[float] = None,
    ):
        self.max_seconds = max_seconds
        self.source_label = source_label
        self.current_seconds = 0.0
        self._initial_timestamp = initial_timestamp
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

    def try_add(self, duration_seconds: float) -> bool:
        """Return True and account for the chunk if it fits under the cap."""
        if duration_seconds <= 0:
            return True
        callbacks_to_notify: list[Callable[[float], None]] = []
        drop_notice_timestamp: float | None = None
        with self._lock:
            if self._initial_timestamp is None:
                self._initial_timestamp = time.time()
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
                    # Capture real timestamp when drop is detected
                    drop_notice_timestamp = time.time()
                    callbacks_to_notify = list(self._drop_callbacks)
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

    @property
    def initial_timestamp(self) -> Optional[float]:
        with self._lock:
            return self._initial_timestamp

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
        self.transcribe_queue = queue.Queue()
        self._pending_drop_notices: deque[TranscriptionNotice] = deque()
        self._drop_notice_lock = threading.Lock()
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
        self.wall_clock_reference = wall_clock_reference
        self.queue_backlog_limiter = queue_backlog_limiter
        self._last_transcript_wall_clock: Optional[float] = None

        # Initialize speech detector with callback
        self.speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            on_segment_complete=self._handle_vad_segment,
        )

        if self.queue_backlog_limiter is not None:
            self.queue_backlog_limiter.register_drop_callback(self._handle_drop_notice)
        #if transcribe_backend == "faster-whisper":
        #    raise NotImplementedError("faster-whisper is not supported with the recent updates yet")
        #    #self._load_faster_whisper()
        #elif transcribe_backend == "whispercpp":
        self._load_whisper_cpp()
        #else:
        #    raise ValueError(f"Unsupported transcribe backend {transcribe_backend}")
        #self.transcribe_backend = transcribe_backend

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

        # Always use wall clock timestamps
        ts_start_dt = datetime.fromtimestamp(self.ts_transcribe_start + segment.t0 / 1000, timezone.utc)
        ts_end_dt = datetime.fromtimestamp(self.ts_transcribe_start + segment.t1 / 1000, timezone.utc)
        print("[%s -> %s] %s" % (ts_start_dt, ts_end_dt, segment.text))

        text_for_storage = zhconv(segment.text, DEFAULT_CHINESE_LOCALE) if self.language in ['yue', 'zh'] else segment.text

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
                self._emit_notice(queued_item.text, queued_item.timestamp)
                continue
            if isinstance(queued_item, AudioSegment):
                audio = queued_item.audio
                segment_offset = queued_item.start
            else:
                audio = queued_item
                segment_offset = 0.0

            self.current_audio_offset = segment_offset

            # Always use real wall_clock_start from segment
            # All input sources now provide wall_clock_start
            if isinstance(queued_item, AudioSegment):
                self.ts_transcribe_start = queued_item.wall_clock_start
            else:
                # Legacy fallback for non-AudioSegment items (shouldn't happen in practice)
                self.ts_transcribe_start = time.time()

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

        # wall_clock_start is now required for all AudioSegments
        self._release_ready_drop_notices(segment.wall_clock_start)
        self.transcribe_queue.put(segment)

        # Consume silence from backlog limiter
        if self.queue_backlog_limiter:
            silence = self.speech_detector.consume_silence()
            if silence > 0:
                self.queue_backlog_limiter.consume(silence)

    def _handle_drop_notice(self, timestamp: float) -> None:
        """
        Emit a placeholder segment when backlog forces audio shedding.

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
            self._pending_drop_notices.append(notice)
        # Fast-forward downstream timestamps so future segments align with the drop point
        self._last_transcript_wall_clock = ts_seconds

    def _emit_notice(self, text: str, wall_clock_ts: Optional[float]) -> None:
        ts_seconds = wall_clock_ts if wall_clock_ts is not None else self._last_transcript_wall_clock
        if ts_seconds is None:
            ts_seconds = time.time()
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

    def _release_ready_drop_notices(self, up_to_wall_clock: Optional[float]) -> None:
        if up_to_wall_clock is None:
            up_to_wall_clock = float("inf")
        ready: list[TranscriptionNotice] = []
        with self._drop_notice_lock:
            while self._pending_drop_notices:
                notice = self._pending_drop_notices[0]
                notice_ts = notice.timestamp if notice.timestamp is not None else float("-inf")
                if notice_ts > up_to_wall_clock:
                    break
                ready.append(self._pending_drop_notices.popleft())
        for notice in ready:
            self.transcribe_queue.put(notice)

    def _flush_pending_drop_notices(self) -> None:
        self._release_ready_drop_notices(None)

    def process_input(self,input_sample_rate):

        window_size_samples = get_window_size_samples()
        window_seconds = window_size_samples / TARGET_SAMPLE_RATE

        transcribing_thread = threading.Thread(target=self._transcribe)
        transcribing_thread.start()

        ts = None
        buffer = []
        ts_wall_clock = None
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
                print("end of audio",ts,file=sys.stderr)
                break
            if isinstance(segment, TranscriptionNotice):
                self.transcribe_queue.put(segment)
                continue
            segment_duration = segment.duration_seconds
            if segment_duration is None and segment.audio is not None:
                segment_duration = len(segment.audio) / input_sample_rate

            # Use the real wall_clock_start from segment (no estimation/calculation)
            # wall_clock_start is now required, all input sources must provide it
            segment_wall_clock_start = segment.wall_clock_start

            if self.queue_backlog_limiter and segment_duration is not None:
                self.queue_backlog_limiter.note_timestamp_progress(segment_duration)
            if ts is None:
                ts = segment.start
                ts_wall_clock = segment.wall_clock_start
            elif len(buffer) == 0:
                logging.debug(f"queue is empty, reset ts {ts},to new_ts {segment.start}")
                ts = segment.start
                ts_wall_clock = segment.wall_clock_start
            if input_sample_rate != TARGET_SAMPLE_RATE:
                #print("resampling audio")
                data_q = scipy.signal.resample(segment.audio, int(len(segment.audio) * TARGET_SAMPLE_RATE / input_sample_rate))
            else:
                data_q = segment.audio
            buffer.extend(data_q)

            while len(buffer) >= window_size_samples:
                arr = buffer[:window_size_samples]
                buffer = buffer[window_size_samples:]
                data_slice = np.asarray(arr)

                # Process window through speech detector
                has_speech = self.speech_detector.process_window(data_slice, ts, ts_wall_clock)

                # Handle backlog for silence (when not in speech and no accumulated segment)
                if not has_speech and not self.speech_detector.is_in_speech:
                    self._release_ready_drop_notices(ts_wall_clock)
                    # Consume pending silence from backlog
                    if self.queue_backlog_limiter is not None:
                        pending_silence = self.speech_detector.pending_silence_duration
                        if pending_silence > 0:
                            self.queue_backlog_limiter.consume(pending_silence)

                if ts is not None:
                    ts += window_seconds
                if ts_wall_clock is not None:
                    ts_wall_clock += window_seconds

        print("finished processing audio",ts,file=sys.stderr)

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

        self._flush_pending_drop_notices()
        self.transcribe_queue.put(None)
        transcribing_thread.join()


def stream_url_thread(
        url,
        audio_input_queue,
        stop_event=None,
        queue_limiter: Optional["QueueBacklogLimiter"] = None,
):
    ts = 0
    # Capture stream start time as base wall clock timestamp
    base_wall_clock = time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            break
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
                    if queue_limiter and not queue_limiter.try_add(duration_seconds):
                        ts += duration_seconds
                        continue
                    # Use stream start time + relative position as wall clock timestamp
                    audio_input_queue.put(
                        AudioSegment(
                            audio=audio,
                            start=ts,
                            wall_clock_start=base_wall_clock + ts,
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
        audio_input_queue.put(TranscriptionNotice("(transcript source temporary error)", time.time()))
        print("stream_stopped, restarting", file=sys.stderr)
        sleep(0.5)
    print("stream_url_thread exiting", file=sys.stderr)
