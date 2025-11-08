
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
from typing import Callable, Optional

import numpy.typing as npt
import scipy
import torch
from silero_vad import load_silero_vad

import soundfile as sf

from zhconv_rs import zhconv

TARGET_SAMPLE_RATE = 16000

DEFAULT_CHINESE_LOCALE = 'zh-Hant'

@contextmanager
def stream_url(url):
    '''

    // Run ffmpeg to get raw PCM (s16le) data at 16kHz
    let mut ffmpeg_process = Command::new("ffmpeg")
        .args(&[
            //-drop_pkts_on_overflow 1
            "-i", input_url,      // Input url
            "-attempt_recovery", "1",
            "-hide_banner",
            "-loglevel", "error",
            "-recovery_wait_time", "1",
            "-f", "s16le",         // Output format: raw PCM, signed 16-bit little-endian
            "-acodec", "pcm_s16le",// Audio codec: PCM 16-bit signed little-endian
            "-ac", "1",            // Number of audio channels (1 = mono)
            "-ar", &format!("{}",target_sample_rate),        // Sample rate: 16 kHz
            "-"                    // Output to stdout
        ])
        .stdout(Stdio::piped())
        //.stderr(Stdio::null()) // Optional: Ignore stderr output
        .spawn()?;
    '''

    command = [
        "ffmpeg",
        "-i", url,
        "-attempt_recovery", "1",
        "-hide_banner",
        "-loglevel", "error",
        "-recovery_wait_time", "1",
        "-f", "s16le",  # Output format
        "-acodec", "pcm_s16le",  # Audio codec
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

class AudioSegment:
    def __init__(
            self,
            start: float,
            audio: npt.NDArray[np.float32],
            duration_seconds: Optional[float] = None,
            wall_clock_start: Optional[float] = None,
    ):
        self.start = start
        self.audio = audio
        self.duration_seconds = duration_seconds
        self.wall_clock_start = wall_clock_start

    def __repr__(self):
        return (
            f"AudioSegment(start={self.start}, duration={self.duration_seconds}, "
            f"wall_clock_start={self.wall_clock_start})"
        )


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
        if resume_seconds is not None:
            self.resume_seconds = max(0.0, resume_seconds)
        elif self.max_seconds:
            self.resume_seconds = min(self.max_seconds, 15.0)
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

    def try_add(self, duration_seconds: float) -> bool:
        """Return True and account for the chunk if it fits under the cap."""
        if duration_seconds <= 0:
            return True
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
            if self.max_seconds is None or self.max_seconds <= 0:
                self.current_seconds += duration_seconds
                self._total_accounted_seconds += duration_seconds
                self._log_backlog_state_locked()
                return True
            if self.current_seconds + duration_seconds > self.max_seconds:
                self._drop_mode = True
                self._dropped_seconds += duration_seconds
                self._log_drop(duration_seconds)
                return False
            self.current_seconds += duration_seconds
            self._total_accounted_seconds += duration_seconds
            self._log_backlog_state_locked()
            return True

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
            self._log_backlog_state_locked()

    def pending_chunk_start_timestamp(self) -> Optional[float]:
        """
        Return the estimated wall-clock start time for the next chunk that will be
        consumed. Includes previously dropped audio so timestamps stay aligned
        with the live source even when backlog forces shedding.
        """

        with self._lock:
            if self._initial_timestamp is None:
                return None
            return self._initial_timestamp + self._dropped_seconds + self._timestamp_consumed_seconds

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

class AudioTranscriber:
    def __init__(
            self,
            audio_input_queue: queue.Queue[AudioSegment],
            language: str,
            show_name="unknown",
            transcribe_model_size="large-v3-turbo",
            audio_segment_callback: Optional[AudioSegmentCallback] = None,
            transcript_persistence_callback: Optional[TranscriptPersistenceCallback] = None,
            segment_callback: Optional[Callable[..., None]] = None,
            timestamp_strategy: str = "wall_clock",
            n_threads: int = 1,
            stop_event: Optional[threading.Event] = None,
            wall_clock_reference: Optional[float] = None,
            queue_backlog_limiter: Optional["QueueBacklogLimiter"] = None,
    ):
        self.vad_model = load_silero_vad()
        self.transcribe_queue = queue.Queue()
        self.audio_input_queue = audio_input_queue
        self.language = language
        self.audio_segment_callback = audio_segment_callback
        self.transcript_persistence_callback = transcript_persistence_callback
        self.ts_transcribe_start = None
        self.show_name = show_name
        self.transcribe_model_size = transcribe_model_size
        self.segment_callback = segment_callback
        self.timestamp_strategy = timestamp_strategy
        self.current_audio_offset = 0.0
        self.n_threads = n_threads
        self.stop_event = stop_event
        self.wall_clock_reference = wall_clock_reference
        self.queue_backlog_limiter = queue_backlog_limiter
        self._pending_silence_seconds = 0.0
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

        self.whisper_cpp_model = Model(self.transcribe_model_size,

                                       print_realtime=False,
                                       print_progress=False,
                                       print_timestamps=False,
                                       n_threads=self.n_threads,

                                       )

        print("Whisper.cpp model loaded:")
        print(self.whisper_cpp_model.get_params())
        print(self.whisper_cpp_model.system_info())

    def process_silero(self, audio):
        #return True
        vad_model = self.vad_model
        window_size_samples = get_window_size_samples()

        if len(audio) != window_size_samples:
            raise ValueError(f"Audio length {len(audio)} does not match window size {window_size_samples}")

        # print(audio)

        # Convert to PyTorch tensor and reshape to (1, num_samples)
        # Silero typically expects a single-channel tensor with shape (1, samples)
        audio_tensor = torch.from_numpy(audio)
        speech_prob = vad_model(audio_tensor, TARGET_SAMPLE_RATE).item()
        #print(speech_prob)
        return speech_prob > 0.5
        #    print("Speech detected")

    def _transcribe(self):
        #if self.transcribe_backend == "faster-whisper":
        #    raise NotImplementedError("faster-whisper is not supported with the recent updates yet")
        #    #self._transcribe_faster_whisper()
        #elif self.transcribe_backend == "whispercpp":
            self._transcribe_whisper_cpp()
        #else:
        #    raise ValueError(f"Unsupported transcribe backend {self.transcribe_backend}")

    def _new_segment_callback(self, segment):
        relative_start = self.current_audio_offset + segment.t0 / 1000
        relative_end = self.current_audio_offset + segment.t1 / 1000

        ts_start_dt = None
        ts_end_dt = None

        if self.timestamp_strategy == "wall_clock":
            ts_start_dt = datetime.fromtimestamp(self.ts_transcribe_start + segment.t0 / 1000, timezone.utc)
            ts_end_dt = datetime.fromtimestamp(self.ts_transcribe_start + segment.t1 / 1000, timezone.utc)
            print("[%s -> %s] %s" % (ts_start_dt, ts_end_dt, segment.text))
        else:
            print("[%.2fs -> %.2fs] %s" % (relative_start, relative_end, segment.text))

        text_for_storage = zhconv(segment.text, DEFAULT_CHINESE_LOCALE) if self.language in ['yue', 'zh'] else segment.text

        if self.transcript_persistence_callback is not None and ts_start_dt is None:
            raise ValueError("Transcript persistence callback requires wall clock timestamps.")

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

    def _transcribe_whisper_cpp(self):
        while True:
            queued_item = self.transcribe_queue.get(block=True)
            if queued_item is None:
                #print("finished transcribing audio",file=sys.stderr)
                break

            if isinstance(queued_item, AudioSegment):
                audio = queued_item.audio
                segment_offset = queued_item.start
            else:
                audio = queued_item
                segment_offset = 0.0

            self.current_audio_offset = segment_offset

            wall_clock_start = getattr(queued_item, "wall_clock_start", None)
            if self.timestamp_strategy == "wall_clock":
                if wall_clock_start is not None:
                    self.ts_transcribe_start = wall_clock_start
                elif self.wall_clock_reference is not None:
                    self.ts_transcribe_start = self.wall_clock_reference + segment_offset
                else:
                    self.ts_transcribe_start = time.time()
            else:
                self.ts_transcribe_start = segment_offset

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


    def _transcribe_faster_whisper(self):

        while True:
            queued_item = self.transcribe_queue.get(block=True)
            if queued_item is None:
                print("finished transcribing audio",file=sys.stderr)
                break

            if isinstance(queued_item, AudioSegment):
                audio = queued_item.audio
                segment_offset = queued_item.start
            else:
                audio = queued_item
                segment_offset = 0.0

            self.current_audio_offset = segment_offset

            wall_clock_start = getattr(queued_item, "wall_clock_start", None)
            if self.timestamp_strategy == "wall_clock":
                if wall_clock_start is not None:
                    self.ts_transcribe_start = wall_clock_start
                elif self.wall_clock_reference is not None:
                    self.ts_transcribe_start = self.wall_clock_reference + segment_offset
                else:
                    self.ts_transcribe_start = time.time()
            else:
                self.ts_transcribe_start = segment_offset

            # else:
            #     sf.write("./tmp/tmp.wav", audio, TARGET_SAMPLE_RATE)
            #continue
            # new_sample_rate = 16000
            #
            # original_sample_rate = TARGET_SAMPLE_RATE
            #
            # # Resample
            # num_samples = int(len(audio) * new_sample_rate / original_sample_rate)
            #
            # resampled_audio = scipy.signal.resample(audio, num_samples)

            print("transcribing audio")
            segments, info = self.faster_whisper_model.transcribe(audio, beam_size=5, language=self.language)

            #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            # or run on GPU with INT8
            # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
            # or run on CPU with INT8
            # model = WhisperModel(model_size, device="cpu", compute_type="int8")

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


    def _process_end_of_speech(self, speech_section, last_has_speech_ts, wall_clock_start):
        s = np.asarray(speech_section)
        start_ts = last_has_speech_ts if last_has_speech_ts is not None else 0.0

        if self.audio_segment_callback is not None:
            self.audio_segment_callback(s, start_ts)

        segment_duration_seconds = len(s) / TARGET_SAMPLE_RATE

        self.transcribe_queue.put(
            AudioSegment(
                audio=s,
                start=start_ts,
                wall_clock_start=wall_clock_start,
                duration_seconds=segment_duration_seconds,
            )
        )
        self.vad_model.reset_states()

    def process_input(self,input_sample_rate):

        window_size_samples = get_window_size_samples()
        window_seconds = window_size_samples / TARGET_SAMPLE_RATE

        transcribing_thread = threading.Thread(target=self._transcribe)
        transcribing_thread.start()

        prev_has_speech = False

        #has_speech_begin_timestamp = None
        speech_section = []

        min_speech_seconds = 3
        max_speech_seconds = 60
        has_speech_begin_timestamp = None
        has_speech_begin_wall_clock = None

        ts = None

        prev_slice = None

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
            segment_duration = segment.duration_seconds
            if segment_duration is None and segment.audio is not None:
                segment_duration = len(segment.audio) / input_sample_rate
            segment_wall_clock_start = getattr(segment, "wall_clock_start", None)
            backlog_wall_clock_start = None
            if self.queue_backlog_limiter and segment_duration is not None:
                backlog_wall_clock_start = self.queue_backlog_limiter.pending_chunk_start_timestamp()
            if backlog_wall_clock_start is not None:
                segment_wall_clock_start = backlog_wall_clock_start
            if segment_wall_clock_start is None:
                segment_wall_clock_start = None
            segment.wall_clock_start = segment_wall_clock_start
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
            #print("buffer size",len(buffer))
            #print("speech_section size",len(speech_section))
            #print("prev_slice size",len(prev_slice) if prev_slice is not None else 0)
            #gc.collect()
            while len(buffer) >= window_size_samples:
                arr = buffer[:window_size_samples]
                buffer = buffer[window_size_samples:]
                data_slice = np.asarray(arr)
                #ts += window_size_samples / TARGET_SAMPLE_RATE
                #if len(data_slice) != window_size_samples:
                #    raise ValueError(f"Audio length {len(data_slice)} does not match window size {window_size_samples}")
                seconds = len(speech_section) / TARGET_SAMPLE_RATE
                has_speech = self.process_silero(data_slice)
                if not prev_has_speech:
                    if has_speech:
                        #print("Transitioning from no speech to speech",file=sys.stderr)
                        has_speech_begin_timestamp = ts
                        has_speech_begin_wall_clock = ts_wall_clock
                        if prev_slice is not None:
                            speech_section.extend(prev_slice)
                            prev_slice = None
                            self._pending_silence_seconds = 0.0
                        speech_section.extend(data_slice)
                    else:
                        #print("still no speech",ts,file=sys.stderr)
                        if self.queue_backlog_limiter is not None:
                            if self._pending_silence_seconds > 0:
                                self.queue_backlog_limiter.consume(self._pending_silence_seconds)
                            self._pending_silence_seconds = window_seconds
                        else:
                            self._pending_silence_seconds = window_seconds
                        prev_slice = data_slice
                else:
                    if seconds > max_speech_seconds:
                        #print("override to no speech because seconds > max_seconds",seconds,file=sys.stderr)
                        has_speech = False
                    elif seconds < min_speech_seconds and not has_speech:
                        #print("override to speech because seconds < min_seconds",seconds,file=sys.stderr)
                        has_speech = True
                    if has_speech:
                        #print("still in speech",ts,file=sys.stderr)
                        speech_section.extend(data_slice)
                    else:
                        #print("Transitioning from speech to no speech",file=sys.stderr)
                        speech_section.extend(data_slice)
                        self._process_end_of_speech(
                            speech_section,
                            has_speech_begin_timestamp,
                            has_speech_begin_wall_clock,
                        )
                        speech_section = []
                        has_speech_begin_timestamp = None
                        has_speech_begin_wall_clock = None
                        prev_slice = None

                prev_has_speech = has_speech

                if ts is not None:
                    ts += window_seconds
                if ts_wall_clock is not None:
                    ts_wall_clock += window_seconds

        print("finished processing audio",ts,file=sys.stderr)

        if len(speech_section) > 0 and not stop_requested:
            self._process_end_of_speech(speech_section, has_speech_begin_timestamp, has_speech_begin_wall_clock)

        if self.queue_backlog_limiter and self._pending_silence_seconds > 0:
            self.queue_backlog_limiter.consume(self._pending_silence_seconds)
            self._pending_silence_seconds = 0.0

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


def stream_url_thread(
        url,
        audio_input_queue,
        stop_event=None,
        queue_limiter: Optional["QueueBacklogLimiter"] = None,
):
    ts = 0
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
                    # ts = time.time()
                    # time.sleep(5)
                    # put audio into queue one by one
                    if stop_event is not None and stop_event.is_set():
                        break
                    duration_seconds = len(audio) / TARGET_SAMPLE_RATE
                    if queue_limiter and not queue_limiter.try_add(duration_seconds):
                        ts += duration_seconds
                        continue
                    audio_input_queue.put(
                        AudioSegment(audio=audio, start=ts, duration_seconds=duration_seconds)
                    )
                    #print("audio_input_queue size", audio_input_queue.qsize())
                    ts += duration_seconds
        except ValueError as exc:
            if stop_event is not None and stop_event.is_set():
                break
            logging.warning("ffmpeg stream exited unexpectedly (%s); retrying shortly.", exc)
        if stop_event is not None and stop_event.is_set():
            break
        print("stream_stopped, restarting", file=sys.stderr)
        sleep(0.5)
    print("stream_url_thread exiting", file=sys.stderr)
