
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
    def __init__(self, start: float, audio: npt.NDArray[np.float32]):
        self.start = start
        self.audio = audio

    def __repr__(self):
        return f"AudioSegment(start={self.start}, audio={self.audio})"


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


def trim_audio_queue_backlog(
        audio_queue: queue.Queue,
        max_seconds: float,
        approx_sample_rate: float,
        source_label: str = "audio input",
) -> None:
    """Ensure the queue never buffers more than ``max_seconds`` of audio."""
    if max_seconds is None or max_seconds <= 0:
        return
    approx_sr = approx_sample_rate or TARGET_SAMPLE_RATE
    dropped = 0
    total_seconds = 0.0

    with audio_queue.mutex:
        for item in audio_queue.queue:
            if isinstance(item, AudioSegment) and item.audio is not None:
                total_seconds += len(item.audio) / approx_sr

        while total_seconds > max_seconds and audio_queue.queue:
            oldest = audio_queue.queue[0]
            if oldest is None:
                break
            oldest = audio_queue.queue.popleft()
            if isinstance(oldest, AudioSegment) and oldest.audio is not None:
                total_seconds -= len(oldest.audio) / approx_sr
            dropped += 1

    if dropped:
        print(
            f"Warning: dropped {dropped} buffered audio segment(s) from {source_label} queue "
            f"to keep backlog under {int(max_seconds)} seconds.",
            file=sys.stderr,
        )

class SpeechDetector:
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
        #if transcribe_backend == "faster-whisper":
        #    raise NotImplementedError("faster-whisper is not supported with the recent updates yet")
        #    #self._load_faster_whisper()
        #elif transcribe_backend == "whispercpp":
        self._load_whisper_cpp()
        #else:
        #    raise ValueError(f"Unsupported transcribe backend {transcribe_backend}")
        #self.transcribe_backend = transcribe_backend

    def _load_faster_whisper(self):
        from faster_whisper import WhisperModel
        model_size = "turbo"
        self.faster_whisper_model = WhisperModel(model_size)


    def _load_whisper_cpp(self):
        from pywhispercpp.model import Model

        self.whisper_cpp_model = Model(self.transcribe_model_size,

                                       print_realtime=False,
                                       print_progress=False,
                                       print_timestamps=False,
                                       n_threads=self.n_threads,

                                       )

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

            if self.timestamp_strategy == "wall_clock":
                if self.wall_clock_reference is not None:
                    self.ts_transcribe_start = self.wall_clock_reference + segment_offset
                else:
                    self.ts_transcribe_start = time.time()
            else:
                self.ts_transcribe_start = segment_offset

            self.whisper_cpp_model.transcribe(audio, new_segment_callback=self._new_segment_callback, language=self.language)


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

            if self.timestamp_strategy == "wall_clock":
                if self.wall_clock_reference is not None:
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


    def _process_end_of_speech(self, speech_section, last_has_speech_ts):
        s = np.asarray(speech_section)
        start_ts = last_has_speech_ts if last_has_speech_ts is not None else 0.0

        if self.audio_segment_callback is not None:
            self.audio_segment_callback(s, start_ts)

        self.transcribe_queue.put(AudioSegment(audio=s, start=start_ts))
        self.vad_model.reset_states()


    def process_input(self,input_sample_rate):

        window_size_samples = get_window_size_samples()

        transcribing_thread = threading.Thread(target=self._transcribe)
        transcribing_thread.start()

        prev_has_speech = False

        #has_speech_begin_timestamp = None
        speech_section = []

        min_speech_seconds = 3
        max_speech_seconds = 60
        has_speech_begin_timestamp = None

        ts = None

        prev_slice = None

        buffer = []

        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                break
            try:
                segment = self.audio_input_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if segment is None:
                print("end of audio",ts,file=sys.stderr)
                break
            if ts is None:
                ts = segment.start
            elif len(buffer) == 0:
                logging.debug(f"queue is empty, reset ts {ts},to new_ts {segment.start}")
                ts = segment.start
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
                        if prev_slice is not None:
                            speech_section.extend(prev_slice)
                            prev_slice = None
                        speech_section.extend(data_slice)
                    else:
                        #print("still no speech",ts,file=sys.stderr)
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
                        self._process_end_of_speech(speech_section, has_speech_begin_timestamp)
                        speech_section = []
                        has_speech_begin_timestamp = None
                        prev_slice = None

                prev_has_speech = has_speech

        print("finished processing audio",ts,file=sys.stderr)
        if len(speech_section) > 0 and not (self.stop_event is not None and self.stop_event.is_set()):
            self._process_end_of_speech(speech_section, has_speech_begin_timestamp)
        self.transcribe_queue.put(None)
        transcribing_thread.join()


def stream_url_thread(
        url,
        audio_input_queue,
        stop_event=None,
        max_queue_seconds: Optional[float] = None,
        queue_label: str = "stream",
):
    ts = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            break
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
                audio_input_queue.put(AudioSegment(audio=audio, start=ts))
                if max_queue_seconds is not None:
                    trim_audio_queue_backlog(
                        audio_input_queue,
                        max_queue_seconds=max_queue_seconds,
                        approx_sample_rate=TARGET_SAMPLE_RATE,
                        source_label=queue_label,
                    )
                #print("audio_input_queue size", audio_input_queue.qsize())
                ts += len(audio) / TARGET_SAMPLE_RATE
        if stop_event is not None and stop_event.is_set():
            break
        print("stream_stopped, restarting", file=sys.stderr)
        sleep(0.5)
    print("stream_url_thread exiting", file=sys.stderr)
