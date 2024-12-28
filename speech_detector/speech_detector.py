
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, tzinfo, timezone

import numpy.typing as npt
import scipy
import torch
from silero_vad import load_silero_vad

import soundfile as sf

TARGET_SAMPLE_RATE = 16000

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

class SpeechDetector:
    def __init__(self, audio_input_queue: queue.SimpleQueue[AudioSegment],language: str,show_name="unknown",transcribe_backend="whispercpp",save_file=True,database_connection=None):
        self.model = load_silero_vad()
        self.transcribe_queue = queue.SimpleQueue()
        self.audio_input_queue = audio_input_queue
        self.language = language
        self.save_file = save_file
        self.database_connection = database_connection
        self.ts_transcribe_start = None
        self.show_name = show_name
        if transcribe_backend == "faster-whisper":
            self._load_faster_whisper()
        elif transcribe_backend == "whispercpp":
            self._load_whisper_cpp()
        else:
            raise ValueError(f"Unsupported transcribe backend {transcribe_backend}")
        self.transcribe_backend = transcribe_backend

    def _load_faster_whisper(self):
        from faster_whisper import WhisperModel
        model_size = "turbo"
        self.faster_whisper_model = WhisperModel(model_size)


    def _load_whisper_cpp(self):
        from pywhispercpp.model import Model

        self.whisper_cpp_model = Model('large-v3-turbo',

                                       print_realtime=False,
                                       print_progress=False,
                                       print_timestamps=False,
                                       n_threads=1,

                                       )

    def process_silero(self, audio):
        #return True
        model = self.model
        window_size_samples = get_window_size_samples()

        if len(audio) != window_size_samples:
            raise ValueError(f"Audio length {len(audio)} does not match window size {window_size_samples}")

        # print(audio)

        # Convert to PyTorch tensor and reshape to (1, num_samples)
        # Silero typically expects a single-channel tensor with shape (1, samples)
        audio_tensor = torch.from_numpy(audio)
        speech_prob = model(audio_tensor, TARGET_SAMPLE_RATE).item()
        #print(speech_prob)
        return speech_prob > 0.5
        #    print("Speech detected")

    def _transcribe(self):
        if self.transcribe_backend == "faster-whisper":
            self._transcribe_faster_whisper()
        elif self.transcribe_backend == "whispercpp":
            self._transcribe_whisper_cpp()
        else:
            raise ValueError(f"Unsupported transcribe backend {self.transcribe_backend}")

    def _new_segment_callback(self, segment):
        #print("[%.2fs -> %.2fs] %s" % (segment.t0/1000, segment.t1/1000, segment.text))

        #print("[%.2fs]" % (segment.t0,))
        ts_start = datetime.fromtimestamp(self.ts_transcribe_start+segment.t0/1000,timezone.utc)
        ts_end = datetime.fromtimestamp(self.ts_transcribe_start+segment.t1/1000,timezone.utc)
        print("[%s -> %s] %s" % (ts_start,
                                 ts_end
                                 , segment.text))

        if self.database_connection is not None:
            with self.database_connection.cursor() as cur:
                cur.execute(
                    '''INSERT INTO transcripts (show_name,"timestamp", content) VALUES (%s, %s, %s)''',
                    (self.show_name,ts_start.strftime('%Y-%m-%d %H:%M:%S.%f'), segment.text,))

    def _transcribe_whisper_cpp(self):
        while True:
            print("transcribing queue size",self.transcribe_queue.qsize())
            audio = self.transcribe_queue.get(block=True)
            if audio is None:
                #print("finished transcribing audio",file=sys.stderr)
                break
            #print("transcribing audio")
            self.ts_transcribe_start = time.time()
            self.whisper_cpp_model.transcribe(audio, new_segment_callback=self._new_segment_callback, language=self.language)


    def _transcribe_faster_whisper(self):

        while True:
            audio = self.transcribe_queue.get(block=True)
            if audio is None:
                print("finished transcribing audio",file=sys.stderr)
                break
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
        directory = "./tmp/speech"
        os.makedirs(directory, exist_ok=True)

        s = np.asarray(speech_section)

        if self.save_file:
            sf.write(os.path.join(directory, f"{last_has_speech_ts}.wav"), s, TARGET_SAMPLE_RATE)
        self.transcribe_queue.put(s)


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
            segment = self.audio_input_queue.get(block=True)
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
        if len(speech_section) > 0:
            self._process_end_of_speech(speech_section, has_speech_begin_timestamp)
        self.transcribe_queue.put(None)
        transcribing_thread.join()