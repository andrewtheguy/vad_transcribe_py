import collections
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from enum import Enum
from time import sleep

import numpy.typing as npt
import scipy
import torch
from silero_vad import load_silero_vad

TARGET_SAMPLE_RATE = 16000


import sounddevice as sd

import soundfile as sf

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

def record():
    fs = TARGET_SAMPLE_RATE
    duration = 10  # seconds
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return myrecording.squeeze()

def get_window_size_samples():
    return 512 if TARGET_SAMPLE_RATE == 16000 else 256

def process_silero(model,audio):
    #print(audio)
    window_size_samples = get_window_size_samples()

    # print(audio)

    # Convert to PyTorch tensor and reshape to (1, num_samples)
    # Silero typically expects a single-channel tensor with shape (1, samples)
    audio_tensor = torch.from_numpy(audio)

    speech_probs = []

    wav = audio_tensor

    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i + window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_prob = model(chunk, TARGET_SAMPLE_RATE).item()
        speech_probs.append((i, speech_prob), )
    model.reset_states()  # reset model states after each audio

    # print indexes of speech_probs where probability is greater than 0.5
    seconds = [i / TARGET_SAMPLE_RATE for i, prob in speech_probs if prob > 0.5]
    print(seconds)

def process_silero_streaming(model,audio):
    #print(audio)
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

transcribe_queue = queue.Queue()

q = queue.Queue()

def transcribe():
    from faster_whisper import WhisperModel
    model_size = "turbo"

    model = WhisperModel(model_size)

    while True:
        audio = transcribe_queue.get(block=True)

        # new_sample_rate = 16000
        #
        # original_sample_rate = TARGET_SAMPLE_RATE
        #
        # # Resample
        # num_samples = int(len(audio) * new_sample_rate / original_sample_rate)
        #
        # resampled_audio = scipy.signal.resample(audio, num_samples)

        print("transcribing audio")
        segments, info = model.transcribe(audio, beam_size=5)

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        # or run on GPU with INT8
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")

def audio_callback(indata, frames, t, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    data_flattened = indata.squeeze()
    #print("frames",frames)
    #print("indata length",len(indata))

    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put((data_flattened,time.time()))

def process_recording():
    model = load_silero_vad()

    window_size_samples = get_window_size_samples()

    transcribing_thread = threading.Thread(target=transcribe)
    transcribing_thread.start()

    cur_has_speech = False
    last_has_speech_ts = None
    speech_section = []
    no_speech_seconds_threshold = 2

    ts = None

    buffer = queue.Queue()

    with sd.InputStream(dtype='float32', callback=audio_callback) as stream:
        input_sample_rate = stream.samplerate
        if stream.channels != 1:
            raise ValueError(f"only support single channel for now")
        while True:
            data_orig,new_ts = q.get(block=True)
            if ts is None:
                ts = new_ts
            elif buffer.qsize() == 0:
                print(f"queue is empty, reset ts {ts},to new_ts {new_ts}",file=sys.stderr)
                ts = new_ts
            if input_sample_rate != TARGET_SAMPLE_RATE:
                data_q = scipy.signal.resample(data_orig, int(len(data_orig) * TARGET_SAMPLE_RATE / input_sample_rate))
            else:
                data_q = data_orig
            for item in data_q:
                buffer.put(item)
            while buffer.qsize() >= window_size_samples:
                data_slice = np.asarray([buffer.get() for _ in range(window_size_samples)])
                ts += window_size_samples / TARGET_SAMPLE_RATE
                if len(data_slice) != window_size_samples:
                    raise ValueError(f"Audio length {len(data_slice)} does not match window size {window_size_samples}")
                has_speech = process_silero_streaming(model,data_slice)
                if has_speech:
                    last_has_speech_ts = ts
                    if not cur_has_speech:
                        print("speech detected",ts,file=sys.stderr)
                        cur_has_speech = True

                if cur_has_speech and ts - last_has_speech_ts > no_speech_seconds_threshold:
                    print(f"no speech detected for {no_speech_seconds_threshold} seconds, stop adding speech and save file",ts,file=sys.stderr)
                    directory = "./tmp/speech"
                    os.makedirs(directory,exist_ok=True)

                    s = np.asarray(speech_section)
                    speech_section = []

                    sf.write(os.path.join(directory,f"{last_has_speech_ts}.wav"), s, TARGET_SAMPLE_RATE)
                    transcribe_queue.put(s)

                    last_has_speech_ts = None
                    cur_has_speech = False


                if cur_has_speech:
                    logging.debug("adding speech",ts)
                    cur_has_speech = True
                    speech_section.extend(data_slice)


