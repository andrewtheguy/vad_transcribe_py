import sys
import time
import threading
from typing import Optional

import sounddevice as sd

from whisper_transcribe_py.speech_detector import (
    AudioSegment,
    SpeechDetector,
    TARGET_SAMPLE_RATE,
    trim_audio_queue_backlog,
)


class MicRecorder:
    def __init__(self, audio_input_queue, stop_event: Optional[threading.Event] = None,
                 queue_time_limit_seconds: Optional[float] = None):
        self.audio_input_queue = audio_input_queue
        self.stop_event = stop_event
        self.queue_time_limit_seconds = queue_time_limit_seconds
        self.approx_input_sample_rate = TARGET_SAMPLE_RATE
        self.stream = sd.InputStream(callback=self.audio_callback)

    def audio_callback(self, indata, frames, t, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        if self.stop_event is not None and self.stop_event.is_set():
            return
        data_flattened = indata.squeeze().copy()
        # print("frames",frames)
        # print("indata length",len(indata))

        # Fancy indexing with mapping creates a (necessary!) copy:
        self.audio_input_queue.put(AudioSegment(start=time.time(), audio=data_flattened))
        if self.queue_time_limit_seconds is not None:
            trim_audio_queue_backlog(
                self.audio_input_queue,
                max_seconds=self.queue_time_limit_seconds,
                approx_sample_rate=self.approx_input_sample_rate,
                source_label="microphone",
            )

    def record(self,language):
        with sd.InputStream(dtype='float32', callback=self.audio_callback) as stream:
            input_sample_rate = stream.samplerate
            if stream.channels != 1:
                raise ValueError(f"only support single channel for now")
            if input_sample_rate:
                self.approx_input_sample_rate = input_sample_rate
            SpeechDetector(self.audio_input_queue,language=language, stop_event=self.stop_event).process_input(input_sample_rate)
