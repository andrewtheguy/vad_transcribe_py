import sys
import time
import threading
from typing import Optional

import sounddevice as sd

from whisper_transcribe_py.speech_detector import (
    AudioSegment,
    SpeechDetector,
    TARGET_SAMPLE_RATE,
    QueueBacklogLimiter,
)


class MicRecorder:
    def __init__(
            self,
            audio_input_queue,
            stop_event: Optional[threading.Event] = None,
            queue_limiter: Optional[QueueBacklogLimiter] = None,
            n_threads: int = 1,
    ):
        self.audio_input_queue = audio_input_queue
        self.stop_event = stop_event
        self.queue_limiter = queue_limiter
        self.n_threads = n_threads
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
        duration_seconds = len(data_flattened) / self.approx_input_sample_rate if self.approx_input_sample_rate else 0

        if self.queue_limiter and not self.queue_limiter.try_add(duration_seconds):
            return

        start_ts = time.time()
        # Fancy indexing with mapping creates a (necessary!) copy:
        self.audio_input_queue.put(
            AudioSegment(
                start=start_ts,
                audio=data_flattened,
                duration_seconds=duration_seconds if duration_seconds > 0 else None,
                wall_clock_start=start_ts,
            )
        )

    def record(self,language):
        with sd.InputStream(dtype='float32', callback=self.audio_callback) as stream:
            input_sample_rate = stream.samplerate
            if stream.channels != 1:
                raise ValueError(f"only support single channel for now")
            if input_sample_rate:
                self.approx_input_sample_rate = input_sample_rate
            SpeechDetector(
                self.audio_input_queue,
                language=language,
                stop_event=self.stop_event,
                queue_backlog_limiter=self.queue_limiter,
                n_threads=self.n_threads,
            ).process_input(input_sample_rate)
