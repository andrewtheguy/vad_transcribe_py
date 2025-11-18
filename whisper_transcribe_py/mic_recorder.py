import sys
import time
import threading
from typing import Optional

try:
    import sounddevice as sd
except ImportError:
    raise ImportError(
        "sounddevice is not installed. "
        "To use microphone recording, install with: uv pip install -e '.[mic]' "
        "Note: Microphone recording is only supported on desktop platforms (Windows, Mac, Linux)."
    )

from whisper_transcribe_py.audio_transcriber import (
    AudioSegment,
    AudioSegmentCallback,
    AudioTranscriber,
    TARGET_SAMPLE_RATE,
    QueueBacklogLimiter,
    TranscriptionCallback,
)


class MicRecorder:
    def __init__(
            self,
            audio_input_queue,
            stop_event: Optional[threading.Event] = None,
            queue_limiter: Optional[QueueBacklogLimiter] = None,
            n_threads: int = 1,
            show_name: str = "unknown",
            transcription_callback: Optional[TranscriptionCallback] = None,
            audio_segment_callback: Optional[AudioSegmentCallback] = None,
            backend: str = 'whisper_cpp',
    ):
        self.audio_input_queue = audio_input_queue
        self.stop_event = stop_event
        self.queue_limiter = queue_limiter
        self.n_threads = n_threads
        self.approx_input_sample_rate = TARGET_SAMPLE_RATE
        self.stream = sd.InputStream(callback=self.audio_callback)
        self.show_name = show_name
        self.transcription_callback = transcription_callback
        self.audio_segment_callback = audio_segment_callback
        self.backend = backend

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

        # Capture timestamp before try_add so drop notices use correct timestamp
        start_ts = time.time()
        if self.queue_limiter and not self.queue_limiter.try_add(duration_seconds, chunk_wall_clock=start_ts):
            return

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
            AudioTranscriber(
                self.audio_input_queue,
                language=language,
                mode='livestream',  # Microphone uses livestream mode
                stop_event=self.stop_event,
                queue_backlog_limiter=self.queue_limiter,
                n_threads=self.n_threads,
                show_name=self.show_name,
                transcription_callback=self.transcription_callback,
                audio_segment_callback=self.audio_segment_callback,
                backend=self.backend,
            )._process_input_livestream(input_sample_rate)
