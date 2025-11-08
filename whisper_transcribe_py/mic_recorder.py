import sys
import time

import sounddevice as sd

from whisper_transcribe_py.speech_detector import AudioSegment, SpeechDetector


class MicRecorder:
    def __init__(self,audio_input_queue):
        self.audio_input_queue = audio_input_queue
        self.stream = sd.InputStream(callback=self.audio_callback)

    def audio_callback(self, indata, frames, t, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        data_flattened = indata.squeeze()
        # print("frames",frames)
        # print("indata length",len(indata))

        # Fancy indexing with mapping creates a (necessary!) copy:
        self.audio_input_queue.put(AudioSegment(start=time.time(), audio=data_flattened))

    def record(self,language):
        with sd.InputStream(dtype='float32', callback=self.audio_callback) as stream:
            input_sample_rate = stream.samplerate
            if stream.channels != 1:
                raise ValueError(f"only support single channel for now")
            SpeechDetector(self.audio_input_queue,language=language).process_input(input_sample_rate)
