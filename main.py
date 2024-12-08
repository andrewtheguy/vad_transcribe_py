import argparse
import os
import queue
import threading
import time

import torch
from silero_vad import (load_silero_vad,
                        read_audio,
                        get_speech_timestamps,
                        save_audio,
                        VADIterator,
                        collect_chunks)


from speech_detector.speech_detector import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32, \
    SpeechDetector, MicRecorder

def file_function(q):

    SpeechDetector(q).process_input(TARGET_SAMPLE_RATE)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('action', type=str, choices=['file','mic'])
    argparse.add_argument('--file', type=str, required=False)
    args = argparse.parse_args()

    model = load_silero_vad()

    if args.action == 'file':
        if not args.file:
            print("Please provide a file path")
            exit(1)
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            exit(1)

        audio_input_queue = queue.Queue()

        # Create a new thread
        thread = threading.Thread(target=file_function, args=(audio_input_queue,))

        # Start the thread
        thread.start()

        with ffmpeg_get_16bit_pcm(args.file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                audio = pcm_s16le_to_float32(chunk)
                ts = time.time()
                # put audio into queue one by one
                audio_input_queue.put((audio,ts,))

        thread.join()

        #SpeechDetector().process_silero(audio)
    elif args.action == 'mic':
        MicRecorder().record()
        #sf.write('output.wav', audio, TARGET_SAMPLE_RATE)