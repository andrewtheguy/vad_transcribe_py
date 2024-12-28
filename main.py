import argparse
import os
import queue
import sys
import threading
import time

import torch
from silero_vad import (load_silero_vad,
                        read_audio,
                        get_speech_timestamps,
                        save_audio,
                        VADIterator,
                        collect_chunks)
from silero_vad.utils_vad import languages

from speech_detector.speech_detector import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32, \
    SpeechDetector, AudioSegment, stream_url
from speech_detector.mic_recorder import MicRecorder


def process_queue(q,language):
    print("process_queue")
    print(args.lang)
    SpeechDetector(audio_input_queue=q,language=language).process_input(TARGET_SAMPLE_RATE)

def process_mic(q,language):
    MicRecorder(q).record(language=language)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('action', type=str, choices=['file','mic','url'])
    argparse.add_argument('--file', type=str, required=False)
    argparse.add_argument('--url', type=str, required=False)
    argparse.add_argument('--lang', type=str, required=False, default='yue')
    args = argparse.parse_args()

    vad_model = load_silero_vad()

    if args.action == 'file':
        if not args.file:
            print("Please provide a file path")
            exit(1)
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            exit(1)

        audio_input_queue = queue.Queue()

        # Create a new thread
        thread = threading.Thread(target=process_queue, args=(audio_input_queue,args.lang))

        # Start the thread
        thread.start()

        ts = 0
        with ffmpeg_get_16bit_pcm(args.file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                audio = pcm_s16le_to_float32(chunk)
                #ts = time.time()
                #time.sleep(5)
                # put audio into queue one by one
                audio_input_queue.put(AudioSegment(audio=audio, start=ts))
                ts += len(audio) / TARGET_SAMPLE_RATE

        audio_input_queue.put(None)
        thread.join()

        #SpeechDetector().process_silero(audio)
    elif args.action == 'mic':
        audio_input_queue = queue.Queue()
        thread = threading.Thread(target=process_mic, args=(audio_input_queue,args.lang,))

        # Start the thread
        thread.start()

        # press q and enter to quit
        while True:
            print("Press q and enter to quit")
            input2 = sys.stdin.read(1)
            if input2 == 'q':
                audio_input_queue.put(None)
                break

        thread.join()
    elif args.action == 'url':
        if not args.url:
            print("Please provide a url")
            exit(1)

        audio_input_queue = queue.Queue()

        # Create a new thread
        thread = threading.Thread(target=process_queue, args=(audio_input_queue,args.lang,))

        # Start the thread
        thread.start()

        ts = 0
        with stream_url(args.url) as stdout:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                audio = pcm_s16le_to_float32(chunk)
                #ts = time.time()
                #time.sleep(5)
                # put audio into queue one by one
                audio_input_queue.put(AudioSegment(audio=audio, start=ts))
                ts += len(audio) / TARGET_SAMPLE_RATE

        audio_input_queue.put(None)
        thread.join()
    else:
        raise ValueError("Invalid action {}".format(args.action))