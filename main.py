import argparse
import os
import queue
import sys
import threading
import time
from os import environ
from time import sleep

from dotenv import load_dotenv

import psycopg
import readchar
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


def process_queue(q,language,save_file=True,show_name=None,database_connection=None):
    print("process_queue")
    print(args.lang)
    SpeechDetector(audio_input_queue=q,language=language,save_file=save_file,
                   show_name=show_name,
                   database_connection=database_connection,
                   ).process_input(TARGET_SAMPLE_RATE)

def process_mic(q,language):
    MicRecorder(q).record(language=language)

def stream_url_thread(url,audio_input_queue):
    ts = 0
    while True:
        with stream_url(url) as stdout:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    break
                audio = pcm_s16le_to_float32(chunk)
                # ts = time.time()
                # time.sleep(5)
                # put audio into queue one by one
                audio_input_queue.put(AudioSegment(audio=audio, start=ts))
                ts += len(audio) / TARGET_SAMPLE_RATE
        print("stream_stopped, restarting", file=sys.stderr)
        sleep(0.5)


if __name__ == '__main__':
    load_dotenv()
    argparse = argparse.ArgumentParser()
    argparse.add_argument('action', type=str, choices=['file','mic','url'])
    argparse.add_argument('--file', type=str, required=False)
    argparse.add_argument('--url', type=str, required=False)
    argparse.add_argument('--show-name', type=str, required=False)
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
        thread_transcribe = threading.Thread(target=process_queue, args=(audio_input_queue, args.lang))

        # Start the thread
        thread_transcribe.start()

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
        thread_transcribe.join()

        #SpeechDetector().process_silero(audio)
    elif args.action == 'mic':
        audio_input_queue = queue.Queue()
        thread_transcribe = threading.Thread(target=process_mic, args=(audio_input_queue, args.lang,))

        # Start the thread
        thread_transcribe.start()

        # press q and enter to quit
        while True:
            print("Press q and enter to quit")
            input2 = sys.stdin.read(1)
            if input2 == 'q':
                audio_input_queue.put(None)
                break

        thread_transcribe.join()
    elif args.action == 'url':
        if not args.url:
            print("Please provide a url")
            exit(1)

        if not args.show_name:
            print("Please provide a show name")
            exit(1)

        # Connect to an existing database
        with psycopg.connect(os.environ['DATABASE_URL'],autocommit=True) as conn:

            # Open a cursor to perform database operations
            with conn.cursor() as cur:
                # Execute a command: this creates a new table
                cur.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                id bigserial PRIMARY KEY,
                show_name varchar(255) NOT NULL,
                "timestamp" TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                content TEXT NOT NULL
                );
                    """)
                cur.execute("""
                create index if not exists transcript_show_name_idx ON transcripts (show_name);
                """)

            audio_input_queue = queue.Queue()

            # Create a new thread
            thread_transcribe = threading.Thread(target=process_queue,  kwargs={
                'q': audio_input_queue,
                'language': args.lang,
                'save_file': False,
                'show_name': args.show_name,
                'database_connection': conn
            })

            # Start the thread
            thread_transcribe.start()

            #stream_url_thread(args.url,audio_input_queue)

            thread_streaming = threading.Thread(target=stream_url_thread, args=(args.url,audio_input_queue,))

            thread_streaming.daemon = True

            thread_streaming.start()

            while True:
                print('press q to quit:')
                char = readchar.readchar()
                if char == 'q':
                    break

            audio_input_queue.put(None)
            thread_transcribe.join()
        print("thread_transcribe joined")
        #thread_streaming.join()
    else:
        raise ValueError("Invalid action {}".format(args.action))