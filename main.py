import argparse
import json
import os
import queue
import sys
import threading
import tomllib

from dotenv import load_dotenv

import psycopg
import readchar
from silero_vad import (load_silero_vad)

from whisper_transcribe_py.speech_detector import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32, \
    SpeechDetector, AudioSegment, stream_url_thread, create_audio_file_saver, TranscribedSegment
from whisper_transcribe_py.mic_recorder import MicRecorder


class JsonTranscriptWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.segments = []

    def add_segment(self, *, start: float, end: float, text: str):
        self.segments.append({
            "start": start,
            "end": end,
            "text": text,
        })

    def flush(self):
        directory = os.path.dirname(self.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump({"segments": self.segments}, f, ensure_ascii=False, indent=2)


def build_database_writer(conn, show_name: str):
    def _persist(segment: TranscribedSegment):
        if segment.start_timestamp is None:
            raise ValueError("Database writes require wall clock timestamps.")
        with conn.cursor() as cur:
            cur.execute(
                '''INSERT INTO transcripts (show_name,"timestamp", content) VALUES (%s, %s, %s)''',
                (show_name, segment.start_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'), segment.text, ))

    return _persist


def process_queue(q,language,save_audio=True,show_name=None,audio_segment_callback=None,
                  transcript_persistence_callback=None,transcribe_model_size='large-v3-turbo',segment_callback=None,
                  timestamp_strategy='wall_clock'):
    print("process_queue")
    if save_audio and audio_segment_callback is None:
        audio_segment_callback = create_audio_file_saver()

    SpeechDetector(audio_input_queue=q,
                   language=language,
                   show_name=show_name,
                   transcribe_model_size=transcribe_model_size,
                   audio_segment_callback=audio_segment_callback,
                   transcript_persistence_callback=transcript_persistence_callback,
                   segment_callback=segment_callback,
                   timestamp_strategy=timestamp_strategy,
                   ).process_input(TARGET_SAMPLE_RATE)

def process_mic(q,language):
    MicRecorder(q).record(language=language)


if __name__ == '__main__':
    load_dotenv()
    argparse = argparse.ArgumentParser()
    argparse.add_argument('action', type=str, choices=['file','mic','config'])
    argparse.add_argument('--file', type=str, required=False)
    argparse.add_argument('--lang', type=str, required=False)
    argparse.add_argument('--config', type=str, required=False) # for url live streaming
    argparse.add_argument('--output', type=str, required=False)
    args = argparse.parse_args()

    vad_model = load_silero_vad()

    if args.action == 'file':
        if not args.file:
            print("Please provide a file path")
            exit(1)
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            exit(1)

        if not args.output:
            print("Please provide an --output path for the JSON transcript")
            exit(1)

        output_path = args.output
        transcript_writer = JsonTranscriptWriter(output_path)

        audio_input_queue = queue.Queue()

        # Create a new thread
        thread_transcribe = threading.Thread(target=process_queue, kwargs={
            'q': audio_input_queue,
            'language': args.lang,
            'segment_callback': transcript_writer.add_segment,
            'timestamp_strategy': 'relative'
        })

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
        transcript_writer.flush()
        print(f"Transcript written to {output_path}")

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
    elif args.action == 'config':
        if args.lang:
            raise ValueError("Language should be provided in the config file rather than as a command line argument")
        with open(args.config, "rb") as f:
            data = tomllib.load(f)

        # Connect to an existing database
        db_timeout = int(os.environ.get('DATABASE_TIMEOUT', '10'))  # Default 10 seconds
        with psycopg.connect(
            os.environ['DATABASE_URL'],
            autocommit=True,
            connect_timeout=db_timeout
        ) as conn:

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

            db_writer = build_database_writer(conn, data['show_name'])

            # Create a new thread
            thread_transcribe = threading.Thread(target=process_queue,  kwargs={
                'q': audio_input_queue,
                'language': data['language'],
                'save_audio': False,
                'show_name': data['show_name'],
                'transcript_persistence_callback': db_writer,
                'transcribe_model_size': data.get('transcribe_model_size', 'large-v3-turbo')
            })

            # Start the thread
            thread_transcribe.start()

            url = data['url']

            thread_streaming = threading.Thread(target=stream_url_thread, args=(url, audio_input_queue,))

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
