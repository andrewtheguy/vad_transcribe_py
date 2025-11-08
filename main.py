import argparse
import json
import os
import queue
import sys
import threading
import tomllib

from dotenv import load_dotenv

from silero_vad import (load_silero_vad)

from whisper_transcribe_py.speech_detector import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32, \
    SpeechDetector, AudioSegment, stream_url_thread, create_audio_file_saver, TranscribedSegment
from whisper_transcribe_py.mic_recorder import MicRecorder
from whisper_transcribe_py.db import build_database_writer, connect_to_database


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


def process_queue(q,language,save_audio=True,show_name=None,audio_segment_callback=None,
                  transcript_persistence_callback=None,transcribe_model_size='large-v3-turbo',segment_callback=None,
                  timestamp_strategy='wall_clock',n_threads=1, stop_event=None):
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
                   n_threads=n_threads,
                   stop_event=stop_event,
                   ).process_input(TARGET_SAMPLE_RATE)

def process_mic(q,language, stop_event=None):
    MicRecorder(q, stop_event=stop_event).record(language=language)


def _request_shutdown(stop_event: threading.Event, audio_queue: queue.Queue):
    """
    Signal worker threads to stop and place a sentinel in the audio queue.
    """
    if not stop_event.is_set():
        stop_event.set()
        audio_queue.put(None)


if __name__ == '__main__':
    load_dotenv()
    argparse = argparse.ArgumentParser()
    argparse.add_argument('action', type=str, choices=['file','mic','config','web'])
    argparse.add_argument('--file', type=str, required=False)
    argparse.add_argument('--lang', type=str, required=False)
    argparse.add_argument('--config', type=str, required=False) # for url live streaming
    argparse.add_argument('--output', type=str, required=False)
    argparse.add_argument('--n-threads', type=int, required=False, default=1, help='Number of threads for whisper model (default: 1)')
    # https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS
    argparse.add_argument('--model', type=str, required=False, default='large-v3-turbo', help='Whisper model name (default: large-v3-turbo)')
    # Web server options
    argparse.add_argument('--host', type=str, required=False, default='0.0.0.0', help='Host to bind web server to (default: 0.0.0.0)')
    argparse.add_argument('--port', type=int, required=False, default=8000, help='Port to bind web server to (default: 8000)')
    argparse.add_argument('--dev', action='store_true', help='Enable development mode with hot reload and CORS')
    args = argparse.parse_args()

    # Skip loading VAD model for web action (it will be loaded on-demand if needed)
    if args.action != 'web':
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
        stop_event = threading.Event()

        # Create a new thread
        thread_transcribe = threading.Thread(target=process_queue, kwargs={
            'q': audio_input_queue,
            'language': args.lang,
            'segment_callback': transcript_writer.add_segment,
            'timestamp_strategy': 'relative',
            'transcribe_model_size': args.model,
            'n_threads': args.n_threads,
            'stop_event': stop_event,
        })

        # Start the thread
        thread_transcribe.start()

        ts = 0
        interrupted = False
        try:
            with ffmpeg_get_16bit_pcm(args.file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
                while True:
                    if stop_event.is_set():
                        break
                    chunk = stdout.read(4096)
                    if not chunk:
                        break
                    audio = pcm_s16le_to_float32(chunk)
                    audio_input_queue.put(AudioSegment(audio=audio, start=ts))
                    ts += len(audio) / TARGET_SAMPLE_RATE
        except KeyboardInterrupt:
            interrupted = True
            print("\nCtrl+C received, stopping transcription...", file=sys.stderr)
        finally:
            _request_shutdown(stop_event, audio_input_queue)
            thread_transcribe.join()

        transcript_writer.flush()
        if interrupted:
            print(f"Transcript written to {output_path} (partial)")
        else:
            print(f"Transcript written to {output_path}")

        #SpeechDetector().process_silero(audio)
    elif args.action == 'mic':
        audio_input_queue = queue.Queue()
        stop_event = threading.Event()
        thread_transcribe = threading.Thread(target=process_mic, args=(audio_input_queue, args.lang, stop_event,))

        # Start the thread
        thread_transcribe.start()

        def stop_mic():
            _request_shutdown(stop_event, audio_input_queue)

        print("Press Ctrl+C to stop microphone capture.")
        try:
            while not stop_event.wait(timeout=1):
                pass
        except KeyboardInterrupt:
            print("\nCtrl+C received, stopping microphone capture...", file=sys.stderr)
            stop_mic()

        thread_transcribe.join()
    elif args.action == 'config':
        if args.lang:
            raise ValueError("Language should be provided in the config file rather than as a command line argument")
        with open(args.config, "rb") as f:
            data = tomllib.load(f)

        # Connect to an existing database
        with connect_to_database() as conn:

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
            stop_event = threading.Event()

            db_writer = build_database_writer(conn, data['show_name'])

            # Create a new thread
            thread_transcribe = threading.Thread(target=process_queue,  kwargs={
                'q': audio_input_queue,
                'language': data['language'],
                'save_audio': False,
                'show_name': data['show_name'],
                'transcript_persistence_callback': db_writer,
                'transcribe_model_size': data.get('transcribe_model_size', 'large-v3-turbo'),
                'n_threads': data.get('n_threads', 1),
                'stop_event': stop_event,
            })

            # Start the thread
            thread_transcribe.start()

            url = data['url']

            thread_streaming = threading.Thread(
                target=stream_url_thread,
                args=(url, audio_input_queue, stop_event,),
                daemon=True
            )

            thread_streaming.start()

            def stop_stream():
                _request_shutdown(stop_event, audio_input_queue)

            print("Press Ctrl+C to stop streaming.")
            try:
                while not stop_event.wait(timeout=1):
                    pass
            except KeyboardInterrupt:
                print("\nCtrl+C received, stopping stream...", file=sys.stderr)
                stop_stream()

            stop_stream()
            thread_transcribe.join()
        print("thread_transcribe joined")
        #thread_streaming.join()
    elif args.action == 'web':
        from whisper_transcribe_py.api.server import run_server

        print(f"Starting web server on {args.host}:{args.port}")
        if args.dev:
            print("Development mode enabled - CORS and hot reload active")
            print("Frontend dev server: http://localhost:5173")
        print(f"API server: http://{args.host}:{args.port}")

        run_server(host=args.host, port=args.port, dev=args.dev)
    else:
        raise ValueError("Invalid action {}".format(args.action))
