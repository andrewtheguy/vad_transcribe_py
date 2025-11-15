import argparse
import json
import os
import queue
import sys
import threading
import time
import tomllib
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from silero_vad import (load_silero_vad)

import numpy as np
import sounddevice as sd

from whisper_transcribe_py.audio_transcriber import TARGET_SAMPLE_RATE, ffmpeg_get_16bit_pcm, pcm_s16le_to_float32, \
    AudioTranscriber, AudioSegment, stream_url_thread, create_audio_file_saver, TranscribedSegment, QueueBacklogLimiter, \
    TranscriptPersistenceCallback, create_default_queue_limiter
from whisper_transcribe_py.mic_recorder import MicRecorder
from whisper_transcribe_py.db import build_database_writer, connect_to_database, initialize_database_schema
from whisper_transcribe_py.vad_processor import SpeechDetector, get_window_size_samples
from file_lock import acquire_lock, LockError


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


def process_vad_only(audio_input_queue, show_name, stop_event=None):
    """
    Process audio through VAD only, saving detected speech segments without transcription.
    File mode - no wall clock timestamps, no backlog limiting.
    """
    audio_segment_callback = create_audio_file_saver(show_name)

    def on_segment_complete(segment: AudioSegment):
        """Called when VAD detects a complete speech segment."""
        audio_segment_callback(segment)
        print(f"Saved audio segment at {segment.start:.2f}s", file=sys.stderr)

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete
    )

    window_size_samples = get_window_size_samples()
    buffer = []
    current_ts = 0  # Track the current timestamp position

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        item = audio_input_queue.get()
        if item is None:
            # Sentinel value - process any remaining audio
            break

        buffer.extend(item.audio)

        # Process complete windows
        while len(buffer) >= window_size_samples:
            window = np.array(buffer[:window_size_samples], dtype=np.float32)
            speech_detector.process_window(window, current_ts, wall_clock_timestamp=None)
            buffer = buffer[window_size_samples:]
            current_ts += window_size_samples / TARGET_SAMPLE_RATE

    # Flush any remaining speech segment
    speech_detector.flush()
    print("VAD processing complete", file=sys.stderr)


def process_vad_only_livestream(audio_input_queue, show_name, input_sample_rate, stop_event=None,
                                  queue_backlog_limiter: Optional[QueueBacklogLimiter] = None):
    """
    Process audio through VAD only for livestream mode (mic/stream).
    Handles wall clock timestamps and queue backlog limiting.
    """
    import scipy.signal

    audio_segment_callback = create_audio_file_saver(show_name)

    def on_segment_complete(segment: AudioSegment):
        """Called when VAD detects a complete speech segment."""
        audio_segment_callback(segment)
        timestamp = datetime.fromtimestamp(segment.wall_clock_start, timezone.utc) if segment.wall_clock_start else None
        print(f"Saved audio segment at {timestamp}", file=sys.stderr)

        # Consume speech segment duration from backlog
        if queue_backlog_limiter:
            segment_duration = segment.duration_seconds
            if segment_duration is None and segment.audio is not None:
                segment_duration = len(segment.audio) / TARGET_SAMPLE_RATE
            if segment_duration is not None and segment_duration > 0:
                queue_backlog_limiter.consume(segment_duration)

        # Consume non-speech gaps from backlog limiter
        if queue_backlog_limiter:
            non_speech = speech_detector.consume_non_speech()
            if non_speech > 0:
                queue_backlog_limiter.consume(non_speech)

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete
    )

    # Validate queue limit is at least twice the max speech duration
    if queue_backlog_limiter and hasattr(queue_backlog_limiter, 'max_seconds') and queue_backlog_limiter.max_seconds is not None:
        min_required_queue_seconds = 2 * speech_detector.max_speech_seconds
        if queue_backlog_limiter.max_seconds < min_required_queue_seconds:
            raise ValueError(
                f"Queue limit ({queue_backlog_limiter.max_seconds}s) must be at least "
                f"twice the max speech duration ({speech_detector.max_speech_seconds}s). "
                f"Required minimum: {min_required_queue_seconds}s. "
                f"Update QUEUE_TIME_LIMIT_SECONDS in audio_transcriber.py or pass a custom limiter."
            )

    window_size_samples = get_window_size_samples()
    window_seconds = window_size_samples / TARGET_SAMPLE_RATE
    buffer = []
    ts_wall_clock = None

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        segment = audio_input_queue.get()
        if segment is None:
            print("End of audio stream", file=sys.stderr)
            break

        # Handle TranscriptionNotice (for stream recovery)
        from whisper_transcribe_py.audio_transcriber import TranscriptionNotice
        if isinstance(segment, TranscriptionNotice):
            # Reset VAD state
            speech_detector.reset()
            buffer.clear()
            ts_wall_clock = None
            continue

        # Track backlog progress
        segment_duration = segment.duration_seconds
        if segment_duration is None and segment.audio is not None:
            segment_duration = len(segment.audio) / input_sample_rate
        if queue_backlog_limiter and segment_duration is not None:
            queue_backlog_limiter.note_timestamp_progress(segment_duration)

        # Initialize wall clock timestamp
        if ts_wall_clock is None or len(buffer) == 0:
            ts_wall_clock = segment.wall_clock_start
            if ts_wall_clock is None:
                raise ValueError("wall_clock_start is required in livestream mode")

        # Resample if needed
        if input_sample_rate != TARGET_SAMPLE_RATE:
            data_q = scipy.signal.resample(
                segment.audio,
                int(len(segment.audio) * TARGET_SAMPLE_RATE / input_sample_rate)
            )
        else:
            data_q = segment.audio
        buffer.extend(data_q)

        # Process complete windows
        while len(buffer) >= window_size_samples:
            window = np.array(buffer[:window_size_samples], dtype=np.float32)
            buffer = buffer[window_size_samples:]

            # Calculate relative timestamp from wall clock
            ts_relative = ts_wall_clock - (segment.wall_clock_start - segment.start) if segment.wall_clock_start and segment.start is not None else 0.0
            has_speech = speech_detector.process_window(window, ts_relative, ts_wall_clock)

            # Consume non-speech from backlog
            if not has_speech and not speech_detector.is_in_speech:
                if queue_backlog_limiter is not None:
                    pending_non_speech = speech_detector.pending_non_speech_duration
                    if pending_non_speech > 0:
                        queue_backlog_limiter.consume(pending_non_speech)

            # Advance wall clock timestamp
            if ts_wall_clock is not None:
                ts_wall_clock += window_seconds

    # Flush any remaining speech segment
    print("Flushing remaining audio", file=sys.stderr)
    speech_detector.flush()

    # Consume any remaining non-speech backlog
    if queue_backlog_limiter:
        remaining_non_speech = speech_detector.consume_non_speech()
        if remaining_non_speech > 0:
            queue_backlog_limiter.consume(remaining_non_speech)

    print("VAD processing complete", file=sys.stderr)


def capture_mic_to_queue(audio_input_queue, stop_event, queue_limiter: Optional[QueueBacklogLimiter] = None):
    """
    Capture microphone audio and put it into a queue for VAD processing.
    Simple version for --no-transcribe mode - no model loading.
    Blocks until stop_event is set.
    Returns the input sample rate.
    """
    approx_input_sample_rate = [TARGET_SAMPLE_RATE]  # Use list to allow modification in callback

    def audio_callback(indata, frames, t, status):
        """Called from separate thread for each audio block."""
        if status:
            print(status, file=sys.stderr)
        if stop_event.is_set():
            return

        data_flattened = indata.squeeze().copy()
        duration_seconds = len(data_flattened) / approx_input_sample_rate[0] if approx_input_sample_rate[0] else 0

        # Capture timestamp before try_add
        start_ts = time.time()
        if queue_limiter and not queue_limiter.try_add(duration_seconds, chunk_wall_clock=start_ts):
            return

        audio_input_queue.put(
            AudioSegment(
                start=start_ts,
                audio=data_flattened,
                duration_seconds=duration_seconds if duration_seconds > 0 else None,
                wall_clock_start=start_ts,
            )
        )

    # Start microphone capture
    with sd.InputStream(dtype='float32', callback=audio_callback) as stream:
        input_sample_rate = stream.samplerate
        if stream.channels != 1:
            raise ValueError("Only support single channel for now")
        if input_sample_rate:
            approx_input_sample_rate[0] = input_sample_rate

        print(f"Microphone capture started (sample rate: {input_sample_rate} Hz)", file=sys.stderr)

        # Wait for stop event
        try:
            while not stop_event.wait(timeout=1):
                pass
        except KeyboardInterrupt:
            pass

        print("Microphone capture stopping...", file=sys.stderr)

    return approx_input_sample_rate[0]


def process_queue(q,language,save_audio=True,show_name=None,audio_segment_callback=None,
                  transcript_persistence_callback=None,model='large-v3-turbo',segment_callback=None,
                  n_threads=1, stop_event=None,
                  queue_backlog_limiter: Optional[QueueBacklogLimiter] = None,
                  mode='livestream', backend='whisper_cpp'):
    print("process_queue")
    if save_audio and audio_segment_callback is None:
        if show_name is None:
            raise ValueError("show_name is required when save_audio=True and no audio_segment_callback provided")
        audio_segment_callback = create_audio_file_saver(show_name)

    AudioTranscriber(audio_input_queue=q,
                     language=language,
                     mode=mode,
                     show_name=show_name,
                     model=model,
                     audio_segment_callback=audio_segment_callback,
                     transcript_persistence_callback=transcript_persistence_callback,
                     segment_callback=segment_callback,
                     n_threads=n_threads,
                     stop_event=stop_event,
                     queue_backlog_limiter=queue_backlog_limiter,
                     backend=backend,
                     ).process_input(TARGET_SAMPLE_RATE)

def process_mic(
        q,
        language,
        stop_event=None,
        max_queue_seconds: Optional[float] = None,
        n_threads: int = 1,
        show_name: str = "microphone",
        transcript_persistence_callback: Optional[TranscriptPersistenceCallback] = None,
        audio_segment_callback=None,
        backend: str = 'whisper_cpp',
):
    if max_queue_seconds is None:
        limiter = create_default_queue_limiter(show_name)
    elif max_queue_seconds > 0:
        limiter = QueueBacklogLimiter(max_queue_seconds, source_label=show_name)
    else:
        limiter = None
    MicRecorder(
        q,
        stop_event=stop_event,
        queue_limiter=limiter,
        n_threads=n_threads,
        show_name=show_name,
        transcript_persistence_callback=transcript_persistence_callback,
        audio_segment_callback=audio_segment_callback,
        backend=backend,
    ).record(language=language)


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
    argparse.add_argument('action', type=str, choices=['file','mic','stream','web'])
    argparse.add_argument('--file', type=str, required=False)
    argparse.add_argument('--lang', type=str, required=False)
    argparse.add_argument('--config', type=str, required=False) # for url live streaming
    argparse.add_argument('--output', type=str, required=False)
    argparse.add_argument('--n-threads', type=int, required=False, default=None, help='Number of threads for whisper model (default: 1 or from config file)')
    # https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS
    argparse.add_argument('--model', type=str, required=False, default=None, help='Whisper model name (default: large-v3-turbo or from config file)')
    argparse.add_argument('--backend', type=str, choices=['whisper_cpp', 'faster_whisper'], default='whisper_cpp', help='Transcription backend (default: whisper_cpp)')
    # Web server options
    argparse.add_argument('--host', type=str, required=False, default='0.0.0.0', help='Host to bind web server to (default: 0.0.0.0)')
    argparse.add_argument('--port', type=int, required=False, default=5002, help='Port to bind web server to (default: 5002)')
    argparse.add_argument('--dev', action='store_true', help='Enable development mode with hot reload and CORS')
    argparse.add_argument('--no-transcribe', action='store_true', help='Skip transcription: for file action, only save VAD-detected audio segments; for web server, enable view-only mode')
    argparse.add_argument('--transcribe-api-url', type=str, required=False, help='Alternate transcribe API endpoint URL (for future use)')
    args = argparse.parse_args()

    if args.action == 'file':
        if not args.file:
            print("Please provide a file path")
            exit(1)
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            exit(1)

        if not args.no_transcribe and not args.output:
            print("Please provide an --output path for the JSON transcript")
            exit(1)

        try:
            with acquire_lock('file'):
                audio_input_queue = queue.Queue()
                stop_event = threading.Event()

                if args.no_transcribe:
                    # VAD-only mode: just save audio segments without transcription
                    # Extract filename without extension for show_name
                    show_name = os.path.splitext(os.path.basename(args.file))[0]
                    thread_transcribe = threading.Thread(target=process_vad_only, kwargs={
                        'audio_input_queue': audio_input_queue,
                        'show_name': show_name,
                        'stop_event': stop_event,
                    })
                    print("VAD-only mode: saving audio segments without transcription", file=sys.stderr)
                else:
                    # Normal mode: transcribe and save to JSON
                    output_path = args.output
                    transcript_writer = JsonTranscriptWriter(output_path)

                    thread_transcribe = threading.Thread(target=process_queue, kwargs={
                        'q': audio_input_queue,
                        'language': args.lang,
                        'segment_callback': transcript_writer.add_segment,
                        'model': args.model if args.model is not None else 'large-v3-turbo',
                        'n_threads': args.n_threads if args.n_threads is not None else 1,
                        'stop_event': stop_event,
                        'mode': 'file',  # File mode: no wall_clock timestamps
                        'backend': args.backend,
                    })

                # Start the thread
                thread_transcribe.start()

                ts = 0
                interrupted = False
                chunks_read = 0
                total_bytes = 0
                try:
                    with ffmpeg_get_16bit_pcm(args.file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
                        while True:
                            if stop_event.is_set():
                                print(f"Stop event set after {chunks_read} chunks, {total_bytes} bytes", file=sys.stderr)
                                break
                            chunk = stdout.read(4096)
                            if not chunk:
                                print(f"End of stream reached after {chunks_read} chunks, {total_bytes} bytes, {ts:.2f} seconds", file=sys.stderr)
                                break
                            chunks_read += 1
                            total_bytes += len(chunk)
                            audio = pcm_s16le_to_float32(chunk)
                            # File mode: no wall_clock_start, only relative timestamps
                            audio_input_queue.put(AudioSegment(
                                audio=audio,
                                start=ts,
                                wall_clock_start=None
                            ))
                            ts += len(audio) / TARGET_SAMPLE_RATE
                            if chunks_read % 1000 == 0:
                                print(f"Progress: {chunks_read} chunks, {total_bytes} bytes, {ts:.2f} seconds", file=sys.stderr)
                except KeyboardInterrupt:
                    interrupted = True
                    print("\nCtrl+C received, stopping processing...", file=sys.stderr)
                finally:
                    # Only set stop_event if we were interrupted
                    # For normal completion, just send sentinel and wait for thread
                    if interrupted:
                        _request_shutdown(stop_event, audio_input_queue)
                    else:
                        # Normal completion: just send sentinel, don't set stop_event
                        audio_input_queue.put(None)
                    thread_transcribe.join()

                if not args.no_transcribe:
                    transcript_writer.flush()
                    if interrupted:
                        print(f"Transcript written to {output_path} (partial)")
                    else:
                        print(f"Transcript written to {output_path}")
                else:
                    print("Audio segments saved to ./tmp/speech/")
        except LockError as e:
            print(str(e), file=sys.stderr)
            exit(1)

        #AudioTranscriber().process_silero(audio)
    elif args.action == 'mic':
        try:
            with acquire_lock('mic_web'):
                audio_input_queue = queue.Queue()
                stop_event = threading.Event()
                show_name = "microphone"

                if args.no_transcribe:
                    # No-transcribe mode: VAD-only, save audio files without loading Whisper model
                    queue_limiter = create_default_queue_limiter(show_name)

                    print("Microphone mode: saving audio segments without transcription", file=sys.stderr)
                    print("Press Ctrl+C to stop microphone capture.")

                    # Start VAD processing thread
                    input_sample_rate = [TARGET_SAMPLE_RATE]  # Will be updated by capture thread
                    thread_vad = threading.Thread(
                        target=process_vad_only_livestream,
                        kwargs={
                            'audio_input_queue': audio_input_queue,
                            'show_name': show_name,
                            'input_sample_rate': TARGET_SAMPLE_RATE,  # Will use actual rate from mic
                            'stop_event': stop_event,
                            'queue_backlog_limiter': queue_limiter,
                        },
                    )
                    thread_vad.start()

                    # Capture microphone audio (blocks until stop_event)
                    try:
                        input_sample_rate[0] = capture_mic_to_queue(
                            audio_input_queue,
                            stop_event,
                            queue_limiter
                        )
                    except KeyboardInterrupt:
                        print("\nCtrl+C received, stopping microphone capture...", file=sys.stderr)
                        stop_event.set()

                    # Send sentinel and wait for VAD thread
                    audio_input_queue.put(None)
                    thread_vad.join()
                    print("Audio segments saved to ./tmp/speech/")
                else:
                    # Transcribe mode: save to database
                    with connect_to_database() as conn:
                        transcript_writer = build_database_writer(conn, show_name)
                        thread_transcribe = threading.Thread(
                            target=process_mic,
                            kwargs={
                                'q': audio_input_queue,
                                'language': args.lang,
                                'stop_event': stop_event,
                                'n_threads': args.n_threads if args.n_threads is not None else 1,
                                'show_name': show_name,
                                'transcript_persistence_callback': transcript_writer,
                                'backend': args.backend,
                            },
                        )

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
        except LockError as e:
            print(str(e), file=sys.stderr)
            exit(1)
    elif args.action == 'stream':
        if args.lang:
            raise ValueError("Language should be provided in the config file rather than as a command line argument")

        # Load config first to get show_name for lock
        with open(args.config, "rb") as f:
            data = tomllib.load(f)

        try:
            with acquire_lock('stream', show_name=data.get('show_name')):
                audio_input_queue = queue.Queue()
                stop_event = threading.Event()
                queue_limiter = create_default_queue_limiter(data.get('show_name', 'stream'))

                if args.no_transcribe:
                    # No-transcribe mode: VAD-only, save audio files without loading Whisper model
                    print("Stream mode: saving audio segments without transcription", file=sys.stderr)

                    # Start VAD processing thread
                    thread_vad = threading.Thread(
                        target=process_vad_only_livestream,
                        kwargs={
                            'audio_input_queue': audio_input_queue,
                            'show_name': data['show_name'],
                            'input_sample_rate': TARGET_SAMPLE_RATE,
                            'stop_event': stop_event,
                            'queue_backlog_limiter': queue_limiter,
                        },
                    )
                    thread_vad.start()

                    url = data['url']

                    thread_streaming = threading.Thread(
                        target=stream_url_thread,
                        kwargs={
                            'url': url,
                            'audio_input_queue': audio_input_queue,
                            'stop_event': stop_event,
                            'queue_limiter': queue_limiter,
                        },
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
                    thread_vad.join()
                    print("Audio segments saved to ./tmp/speech/")
                else:
                    # Transcribe mode: save to database
                    with connect_to_database() as conn:
                        # Initialize database schema
                        initialize_database_schema(conn)

                        db_writer = build_database_writer(conn, data['show_name'])

                        # Create a new thread
                        thread_transcribe = threading.Thread(target=process_queue,  kwargs={
                            'q': audio_input_queue,
                            'language': data['language'],
                            'save_audio': False,
                            'show_name': data['show_name'],
                            'transcript_persistence_callback': db_writer,
                            'model': args.model if args.model is not None else data.get('model', 'large-v3-turbo'),
                            'n_threads': args.n_threads if args.n_threads is not None else int(data.get('n_threads', 1)),
                            'stop_event': stop_event,
                            'queue_backlog_limiter': queue_limiter,
                            'backend': args.backend if args.backend != 'whisper_cpp' else data.get('backend', 'whisper_cpp'),
                        })

                        # Start the thread
                        thread_transcribe.start()

                        url = data['url']

                        thread_streaming = threading.Thread(
                            target=stream_url_thread,
                            kwargs={
                                'url': url,
                                'audio_input_queue': audio_input_queue,
                                'stop_event': stop_event,
                                'queue_limiter': queue_limiter,
                            },
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
        except LockError as e:
            print(str(e), file=sys.stderr)
            exit(1)
        #thread_streaming.join()
    elif args.action == 'web':
        from whisper_transcribe_py.api.server import run_server

        try:
            with acquire_lock('mic_web'):
                print(f"Starting web server on {args.host}:{args.port}")
                if args.dev:
                    print("Development mode enabled - CORS and hot reload active")
                    print("Frontend dev server: http://localhost:5173")
                if args.no_transcribe:
                    print("Transcription disabled - running in view-only mode")
                if args.transcribe_api_url:
                    print(f"Alternate transcribe API URL: {args.transcribe_api_url}")
                print(f"API server: http://{args.host}:{args.port}")

                run_server(
                    host=args.host,
                    port=args.port,
                    dev=args.dev,
                    no_transcribe=args.no_transcribe,
                    transcribe_api_url=args.transcribe_api_url,
                )
        except LockError as e:
            print(str(e), file=sys.stderr)
            exit(1)
    else:
        raise ValueError("Invalid action {}".format(args.action))
