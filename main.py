import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

import numpy as np

from whisper_transcribe_py.audio_transcriber import (
    TARGET_SAMPLE_RATE,
    ffmpeg_get_16bit_pcm,
    pcm_s16le_to_float32,
    AudioTranscriber,
    AudioSegment,
    TranscribedSegment,
    TranscriptionCallback,
)
from whisper_transcribe_py.vad_processor import SpeechDetector, get_window_size_samples
from whisper_transcribe_py.file_lock import acquire_lock, LockError


class JsonTranscriptWriter:
    """Writes transcribed segments to a JSON file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.segments = []

    def add_segments(self, segments: list[TranscribedSegment]):
        """Add transcribed segments to the output."""
        for segment in segments:
            self.segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })

    def flush(self):
        """Write accumulated segments to JSON file."""
        directory = os.path.dirname(self.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump({"segments": self.segments}, f, ensure_ascii=False, indent=2)


def save_audio_segment_wav(segment: AudioSegment, output_dir: str, index: int) -> str:
    """Save audio segment as WAV file using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    filename = f"segment_{index:04d}_{segment.start:.2f}s.wav"
    output_path = os.path.join(output_dir, filename)

    # Convert float32 to int16
    max_int16 = np.iinfo(np.int16).max
    audio_int16 = (segment.audio * max_int16).astype(np.int16)

    # Write WAV using ffmpeg
    command = [
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "pcm_s16le",
        output_path
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(input=audio_int16.tobytes())

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

    return output_path


def process_vad_only(audio_input_queue, output_dir, stop_event=None):
    """
    Process audio through VAD only, saving detected speech segments as WAV files.
    File mode - no wall clock timestamps, no backlog limiting.
    """
    def on_segment_complete(segment: AudioSegment):
        """Called when VAD detects a complete speech segment."""
        try:
            path = save_audio_segment_wav(segment, output_dir, on_segment_complete.count)
            on_segment_complete.count += 1
            print(f"Saved audio segment at {segment.start:.2f}s to {path}", file=sys.stderr)
        except Exception as e:
            print(f"Error saving audio segment: {e}", file=sys.stderr)
            raise

    on_segment_complete.count = 0

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete
    )

    window_size_samples = get_window_size_samples()
    buffer = []
    current_ts = 0.0

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        item = audio_input_queue.get()
        if item is None:
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


def process_queue(
        q,
        language,
        transcription_callback: Optional[TranscriptionCallback] = None,
        model='large-v3-turbo',
        n_threads=1,
        stop_event=None,
        backend='whisper_cpp',
):
    """Process audio queue through transcription."""
    transcriber = AudioTranscriber(
        audio_input_queue=q,
        language=language,
        model=model,
        transcription_callback=transcription_callback,
        n_threads=n_threads,
        stop_event=stop_event,
        backend=backend,
    )
    transcriber._process_input_prerecorded(TARGET_SAMPLE_RATE)


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser(
        description='Whisper file transcription tool'
    )

    # Global arguments
    parser.add_argument('--model', type=str, default=None, help='Whisper model name (default: large-v3-turbo)')
    parser.add_argument('--n-threads', type=int, default=None, help='Number of threads for whisper model (default: 1)')
    parser.add_argument('--backend', type=str, choices=['whisper_cpp', 'faster_whisper'], default='whisper_cpp', help='Transcription backend (default: whisper_cpp)')

    # Create subparsers for each action
    subparsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

    # FILE subcommand
    parser_file = subparsers.add_parser('file', help='Transcribe audio from a file')
    parser_file.add_argument('--file', type=str, required=True, help='Path to audio file')
    parser_file.add_argument('--output', type=str, help='Output path for JSON transcript (required if --transcribe)')
    parser_file.add_argument('--lang', type=str, default='en', help='Language code for transcription (default: en)')
    parser_file.add_argument('--transcribe', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable transcription (default: enabled). Use --no-transcribe to skip transcription and only save VAD-detected audio segments')

    args = parser.parse_args()

    if args.action == 'file':
        # Validate file exists
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            exit(1)

        # Conditional validation: --output required when --transcribe is enabled
        if args.transcribe and not args.output:
            print("Please provide an --output path for the JSON transcript")
            exit(1)

        try:
            with acquire_lock('file'):
                audio_input_queue = queue.Queue()
                stop_event = threading.Event()

                if not args.transcribe:
                    # VAD-only mode: just save audio segments without transcription
                    output_dir = os.path.expanduser("~/whisper_segments")
                    thread_transcribe = threading.Thread(target=process_vad_only, kwargs={
                        'audio_input_queue': audio_input_queue,
                        'output_dir': output_dir,
                        'stop_event': stop_event,
                    })
                    print(f"VAD-only mode: saving audio segments to {output_dir}", file=sys.stderr)
                else:
                    # Normal mode: transcribe and save to JSON
                    output_path = args.output
                    transcript_writer = JsonTranscriptWriter(output_path)

                    thread_transcribe = threading.Thread(target=process_queue, kwargs={
                        'q': audio_input_queue,
                        'language': args.lang,
                        'transcription_callback': transcript_writer.add_segments,
                        'model': args.model if args.model is not None else 'large-v3-turbo',
                        'n_threads': args.n_threads if args.n_threads is not None else 1,
                        'stop_event': stop_event,
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
                            audio_input_queue.put(AudioSegment(
                                audio=audio,
                                start=ts
                            ))
                            ts += len(audio) / TARGET_SAMPLE_RATE
                            if chunks_read % 1000 == 0:
                                print(f"Progress: {chunks_read} chunks, {total_bytes} bytes, {ts:.2f} seconds", file=sys.stderr)
                except KeyboardInterrupt:
                    interrupted = True
                    print("\nCtrl+C received, stopping processing...", file=sys.stderr)
                finally:
                    # Send sentinel to indicate end of audio
                    audio_input_queue.put(None)
                    thread_transcribe.join()

                if args.transcribe:
                    transcript_writer.flush()
                    if interrupted:
                        print(f"Transcript written to {output_path} (partial)")
                    else:
                        print(f"Transcript written to {output_path}")
                else:
                    print("Audio segments saved to ~/whisper_segments/")
        except LockError as e:
            print(str(e), file=sys.stderr)
            exit(1)
