import argparse
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

import numpy as np

from whisper_transcribe_py.audio_transcriber import (
    TARGET_SAMPLE_RATE,
    ffmpeg_get_16bit_pcm,
    pcm_s16le_to_float32,
    get_window_size_samples,
    create_transcriber,
    TranscribedSegment,
    WhisperTranscriber,
)
from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment
from whisper_transcribe_py.file_lock import acquire_lock, LockError


# Maximum duration for --no-vad mode (2 hours) to prevent OOM
MAX_NO_VAD_DURATION_SECONDS = 2 * 60 * 60


def get_audio_duration(audio_file: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def save_audio_segment_opus(segment: AudioSegment, output_dir: str, index: int) -> str:
    """Save audio segment as Opus file (16kbps mono) using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with start and end in milliseconds
    start_ms = int(segment.start * 1000)
    end_ms = int((segment.start + segment.duration_seconds) * 1000)
    filename = f"segment_{index:04d}_{start_ms}ms_{end_ms}ms.opus"
    output_path = os.path.join(output_dir, filename)

    # Convert float32 to int16
    max_int16 = np.iinfo(np.int16).max
    audio_int16 = (segment.audio * max_int16).astype(np.int16)

    # Encode to Opus 16kbps mono using ffmpeg
    # Using -application voip optimizes for speech (better quality at low bitrates)
    command = [
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", "libopus",
        "-b:a", "16k",
        "-ac", "1",
        "-application", "voip",
        "-loglevel", "error",
        output_path
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    _, stderr = process.communicate(input=audio_int16.tobytes())

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

    return output_path


def write_jsonl_segment(segment: TranscribedSegment, output_file):
    """Write a single segment as JSONL to the output file."""
    line = json.dumps({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }, ensure_ascii=False)
    output_file.write(line + "\n")
    output_file.flush()


def stream_transcribe_with_vad(
    audio_file: str,
    transcriber: WhisperTranscriber,
    output_file,
) -> int:
    """
    Stream audio through VAD and transcribe each segment immediately.

    Args:
        audio_file: Path to audio file
        transcriber: Pre-loaded WhisperTranscriber instance
        output_file: File object to write JSONL output

    Returns:
        Number of segments transcribed
    """
    segment_count = 0

    def on_segment_complete(segment: AudioSegment):
        nonlocal segment_count
        # Print VAD status to stderr
        print(f"[VAD] Segment {segment_count}: {segment.start:.2f}s, duration={segment.duration_seconds:.2f}s", file=sys.stderr)

        # Transcribe immediately and output to JSONL
        transcribed = transcriber.transcribe(segment.audio, segment.start)
        for ts in transcribed:
            write_jsonl_segment(ts, output_file)
        segment_count += 1

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete
    )

    window_size = get_window_size_samples()
    buffer = []
    current_ts = 0.0
    chunks_read = 0
    total_bytes = 0

    with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                print(f"End of stream: {chunks_read} chunks, {total_bytes} bytes, {current_ts:.2f}s", file=sys.stderr)
                break
            chunks_read += 1
            total_bytes += len(chunk)
            audio = pcm_s16le_to_float32(chunk)
            buffer.extend(audio)

            while len(buffer) >= window_size:
                window = np.array(buffer[:window_size], dtype=np.float32)
                speech_detector.process_window(window, current_ts)
                buffer = buffer[window_size:]
                current_ts += window_size / TARGET_SAMPLE_RATE

            if chunks_read % 1000 == 0:
                print(f"Progress: {chunks_read} chunks, {current_ts:.2f}s", file=sys.stderr)

    speech_detector.flush()
    print(f"Found {segment_count} speech segments", file=sys.stderr)
    return segment_count


def stream_transcribe_no_vad(
    audio_file: str,
    transcriber: WhisperTranscriber,
    output_file,
) -> int:
    """
    Stream audio directly to transcriber without VAD.

    Args:
        audio_file: Path to audio file
        transcriber: Pre-loaded WhisperTranscriber instance
        output_file: File object to write JSONL output

    Returns:
        Number of segments transcribed

    Raises:
        ValueError: If audio duration exceeds 2-hour limit
    """
    # Check duration limit first
    duration = get_audio_duration(audio_file)
    if duration > MAX_NO_VAD_DURATION_SECONDS:
        raise ValueError(
            f"Audio duration {duration/3600:.1f}h exceeds 2-hour limit for --no-vad mode. "
            "Use --vad for longer files."
        )

    print(f"Streaming audio ({duration:.2f}s) without VAD...", file=sys.stderr)

    # Stream and accumulate audio
    audio_chunks: list[float] = []
    chunks_read = 0

    with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                break
            chunks_read += 1
            audio = pcm_s16le_to_float32(chunk)
            audio_chunks.extend(audio)

            if chunks_read % 1000 == 0:
                current_duration = len(audio_chunks) / TARGET_SAMPLE_RATE
                print(f"Progress: {current_duration:.2f}s", file=sys.stderr)

    # Transcribe full audio and output as JSONL
    audio_array = np.array(audio_chunks, dtype=np.float32)
    print(f"Transcribing {len(audio_array) / TARGET_SAMPLE_RATE:.2f}s of audio...", file=sys.stderr)
    results = transcriber.transcribe(audio_array, 0.0)
    for segment in results:
        write_jsonl_segment(segment, output_file)
    return len(results)


def split_by_vad(audio_file: str) -> int:
    """
    Stream audio through VAD and save each segment as Opus file.

    Args:
        audio_file: Path to audio file

    Returns:
        Number of segments saved
    """
    # Create output directory: tmp/(filename without extension)/
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = os.path.join("tmp", base_name)

    segment_count = 0

    def on_segment_complete(segment: AudioSegment):
        nonlocal segment_count
        path = save_audio_segment_opus(segment, output_dir, segment_count)
        print(f"[VAD] Saved: {path} (duration={segment.duration_seconds:.2f}s)", file=sys.stderr)
        segment_count += 1

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete
    )

    window_size = get_window_size_samples()
    buffer = []
    current_ts = 0.0
    chunks_read = 0

    with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                print(f"End of stream: {chunks_read} chunks, {current_ts:.2f}s", file=sys.stderr)
                break
            chunks_read += 1
            audio = pcm_s16le_to_float32(chunk)
            buffer.extend(audio)

            while len(buffer) >= window_size:
                window = np.array(buffer[:window_size], dtype=np.float32)
                speech_detector.process_window(window, current_ts)
                buffer = buffer[window_size:]
                current_ts += window_size / TARGET_SAMPLE_RATE

            if chunks_read % 1000 == 0:
                print(f"Progress: {current_ts:.2f}s", file=sys.stderr)

    speech_detector.flush()
    return segment_count


def main():
    """Main entry point for the CLI."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        description='Whisper transcription tool with streaming audio processing'
    )

    # Global arguments
    parser.add_argument('--model', type=str, default='large-v3-turbo',
                        help='Whisper model name (default: large-v3-turbo)')
    parser.add_argument('--n-threads', type=int, default=1,
                        help='Number of threads for whisper model (default: 1)')
    parser.add_argument('--backend', type=str, choices=['whisper_cpp', 'faster_whisper'],
                        default='whisper_cpp', help='Transcription backend (default: whisper_cpp)')

    # Create subparsers for each action
    subparsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

    # TRANSCRIBE subcommand
    parser_transcribe = subparsers.add_parser('transcribe', help='Transcribe audio from a file')
    parser_transcribe.add_argument('--file', type=str, required=True, help='Path to audio file')
    parser_transcribe.add_argument('--output', type=str, default=None,
                                   help='Output path for JSONL transcript (default: stdout)')
    parser_transcribe.add_argument('--lang', type=str, default='en',
                                   help='Language code for transcription (default: en)')
    parser_transcribe.add_argument('--vad', action=argparse.BooleanOptionalAction, default=True,
                                   help='Use VAD segmentation (default: enabled). '
                                        'Use --no-vad to transcribe without VAD (max 2 hours)')

    # SPLIT subcommand
    parser_split = subparsers.add_parser('split', help='Split audio file by VAD into Opus segments')
    parser_split.add_argument('--file', type=str, required=True, help='Path to audio file')

    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.file):
        print(f"File {args.file} does not exist", file=sys.stderr)
        sys.exit(1)

    try:
        with acquire_lock(args.action):
            if args.action == 'transcribe':
                # Load transcriber once
                print(f"Loading {args.model} model...", file=sys.stderr)
                transcriber = create_transcriber(
                    args.lang, args.model, args.backend, args.n_threads
                )

                # Determine output destination
                if args.output:
                    directory = os.path.dirname(args.output)
                    if directory:
                        os.makedirs(directory, exist_ok=True)
                    output_file = open(args.output, "w", encoding="utf-8")
                else:
                    output_file = sys.stdout

                try:
                    # Transcribe with or without VAD, streaming JSONL output
                    if args.vad:
                        segment_count = stream_transcribe_with_vad(args.file, transcriber, output_file)
                    else:
                        segment_count = stream_transcribe_no_vad(args.file, transcriber, output_file)

                    if args.output:
                        print(f"Transcript written to {args.output} ({segment_count} segments)", file=sys.stderr)
                finally:
                    if args.output:
                        output_file.close()

            elif args.action == 'split':
                segment_count = split_by_vad(args.file)
                base_name = os.path.splitext(os.path.basename(args.file))[0]
                output_dir = os.path.join("tmp", base_name)
                print(f"Saved {segment_count} segments to {output_dir}", file=sys.stderr)

    except LockError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
