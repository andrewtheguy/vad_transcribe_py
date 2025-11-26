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
    _, stderr = process.communicate(input=audio_int16.tobytes())

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

    return output_path


def stream_transcribe_with_vad(
    audio_file: str,
    transcriber: WhisperTranscriber,
) -> list[TranscribedSegment]:
    """
    Stream audio through VAD and transcribe each segment immediately.

    Args:
        audio_file: Path to audio file
        transcriber: Pre-loaded WhisperTranscriber instance

    Returns:
        List of TranscribedSegment objects
    """
    results: list[TranscribedSegment] = []
    segment_count = 0

    def on_segment_complete(segment: AudioSegment):
        nonlocal segment_count
        # Print VAD status to stderr
        print(f"[VAD] Segment {segment_count}: {segment.start:.2f}s, duration={segment.duration_seconds:.2f}s", file=sys.stderr)

        # Transcribe immediately
        transcribed = transcriber.transcribe(segment.audio, segment.start)
        results.extend(transcribed)
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
    return results


def stream_transcribe_no_vad(
    audio_file: str,
    transcriber: WhisperTranscriber,
) -> list[TranscribedSegment]:
    """
    Stream audio directly to transcriber without VAD.

    Args:
        audio_file: Path to audio file
        transcriber: Pre-loaded WhisperTranscriber instance

    Returns:
        List of TranscribedSegment objects

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

    # Transcribe full audio
    audio_array = np.array(audio_chunks, dtype=np.float32)
    print(f"Transcribing {len(audio_array) / TARGET_SAMPLE_RATE:.2f}s of audio...", file=sys.stderr)
    return transcriber.transcribe(audio_array, 0.0)


def split_by_vad(audio_file: str, output_dir: str) -> int:
    """
    Stream audio through VAD and save each segment as WAV file.

    Args:
        audio_file: Path to audio file
        output_dir: Directory to save WAV segments

    Returns:
        Number of segments saved
    """
    segment_count = 0

    def on_segment_complete(segment: AudioSegment):
        nonlocal segment_count
        path = save_audio_segment_wav(segment, output_dir, segment_count)
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


def write_json_output(results: list[TranscribedSegment], output_path: str):
    """Write transcription results to JSON file."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    segments_data = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in results
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments_data}, f, ensure_ascii=False, indent=2)


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
    parser_transcribe.add_argument('--output', type=str, required=True,
                                   help='Output path for JSON transcript')
    parser_transcribe.add_argument('--lang', type=str, default='en',
                                   help='Language code for transcription (default: en)')
    parser_transcribe.add_argument('--vad', action=argparse.BooleanOptionalAction, default=True,
                                   help='Use VAD segmentation (default: enabled). '
                                        'Use --no-vad to transcribe without VAD (max 2 hours)')

    # SPLIT subcommand
    parser_split = subparsers.add_parser('split', help='Split audio file by VAD into WAV segments')
    parser_split.add_argument('--file', type=str, required=True, help='Path to audio file')
    parser_split.add_argument('--output-dir', type=str, required=True,
                              help='Directory to save WAV segments')

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

                # Transcribe with or without VAD
                if args.vad:
                    results = stream_transcribe_with_vad(args.file, transcriber)
                else:
                    results = stream_transcribe_no_vad(args.file, transcriber)

                # Write JSON output
                write_json_output(results, args.output)
                print(f"Transcript written to {args.output}")

            elif args.action == 'split':
                segment_count = split_by_vad(args.file, args.output_dir)
                print(f"Saved {segment_count} segments to {args.output_dir}")

    except LockError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
