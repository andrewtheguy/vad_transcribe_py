import argparse
import json
import os
import subprocess
import sys
from typing import Optional

from dotenv import load_dotenv

import numpy as np

from whisper_transcribe_py.audio_transcriber import (
    TARGET_SAMPLE_RATE,
    ffmpeg_get_16bit_pcm,
    pcm_s16le_to_float32,
    get_window_size_samples,
    create_transcriber,
    TranscribedSegment,
)
from whisper_transcribe_py.vad_processor import SpeechDetector, AudioSegment
from whisper_transcribe_py.file_lock import acquire_lock, LockError


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


def load_full_audio(audio_file: str) -> AudioSegment:
    """
    Load entire audio file as a single AudioSegment (no VAD).

    Args:
        audio_file: Path to audio file

    Returns:
        AudioSegment containing the entire file
    """
    audio_samples: list[float] = []

    with ffmpeg_get_16bit_pcm(audio_file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                break
            audio = pcm_s16le_to_float32(chunk)
            audio_samples.extend(audio)

    audio_array = np.array(audio_samples, dtype=np.float32)
    duration = len(audio_array) / TARGET_SAMPLE_RATE

    return AudioSegment(
        start=0.0,
        audio=audio_array,
        duration_seconds=duration,
    )


def run_vad(audio_file: str, save_wav_dir: Optional[str] = None) -> list[AudioSegment]:
    """
    Run VAD on audio file and return detected speech segments.

    Args:
        audio_file: Path to audio file
        save_wav_dir: If provided, save segments as WAV files to this directory

    Returns:
        List of AudioSegment objects
    """
    segments: list[AudioSegment] = []
    segment_count = 0

    def on_segment_complete(segment: AudioSegment):
        nonlocal segment_count
        if save_wav_dir:
            path = save_audio_segment_wav(segment, save_wav_dir, segment_count)
            print(f"[VAD] Segment at {segment.start:.2f}s, duration={segment.duration_seconds:.2f}s -> {path}", file=sys.stderr)
        else:
            print(f"[VAD] Segment at {segment.start:.2f}s, duration={segment.duration_seconds:.2f}s", file=sys.stderr)
        segments.append(segment)
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
                print(f"End of stream reached after {chunks_read} chunks, {total_bytes} bytes, {current_ts:.2f} seconds", file=sys.stderr)
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
                print(f"Progress: {chunks_read} chunks, {total_bytes} bytes, {current_ts:.2f} seconds", file=sys.stderr)

    speech_detector.flush()
    return segments


def transcribe_segments(
    segments: list[AudioSegment],
    language: str,
    model: str,
    backend: str,
    n_threads: int
) -> list[TranscribedSegment]:
    """Transcribe a list of audio segments."""
    # Load model once
    transcriber = create_transcriber(language, model, backend, n_threads)

    results: list[TranscribedSegment] = []
    for segment in segments:
        transcribed = transcriber.transcribe(segment.audio, segment.start)
        results.extend(transcribed)

    return results


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
    parser_file.add_argument('--vad', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable VAD (default: enabled). Use --no-vad to transcribe entire file without voice activity detection')

    args = parser.parse_args()

    if args.action == 'file':
        # Validate file exists
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist")
            sys.exit(1)

        # Conditional validation: --output required when --transcribe is enabled
        if args.transcribe and not args.output:
            print("Please provide an --output path for the JSON transcript")
            sys.exit(1)

        # --no-vad requires --transcribe (can't skip VAD without transcribing)
        if not args.vad and not args.transcribe:
            print("--no-vad requires transcription. Cannot use --no-vad with --no-transcribe")
            sys.exit(1)

        try:
            with acquire_lock('file'):
                if args.vad:
                    # 1. Run VAD (optionally save WAV files if --no-transcribe)
                    output_dir = os.path.expanduser("~/whisper_segments") if not args.transcribe else None
                    if output_dir:
                        print(f"VAD-only mode: saving audio segments to {output_dir}", file=sys.stderr)

                    segments = run_vad(args.file, save_wav_dir=output_dir)
                    print(f"Found {len(segments)} speech segments", file=sys.stderr)

                    if not args.transcribe:
                        # VAD-only mode - segments already saved as WAV
                        print("Audio segments saved to ~/whisper_segments/")
                        sys.exit(0)
                else:
                    # No VAD - load entire file as single segment
                    print("Skipping VAD, loading entire audio file...", file=sys.stderr)
                    full_audio = load_full_audio(args.file)
                    print(f"Loaded {full_audio.duration_seconds:.2f}s of audio", file=sys.stderr)
                    segments = [full_audio]

                # 2. Transcribe
                results = transcribe_segments(
                    segments,
                    args.lang,
                    args.model if args.model is not None else 'large-v3-turbo',
                    args.backend,
                    args.n_threads if args.n_threads is not None else 1,
                )

                # 3. Write JSON output
                write_json_output(results, args.output)
                print(f"Transcript written to {args.output}")

        except LockError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    else:
        print("only file action is supported", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()