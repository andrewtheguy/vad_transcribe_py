import argparse
import json
import os
import subprocess
import sys
from typing import Optional

from dotenv import load_dotenv

import numpy as np
from scipy import signal

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


def format_timestamp(seconds: float, include_decimals=True) -> str:
    if include_decimals:
        milliseconds = round(seconds * 1000)
        minutes_remaining, remaining_milliseconds = divmod(milliseconds, 60000)
        hours, minutes = divmod(minutes_remaining, 60)
        remaining_seconds = f"{remaining_milliseconds:05d}"
        remaining_seconds = remaining_seconds[:-3] + '.' + remaining_seconds[-3:]
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds}"
    else:
        seconds = round(seconds)
        minutes_remaining, remaining_seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes_remaining, 60)
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


# Maximum duration for --no-vad mode (2 hours) to prevent OOM
MAX_NO_VAD_DURATION_SECONDS = 2 * 60 * 60


def is_url(path: str) -> bool:
    """Check if the path is a URL."""
    return path.startswith(('http://', 'https://', 'rtmp://', 'rtsp://'))


def get_audio_duration(audio_source: str) -> float | None:
    """
    Get audio duration in seconds using ffprobe.

    Returns None if duration cannot be determined (e.g., live stream).
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_source],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    duration_str = result.stdout.strip()
    if not duration_str or duration_str == "N/A":
        return None
    return float(duration_str)


def get_audio_properties(audio_source: str) -> dict:
    """
    Get audio properties (sample rate and channels) using ffprobe.

    Returns:
        dict with 'sample_rate' (int) and 'channels' (int)
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "json", audio_source],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError("No audio stream found in source")

    stream = data["streams"][0]
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
    }


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resample mono audio to 16kHz for VAD processing.

    Args:
        audio: Float32 mono audio array
        orig_sr: Original sample rate

    Returns:
        Float32 mono audio at 16kHz
    """
    # If already 16kHz, return as-is
    if orig_sr == TARGET_SAMPLE_RATE:
        return audio.astype(np.float32)

    # Resample to 16kHz using scipy
    num_output_samples = int(len(audio) * TARGET_SAMPLE_RATE / orig_sr)
    resampled = signal.resample(audio, num_output_samples)
    return resampled.astype(np.float32)


def validate_audio_source(audio_source: str) -> float:
    """
    Validate audio source (file or URL) and return duration.

    Raises:
        ValueError: If source is invalid or is a live stream
    """
    if not is_url(audio_source) and not os.path.exists(audio_source):
        raise ValueError(f"File does not exist: {audio_source}")

    print(f"Checking audio source: {audio_source}", file=sys.stderr)
    duration = get_audio_duration(audio_source)

    if duration is None:
        raise ValueError(
            "Cannot determine audio duration. Live streams are not supported. "
            "Please provide a file or URL with fixed duration."
        )

    print(f"Audio duration: {duration:.2f}s", file=sys.stderr)
    return duration


def save_audio_segment(
    segment: AudioSegment,
    output_dir: str,
    index: int,
    original_audio: Optional[np.ndarray] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = 1,
    output_format: str = "opus",
) -> str:
    """
    Save audio segment to file using ffmpeg.

    If original_audio is provided, encode from that (preserving quality).
    Otherwise fall back to segment.audio (16kHz VAD audio).

    Args:
        segment: AudioSegment with timing info
        output_dir: Directory to save the file
        index: Segment index for filename
        original_audio: Optional int16 array at original sample rate/channels
        sample_rate: Sample rate of original_audio (default: 16kHz)
        channels: Number of channels in original_audio (default: 1)
        output_format: Output format - "opus" (16kbps) or "wav" (default: opus)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with start and end in milliseconds
    start_ms = int(segment.start * 1000)
    end_ms = int((segment.start + segment.duration_seconds) * 1000)
    filename = f"segment_{index:04d}_{start_ms}ms_{end_ms}ms.{output_format}"
    output_path = os.path.join(output_dir, filename)

    # Determine audio source and format
    if original_audio is not None:
        # Use original quality audio (already int16)
        audio_to_encode = original_audio
        ar = sample_rate
        ac = channels
    else:
        # Fall back to VAD audio (float32 -> int16)
        max_int16 = np.iinfo(np.int16).max
        audio_to_encode = (segment.audio * max_int16).astype(np.int16)
        ar = TARGET_SAMPLE_RATE
        ac = 1

    # Build ffmpeg command based on output format
    command = [
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(ar),
        "-ac", str(ac),
        "-i", "pipe:0",
    ]

    if output_format == "opus":
        # Opus 16kbps with voip optimization for speech
        command.extend([
            "-c:a", "libopus",
            "-b:a", "16k",
            "-application", "voip",
        ])
    elif output_format == "wav":
        # WAV preserves exact sample rate (useful for testing)
        command.extend([
            "-c:a", "pcm_s16le",
        ])
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    command.extend(["-loglevel", "error", output_path])

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    _, stderr = process.communicate(input=audio_to_encode.tobytes())

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

    return output_path


def write_jsonl_segment(segment: TranscribedSegment, output_file):
    """Write a single transcription segment as JSONL to the output file."""
    line = json.dumps({
        "type": "transcription",
        "start": segment.start,
        "start_formatted": format_timestamp(segment.start),
        "text": segment.text,
        "end": segment.end,
        "end_formatted": format_timestamp(segment.end),
    }, ensure_ascii=False)
    output_file.write(line + "\n")
    output_file.flush()


def write_jsonl_boundary(event: str, timestamp: float, output_file):
    """Write a segment boundary event as JSONL.

    Args:
        event: "segment_start" or "segment_end"
        timestamp: The timestamp of the boundary in seconds
        output_file: File object to write to
    """
    line = json.dumps({
        "type": event,
        "timestamp": timestamp,
        "timestamp_formatted": format_timestamp(timestamp),
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
        # Print VAD status to stderr with formatted timestamp
        start_fmt = format_timestamp(segment.start)
        print(f"[VAD] Segment {segment_count}: {start_fmt} ({segment.start:.2f}s), duration={segment.duration_seconds:.2f}s", file=sys.stderr)

        # Emit segment_start boundary
        write_jsonl_boundary("segment_start", segment.start, output_file)

        # Transcribe immediately and output to JSONL
        transcribed = transcriber.transcribe(segment.audio, segment.start)
        for ts in transcribed:
            write_jsonl_segment(ts, output_file)

        # Emit segment_end boundary
        segment_end_time = segment.start + segment.duration_seconds
        write_jsonl_boundary("segment_end", segment_end_time, output_file)

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


def stream_transcribe_stdin_with_vad(
    transcriber: WhisperTranscriber,
) -> int:
    """
    Stream WAV audio from stdin through VAD and transcribe each segment immediately.
    Output is always JSONL to stdout.

    Args:
        transcriber: Pre-loaded WhisperTranscriber instance

    Returns:
        Number of segments transcribed
    """
    segment_count = 0
    output_file = sys.stdout

    def on_segment_complete(segment: AudioSegment):
        nonlocal segment_count
        # Print VAD status to stderr with formatted timestamp
        start_fmt = format_timestamp(segment.start)
        print(f"[VAD] Segment {segment_count}: {start_fmt} ({segment.start:.2f}s), duration={segment.duration_seconds:.2f}s", file=sys.stderr)

        # Emit segment_start boundary
        write_jsonl_boundary("segment_start", segment.start, output_file)

        # Transcribe immediately and output to JSONL
        transcribed = transcriber.transcribe(segment.audio, segment.start)
        for ts in transcribed:
            write_jsonl_segment(ts, output_file)

        # Emit segment_end boundary
        segment_end_time = segment.start + segment.duration_seconds
        write_jsonl_boundary("segment_end", segment_end_time, output_file)

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

    print("Reading WAV audio from stdin...", file=sys.stderr)
    with ffmpeg_get_16bit_pcm(from_stdin=True, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
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
    audio_source: str,
    transcriber: WhisperTranscriber,
    output_file,
    duration: float,
) -> int:
    """
    Stream audio directly to transcriber without VAD.

    Args:
        audio_source: Path to audio file or URL
        transcriber: Pre-loaded WhisperTranscriber instance
        output_file: File object to write JSONL output
        duration: Pre-validated audio duration in seconds

    Returns:
        Number of segments transcribed

    Raises:
        ValueError: If audio duration exceeds 2-hour limit
    """
    # Check duration limit
    if duration > MAX_NO_VAD_DURATION_SECONDS:
        raise ValueError(
            f"Audio duration {duration/3600:.1f}h exceeds 2-hour limit for --no-vad mode. "
            "Use --vad for longer files."
        )

    print(f"Streaming audio ({duration:.2f}s) without VAD...", file=sys.stderr)

    # Stream and accumulate audio
    audio_chunks: list[float] = []
    chunks_read = 0

    with ffmpeg_get_16bit_pcm(audio_source, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
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


def split_by_vad(
    audio_source: str,
    preserve_sample_rate: bool = False,
    output_format: str = "opus",
) -> int:
    """
    Stream audio through VAD and save each segment to file.

    Args:
        audio_source: Path to audio file or URL
        preserve_sample_rate: If True, preserve original sample rate (mono).
                              If False (default), downsample to 16kHz mono.
        output_format: Output format - "opus" (16kbps) or "wav" (default: opus)

    Returns:
        Number of segments saved
    """
    # Create output directory: tmp/(filename without extension)/
    base_name = os.path.splitext(os.path.basename(audio_source))[0]
    output_dir = os.path.join("tmp", base_name)

    segment_count = 0

    if preserve_sample_rate:
        # Get original sample rate for preservation mode
        props = get_audio_properties(audio_source)
        orig_sr = props["sample_rate"]
        print(f"Audio properties: {orig_sr}Hz (preserving sample rate)", file=sys.stderr)

        # Rolling buffer for original audio (int16 mono)
        original_buffer: list[int] = []
        buffer_start_time = 0.0
        look_back_seconds = 0.5

        def on_segment_complete(segment: AudioSegment):
            nonlocal segment_count, original_buffer, buffer_start_time

            # Calculate sample range in original buffer (mono)
            seg_start_in_buffer = segment.start - buffer_start_time
            start_sample = int(seg_start_in_buffer * orig_sr)
            end_sample = int((seg_start_in_buffer + segment.duration_seconds) * orig_sr)

            # Clamp to buffer bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(original_buffer), end_sample)

            # Extract original audio for this segment
            original_audio = np.array(original_buffer[start_sample:end_sample], dtype=np.int16)

            # Save with original sample rate (mono)
            path = save_audio_segment(
                segment, output_dir, segment_count,
                original_audio=original_audio,
                sample_rate=orig_sr,
                channels=1,
                output_format=output_format,
            )
            print(f"[VAD] Saved: {path} (duration={segment.duration_seconds:.2f}s)", file=sys.stderr)

            # Trim buffer - keep only look-back audio for next segment
            look_back_samples = int(look_back_seconds * orig_sr)
            trim_to = max(0, end_sample - look_back_samples)
            if trim_to > 0:
                original_buffer = original_buffer[trim_to:]
                buffer_start_time += trim_to / orig_sr

            segment_count += 1

        speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            on_segment_complete=on_segment_complete
        )

        window_size = get_window_size_samples()
        vad_buffer: list[float] = []
        current_ts = 0.0
        chunks_read = 0

        # Stream at original sample rate, convert to mono
        with ffmpeg_get_16bit_pcm(audio_source, ac=1) as stdout:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    print(f"End of stream: {chunks_read} chunks, {current_ts:.2f}s", file=sys.stderr)
                    break
                chunks_read += 1

                # Store original audio (int16) for later extraction
                chunk_int16 = np.frombuffer(chunk, dtype=np.int16)
                original_buffer.extend(chunk_int16.tolist())

                # Convert to float32 and resample to 16kHz for VAD
                chunk_float = pcm_s16le_to_float32(chunk)
                resampled = resample_to_16k(chunk_float, orig_sr)
                vad_buffer.extend(resampled.tolist())

                # Process VAD windows
                while len(vad_buffer) >= window_size:
                    window = np.array(vad_buffer[:window_size], dtype=np.float32)
                    speech_detector.process_window(window, current_ts)
                    vad_buffer = vad_buffer[window_size:]
                    current_ts += window_size / TARGET_SAMPLE_RATE

                if chunks_read % 1000 == 0:
                    print(f"Progress: {current_ts:.2f}s", file=sys.stderr)

        speech_detector.flush()

    else:
        # Default mode: downsample to 16kHz mono (simpler, less memory)
        def on_segment_complete(segment: AudioSegment):
            nonlocal segment_count
            path = save_audio_segment(segment, output_dir, segment_count, output_format=output_format)
            print(f"[VAD] Saved: {path} (duration={segment.duration_seconds:.2f}s)", file=sys.stderr)
            segment_count += 1

        speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            on_segment_complete=on_segment_complete
        )

        window_size = get_window_size_samples()
        buffer: list[float] = []
        current_ts = 0.0
        chunks_read = 0

        # Stream at 16kHz mono
        with ffmpeg_get_16bit_pcm(audio_source, target_sample_rate=TARGET_SAMPLE_RATE, ac=1) as stdout:
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

    # Create subparsers for each action
    subparsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

    # TRANSCRIBE subcommand
    parser_transcribe = subparsers.add_parser('transcribe', help='Transcribe audio from a file or URL')
    transcribe_input = parser_transcribe.add_mutually_exclusive_group(required=True)
    transcribe_input.add_argument('--file', type=str, help='Path to audio file')
    transcribe_input.add_argument('--url', type=str, help='URL to audio (live streams not supported)')
    transcribe_input.add_argument('--stdin', action='store_true', help='Read WAV audio from stdin (always uses VAD, outputs to stdout)')
    parser_transcribe.add_argument('--output', type=str, default=None,
                                   help='Output path for JSONL transcript (default: stdout)')
    parser_transcribe.add_argument('--lang', type=str, default='en',
                                   help='Language code for transcription (default: en)')
    parser_transcribe.add_argument('--model', type=str, default='large-v3-turbo',
                                   help='Whisper model name (default: large-v3-turbo)')
    parser_transcribe.add_argument('--n-threads', type=int, default=1,
                                   help='Number of threads for whisper model (default: 1)')
    parser_transcribe.add_argument('--backend', type=str, choices=['whisper_cpp', 'faster_whisper'],
                                   default='whisper_cpp', help='Transcription backend (default: whisper_cpp)')
    parser_transcribe.add_argument('--vad', action=argparse.BooleanOptionalAction, default=True,
                                   help='Use VAD segmentation (default: enabled). '
                                        'Use --no-vad to transcribe without VAD (max 2 hours)')
    parser_transcribe.add_argument('--chinese-conversion', type=str,
                                   choices=['none', 'simplified', 'traditional'],
                                   default='none',
                                   help='Chinese character conversion for zh/yue languages: '
                                        'none (default), simplified (zh-Hans), traditional (zh-Hant)')

    # SPLIT subcommand
    parser_split = subparsers.add_parser('split', help='Split audio by VAD into Opus segments')
    split_input = parser_split.add_mutually_exclusive_group(required=True)
    split_input.add_argument('--file', type=str, help='Path to audio file')
    split_input.add_argument('--url', type=str, help='URL to audio (live streams not supported)')
    parser_split.add_argument('--preserve-sample-rate', action='store_true',
                              help='Preserve original sample rate (default: downsample to 16kHz)')
    parser_split.add_argument('--format', type=str, choices=['opus', 'wav'], default='opus',
                              help='Output format: opus (16kbps, default) or wav')

    args = parser.parse_args()

    try:
        with acquire_lock(args.action):
            if args.action == 'transcribe':
                # Handle stdin mode separately (no validation, always VAD, always stdout)
                if getattr(args, 'stdin', False):
                    print(f"Loading {args.model} model...", file=sys.stderr)
                    transcriber = create_transcriber(
                        args.lang, args.model, args.backend, args.n_threads,
                        args.chinese_conversion
                    )
                    segment_count = stream_transcribe_stdin_with_vad(transcriber)
                    print(f"Transcribed {segment_count} segments from stdin", file=sys.stderr)
                else:
                    # File or URL mode
                    audio_source = args.file if args.file else args.url
                    duration = validate_audio_source(audio_source)

                    print(f"Loading {args.model} model...", file=sys.stderr)
                    transcriber = create_transcriber(
                        args.lang, args.model, args.backend, args.n_threads,
                        args.chinese_conversion
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
                            segment_count = stream_transcribe_with_vad(audio_source, transcriber, output_file)
                        else:
                            segment_count = stream_transcribe_no_vad(audio_source, transcriber, output_file, duration)

                        if args.output:
                            print(f"Transcript written to {args.output} ({segment_count} segments)", file=sys.stderr)
                    finally:
                        if args.output:
                            output_file.close()

            elif args.action == 'split':
                audio_source = args.file if args.file else args.url
                duration = validate_audio_source(audio_source)
                segment_count = split_by_vad(audio_source, args.preserve_sample_rate, args.format)
                base_name = os.path.splitext(os.path.basename(audio_source))[0]
                output_dir = os.path.join("tmp", base_name)
                print(f"Saved {segment_count} segments to {output_dir}", file=sys.stderr)

    except LockError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
