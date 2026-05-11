import argparse
import json
import logging
import os
import subprocess
import sys
import uuid
from typing import IO

from dotenv import load_dotenv

import numpy as np
from scipy import signal

from vad_transcribe_py.audio_transcriber import (
    TARGET_SAMPLE_RATE,
    ffmpeg_stream_float32,
    stream_stdin_wav,
    get_window_size_samples,
    create_transcriber,
    TranscribedSegment,
    AudioTranscriber,
)
from vad_transcribe_py._utils import clip_repetitive_text
from vad_transcribe_py.vad_processor import (
    SpeechDetector,
    AudioSegment,
    DEFAULT_HARD_LIMIT_SECONDS,
    DEFAULT_MIN_SPEECH_SECONDS,
    DEFAULT_SOFT_LIMIT_SECONDS,
    DEFAULT_SPEECH_THRESHOLD,
    DEFAULT_MIN_SILENCE_DURATION_MS,
    DEFAULT_LOOK_BACK_SECONDS,
)
from vad_transcribe_py.file_lock import acquire_lock, LockError

logger = logging.getLogger(__name__)


def add_vad_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common VAD arguments to a parser."""
    parser.add_argument('--min-speech-seconds', type=float, default=DEFAULT_MIN_SPEECH_SECONDS,
                        help=f'Minimum speech duration in seconds (default: {DEFAULT_MIN_SPEECH_SECONDS})')
    parser.add_argument('--soft-limit-seconds', type=float, default=DEFAULT_SOFT_LIMIT_SECONDS,
                        help=f'Soft limit on speech segment duration in seconds (default: {DEFAULT_SOFT_LIMIT_SECONDS})')
    parser.add_argument('--speech-threshold', type=float, default=DEFAULT_SPEECH_THRESHOLD,
                        help=f'VAD speech detection threshold 0.0-1.0 (default: {DEFAULT_SPEECH_THRESHOLD})')
    parser.add_argument('--min-silence-duration-ms', type=int, default=DEFAULT_MIN_SILENCE_DURATION_MS,
                        help=f'Minimum silence duration in ms to end segment (default: {DEFAULT_MIN_SILENCE_DURATION_MS})')
    parser.add_argument('--look-back-seconds', type=float, default=DEFAULT_LOOK_BACK_SECONDS,
                        help=f'Look-back buffer in seconds for segment start (default: {DEFAULT_LOOK_BACK_SECONDS})')


def get_vad_params(args: argparse.Namespace) -> dict[str, float | int]:
    """Extract VAD parameters from parsed arguments."""
    return {
        "min_speech_seconds": args.min_speech_seconds,
        "soft_limit_seconds": args.soft_limit_seconds,
        "speech_threshold": args.speech_threshold,
        "min_silence_duration_ms": args.min_silence_duration_ms,
        "look_back_seconds": args.look_back_seconds,
    }


def format_timestamp(seconds: float, include_decimals: bool = True) -> str:
    if include_decimals:
        milliseconds = round(seconds * 1000)
        minutes_remaining, remaining_milliseconds = divmod(milliseconds, 60000)
        hours, minutes = divmod(minutes_remaining, 60)
        remaining_seconds = f"{remaining_milliseconds:05d}"
        remaining_seconds = remaining_seconds[:-3] + '.' + remaining_seconds[-3:]
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds}"
    else:
        rounded_seconds = round(seconds)
        minutes_remaining, remaining_seconds = divmod(rounded_seconds, 60)
        hours, minutes = divmod(minutes_remaining, 60)
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"



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


def get_audio_properties(audio_source: str) -> dict[str, int]:
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
    resampled = np.asarray(signal.resample(audio, num_output_samples))
    return resampled.astype(np.float32)


def validate_audio_source(audio_source: str) -> None:
    """
    Validate audio source (file or URL).

    Raises:
        ValueError: If source is invalid or is a live stream
    """
    if not is_url(audio_source) and not os.path.exists(audio_source):
        raise ValueError(f"File does not exist: {audio_source}")

    logger.info("Checking audio source: %s", audio_source)
    duration = get_audio_duration(audio_source)

    if duration is None:
        raise ValueError(
            "Cannot determine audio duration. Live streams are not supported. "
            "Please provide a file or URL with fixed duration."
        )

    logger.info("Audio duration: %.2fs", duration)


def save_audio_segment(
    segment: AudioSegment,
    output_dir: str,
    index: int,
    original_audio: np.ndarray | None = None,
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
        original_audio: Optional float32 array at original sample rate/channels
        sample_rate: Sample rate of original_audio (default: 16kHz)
        channels: Number of channels in original_audio (default: 1)
        output_format: Output format - "opus" (16kbps) or "wav" (default: opus)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with start and end in milliseconds
    assert segment.duration_seconds is not None
    start_ms = int(segment.start * 1000)
    end_ms = int((segment.start + segment.duration_seconds) * 1000)
    filename = f"segment_{index:04d}_{start_ms}ms_{end_ms}ms.{output_format}"
    output_path = os.path.join(output_dir, filename)

    # Determine audio source and format
    if original_audio is not None:
        # Use original quality audio (float32)
        audio_to_encode = original_audio
        ar = sample_rate
        ac = channels
    else:
        # Fall back to VAD audio (float32)
        audio_to_encode = segment.audio
        ar = TARGET_SAMPLE_RATE
        ac = 1

    # Build ffmpeg command based on output format
    command = [
        "ffmpeg", "-y",
        "-f", "f32le",
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


def write_jsonl_segment(
    segment: TranscribedSegment,
    output_file: IO[str],
    clip_repetitions: bool = False,
) -> None:
    """Write a single transcription segment as JSONL to the output file.

    When ``clip_repetitions`` is True, the tail of a heavily-repetitive transcript
    line is truncated after the first copy of the repeating pattern and replaced
    with an ``(indistinguishable speech)`` placeholder. The meaningful prefix is
    preserved.
    """
    text = segment.text
    repetition_clipped = False
    if clip_repetitions:
        text = clip_repetitive_text(segment.text)
        repetition_clipped = text != segment.text

    line = json.dumps({
        "type": "transcript",
        "id": str(uuid.uuid7()),
        "start_ms": round(segment.start * 1000),
        "start_formatted": format_timestamp(segment.start),
        "text": text,
        "end_ms": round(segment.end * 1000),
        "end_formatted": format_timestamp(segment.end),
        "prompt_retry": segment.prompt_retry,
        "repetition_clipped": repetition_clipped,
    }, ensure_ascii=False)
    output_file.write(line + "\n")
    output_file.flush()


def write_jsonl_marker(event: str, output_file: IO[str]) -> None:
    """Write a stream marker (stream_start or stream_end) as JSONL."""
    line = json.dumps({"type": event}, ensure_ascii=False)
    output_file.write(line + "\n")
    output_file.flush()


def write_jsonl_boundary(event: str, timestamp: float, output_file: IO[str]) -> None:
    """Write a segment boundary event as JSONL.

    Args:
        event: "segment_start" or "segment_end"
        timestamp: The timestamp of the boundary in seconds
        output_file: File object to write to
    """
    line = json.dumps({
        "type": event,
        "timestamp_ms": round(timestamp * 1000),
        "timestamp_formatted": format_timestamp(timestamp),
    }, ensure_ascii=False)
    output_file.write(line + "\n")
    output_file.flush()


def stream_transcribe_with_vad(
    audio_file: str,
    transcriber: AudioTranscriber,
    output_file: IO[str],
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
    soft_limit_seconds: float | None = DEFAULT_SOFT_LIMIT_SECONDS,
    speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    look_back_seconds: float = DEFAULT_LOOK_BACK_SECONDS,
    hard_limit_seconds: float | None = None,
    clip_repetitions: bool = False,
) -> int:
    """
    Stream audio through VAD and transcribe each segment immediately.

    Args:
        audio_file: Path to audio file
        transcriber: Pre-loaded AudioTranscriber instance
        output_file: File object to write JSONL output
        min_speech_seconds: Minimum speech duration in seconds
        soft_limit_seconds: Soft limit on speech segment duration in seconds
        speech_threshold: VAD speech detection threshold (0.0-1.0)
        min_silence_duration_ms: Minimum silence duration in ms to end segment
        look_back_seconds: Look-back buffer in seconds for segment start
        hard_limit_seconds: Hard limit on segment duration (from transcriber)

    Returns:
        Number of segments transcribed
    """
    write_jsonl_marker("stream_start", output_file)
    segment_count = 0

    def on_segment_complete(segment: AudioSegment) -> None:
        nonlocal segment_count
        assert segment.duration_seconds is not None
        start_fmt = format_timestamp(segment.start)
        logger.info("[VAD] Segment %d: %s (%.2fs), duration=%.2fs", segment_count, start_fmt, segment.start, segment.duration_seconds)

        write_jsonl_boundary("segment_start", segment.start, output_file)

        transcribed = transcriber.transcribe(segment.audio, segment.start)
        for ts in transcribed:
            write_jsonl_segment(ts, output_file, clip_repetitions=clip_repetitions)

        segment_end_time = segment.start + segment.duration_seconds
        write_jsonl_boundary("segment_end", segment_end_time, output_file)

        segment_count += 1

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete,
        min_speech_seconds=min_speech_seconds,
        soft_limit_seconds=soft_limit_seconds,
        speech_threshold=speech_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        look_back_seconds=look_back_seconds,
        hard_limit_seconds=hard_limit_seconds if hard_limit_seconds is not None else DEFAULT_HARD_LIMIT_SECONDS,
    )

    window_size = get_window_size_samples()
    buffer: list[float] = []
    current_ts = 0.0
    chunks_read = 0

    for audio in ffmpeg_stream_float32(audio_file, target_sample_rate=TARGET_SAMPLE_RATE, ac=1):
        chunks_read += 1
        buffer.extend(audio)

        while len(buffer) >= window_size:
            window = np.array(buffer[:window_size], dtype=np.float32)
            speech_detector.process_window(window, current_ts)
            buffer = buffer[window_size:]
            current_ts += window_size / TARGET_SAMPLE_RATE

        if chunks_read % 1000 == 0:
            logger.info("Progress: %d chunks, %.2fs", chunks_read, current_ts)

    logger.info("End of stream: %d chunks, %.2fs", chunks_read, current_ts)
    speech_detector.flush()
    write_jsonl_marker("stream_end", output_file)
    logger.info("Found %d speech segments", segment_count)
    return segment_count


def stream_transcribe_stdin_with_vad(
    transcriber: AudioTranscriber,
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
    soft_limit_seconds: float | None = DEFAULT_SOFT_LIMIT_SECONDS,
    speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    look_back_seconds: float = DEFAULT_LOOK_BACK_SECONDS,
    hard_limit_seconds: float | None = None,
    clip_repetitions: bool = False,
) -> int:
    """
    Stream WAV audio from stdin through VAD and transcribe each segment immediately.
    Output is always JSONL to stdout.

    Args:
        transcriber: Pre-loaded AudioTranscriber instance
        min_speech_seconds: Minimum speech duration in seconds
        soft_limit_seconds: Soft limit on speech segment duration in seconds
        speech_threshold: VAD speech detection threshold (0.0-1.0)
        min_silence_duration_ms: Minimum silence duration in ms to end segment
        look_back_seconds: Look-back buffer in seconds for segment start
        hard_limit_seconds: Hard limit on segment duration (from transcriber)

    Returns:
        Number of segments transcribed
    """
    segment_count = 0
    output_file = sys.stdout
    write_jsonl_marker("stream_start", output_file)

    def on_segment_complete(segment: AudioSegment) -> None:
        nonlocal segment_count
        assert segment.duration_seconds is not None
        start_fmt = format_timestamp(segment.start)
        logger.info("[VAD] Segment %d: %s (%.2fs), duration=%.2fs", segment_count, start_fmt, segment.start, segment.duration_seconds)

        write_jsonl_boundary("segment_start", segment.start, output_file)

        transcribed = transcriber.transcribe(segment.audio, segment.start)
        for ts in transcribed:
            write_jsonl_segment(ts, output_file, clip_repetitions=clip_repetitions)

        segment_end_time = segment.start + segment.duration_seconds
        write_jsonl_boundary("segment_end", segment_end_time, output_file)

        segment_count += 1

    speech_detector = SpeechDetector(
        sample_rate=TARGET_SAMPLE_RATE,
        on_segment_complete=on_segment_complete,
        min_speech_seconds=min_speech_seconds,
        soft_limit_seconds=soft_limit_seconds,
        speech_threshold=speech_threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        look_back_seconds=look_back_seconds,
        hard_limit_seconds=hard_limit_seconds if hard_limit_seconds is not None else DEFAULT_HARD_LIMIT_SECONDS,
    )

    window_size = get_window_size_samples()
    buffer: list[float] = []
    current_ts = 0.0
    chunks_read = 0

    for audio in stream_stdin_wav():
        chunks_read += 1
        buffer.extend(audio)

        while len(buffer) >= window_size:
            window = np.array(buffer[:window_size], dtype=np.float32)
            speech_detector.process_window(window, current_ts)
            buffer = buffer[window_size:]
            current_ts += window_size / TARGET_SAMPLE_RATE

        if chunks_read % 1000 == 0:
            logger.info("Progress: %d chunks, %.2fs", chunks_read, current_ts)

    logger.info("End of stream: %d chunks, %.2fs", chunks_read, current_ts)
    speech_detector.flush()
    write_jsonl_marker("stream_end", output_file)
    logger.info("Found %d speech segments", segment_count)
    return segment_count



def split_by_vad(
    audio_source: str,
    preserve_sample_rate: bool = False,
    output_format: str = "opus",
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
    soft_limit_seconds: float = DEFAULT_SOFT_LIMIT_SECONDS,
    speech_threshold: float = DEFAULT_SPEECH_THRESHOLD,
    min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_DURATION_MS,
    look_back_seconds: float = DEFAULT_LOOK_BACK_SECONDS,
) -> int:
    """
    Stream audio through VAD and save each segment to file.

    Args:
        audio_source: Path to audio file or URL
        preserve_sample_rate: If True, preserve original sample rate (mono).
                              If False (default), downsample to 16kHz mono.
        output_format: Output format - "opus" (16kbps) or "wav" (default: opus)
        min_speech_seconds: Minimum speech duration in seconds
        soft_limit_seconds: Soft limit on speech segment duration in seconds
        speech_threshold: VAD speech detection threshold (0.0-1.0)
        min_silence_duration_ms: Minimum silence duration in ms to end segment
        look_back_seconds: Look-back buffer in seconds for segment start

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
        logger.info("Audio properties: %dHz (preserving sample rate)", orig_sr)

        # Rolling buffer for original audio (float32 mono)
        original_buffer: list[float] = []
        buffer_start_time = 0.0

        def on_segment_complete(segment: AudioSegment) -> None:
            nonlocal segment_count, original_buffer, buffer_start_time
            assert segment.duration_seconds is not None

            # Calculate sample range in original buffer (mono)
            seg_start_in_buffer = segment.start - buffer_start_time
            start_sample = int(seg_start_in_buffer * orig_sr)
            end_sample = int((seg_start_in_buffer + segment.duration_seconds) * orig_sr)

            # Clamp to buffer bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(original_buffer), end_sample)

            # Extract original audio for this segment
            original_audio = np.array(original_buffer[start_sample:end_sample], dtype=np.float32)

            # Save with original sample rate (mono)
            path = save_audio_segment(
                segment, output_dir, segment_count,
                original_audio=original_audio,
                sample_rate=orig_sr,
                channels=1,
                output_format=output_format,
            )
            logger.info("[VAD] Saved: %s (duration=%.2fs)", path, segment.duration_seconds)

            # Trim buffer - keep only look-back audio for next segment
            look_back_samples = int(look_back_seconds * orig_sr)
            trim_to = max(0, end_sample - look_back_samples)
            if trim_to > 0:
                original_buffer = original_buffer[trim_to:]
                buffer_start_time += trim_to / orig_sr

            segment_count += 1

        speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            on_segment_complete=on_segment_complete,
            min_speech_seconds=min_speech_seconds,
            soft_limit_seconds=soft_limit_seconds,
            speech_threshold=speech_threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            look_back_seconds=look_back_seconds,
        )

        window_size = get_window_size_samples()
        vad_buffer: list[float] = []
        current_ts = 0.0
        chunks_read = 0

        # Stream at original sample rate, convert to mono
        for audio in ffmpeg_stream_float32(audio_source, ac=1):
            chunks_read += 1

            # Store original audio (float32) for later extraction
            original_buffer.extend(audio.tolist())

            # Resample to 16kHz for VAD
            resampled = resample_to_16k(audio, orig_sr)
            vad_buffer.extend(resampled.tolist())

            # Process VAD windows
            while len(vad_buffer) >= window_size:
                window = np.array(vad_buffer[:window_size], dtype=np.float32)
                speech_detector.process_window(window, current_ts)
                vad_buffer = vad_buffer[window_size:]
                current_ts += window_size / TARGET_SAMPLE_RATE

            if chunks_read % 1000 == 0:
                logger.info("Progress: %.2fs", current_ts)

        logger.info("End of stream: %d chunks, %.2fs", chunks_read, current_ts)
        speech_detector.flush()

    else:
        # Default mode: downsample to 16kHz mono (simpler, less memory)
        def on_segment_complete_default(segment: AudioSegment) -> None:
            nonlocal segment_count
            path = save_audio_segment(segment, output_dir, segment_count, output_format=output_format)
            logger.info("[VAD] Saved: %s (duration=%.2fs)", path, segment.duration_seconds)
            segment_count += 1

        speech_detector = SpeechDetector(
            sample_rate=TARGET_SAMPLE_RATE,
            on_segment_complete=on_segment_complete_default,
            min_speech_seconds=min_speech_seconds,
            soft_limit_seconds=soft_limit_seconds,
            speech_threshold=speech_threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            look_back_seconds=look_back_seconds,
        )

        window_size = get_window_size_samples()
        buffer: list[float] = []
        current_ts = 0.0
        chunks_read = 0

        # Stream at 16kHz mono
        for audio in ffmpeg_stream_float32(audio_source, target_sample_rate=TARGET_SAMPLE_RATE, ac=1):
            chunks_read += 1
            buffer.extend(audio)

            while len(buffer) >= window_size:
                window = np.array(buffer[:window_size], dtype=np.float32)
                speech_detector.process_window(window, current_ts)
                buffer = buffer[window_size:]
                current_ts += window_size / TARGET_SAMPLE_RATE

            if chunks_read % 1000 == 0:
                logger.info("Progress: %.2fs", current_ts)

        logger.info("End of stream: %d chunks, %.2fs", chunks_read, current_ts)
        speech_detector.flush()

    return segment_count


def main():
    """Main entry point for the CLI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    load_dotenv()
    parser = argparse.ArgumentParser(
        description='Whisper transcription tool with streaming audio processing'
    )

    # Create subparsers for each action
    subparsers = parser.add_subparsers(dest='action', required=True, help='Action to perform')

    # TRANSCRIBE subcommand
    parser_transcribe = subparsers.add_parser('transcribe', help='Transcribe audio from a file or stdin')
    transcribe_input = parser_transcribe.add_mutually_exclusive_group(required=True)
    transcribe_input.add_argument('--file', type=str, help='Path to audio file')
    transcribe_input.add_argument('--stdin', action='store_true', help='Read WAV audio from stdin (always uses VAD, outputs to stdout)')
    parser_transcribe.add_argument('--output', type=str, default=None,
                                   help='Output path for JSONL transcript (default: stdout)')
    parser_transcribe.add_argument('--language', type=str, default=None,
                                   help='Language code for transcription '
                                        '(required for moonshine, optional for others)')
    parser_transcribe.add_argument('--model', type=str, default=None,
                                   help='Model name (default: large-v3-turbo for whisper, '
                                        'zai-org/GLM-ASR-Nano-2512 for glm-asr, '
                                        'mlx-community/Qwen3-ASR-0.6B-bf16 for qwen-asr-mlx, '
                                        'mlx-community/GLM-ASR-Nano-2512-8bit for glm-asr-mlx, '
                                        'auto-selected for moonshine, '
                                        'ignored for nvidia-whisper). '
                                        'Local directory paths are accepted by whisper, qwen-asr-rs, and glm-asr. '
                                        'GGUF files are accepted only by qwen-asr-rs.')
    parser_transcribe.add_argument('--backend', type=str,
                                   choices=['whisper', 'moonshine', 'qwen-asr-rs', 'qwen-asr-mlx',
                                            'glm-asr', 'glm-asr-mlx', 'nvidia-whisper'],
                                   default='whisper',
                                   help='Transcription backend (default: whisper). '
                                        'nvidia-whisper hits the hosted whisper-large-v3 endpoint at '
                                        'build.nvidia.com and requires NVIDIA_API_KEY in .env.')
    parser_transcribe.add_argument('--chinese-conversion', type=str,
                                   choices=['none', 'simplified', 'traditional'],
                                   default='none',
                                   help='Chinese character conversion for zh/yue languages: '
                                        'none (default), simplified (zh-Hans), traditional (zh-Hant)')
    parser_transcribe.add_argument('--threads', type=int, default=None,
                                   help='Number of CPU threads for inference '
                                        '(default: min(2, cpu_count) for moonshine, '
                                        'none for other backends)')
    parser_transcribe.add_argument('--no-condition', action='store_true',
                                   help='Disable conditioning on previous segment output '
                                        '(whisper, qwen-asr-rs, and qwen-asr-mlx backends). '
                                        'By default, each segment is conditioned on the prior transcript '
                                        'for consistency.')
    parser_transcribe.add_argument('--no-sub-timestamps', action='store_true',
                                   help='Disable sub-sentence timestamp splitting '
                                        '(whisper backend only). Returns one segment per '
                                        'VAD segment instead of multiple timestamped chunks.')
    parser_transcribe.add_argument('--device', type=str, default=None,
                                   help='Device for whisper, qwen-asr-rs, and glm-asr backends: '
                                        'cpu, metal/mps, or cuda '
                                        '(default: auto-detect cuda > mps > cpu). '
                                        'The qwen-asr-mlx and glm-asr-mlx backends always use Metal and ignore this flag. '
                                        'The nvidia-whisper backend runs server-side and ignores this flag.')
    parser_transcribe.add_argument('--single-instance', action='store_true',
                                   help='Prevent multiple instances from running simultaneously')
    parser_transcribe.add_argument('--clip-repetitions', action='store_true',
                                   help='Truncate heavily-repetitive transcript text. '
                                        'When a char-level pattern repeats ≥10 times in a row '
                                        '(e.g. "dungu dungu dungu …" or "好好好…"), the looped '
                                        'tail is replaced with "(indistinguishable speech)" '
                                        'after the first copy of the pattern. '
                                        'Lines under 100 chars are passed through unchanged. '
                                        'Off by default.')

    # SPLIT subcommand
    parser_split = subparsers.add_parser('split', help='Split audio by VAD into Opus segments')
    split_input = parser_split.add_mutually_exclusive_group(required=True)
    split_input.add_argument('--file', type=str, help='Path to audio file')
    split_input.add_argument('--url', type=str, help='URL to audio (live streams not supported)')
    parser_split.add_argument('--preserve-sample-rate', action='store_true',
                              help='Preserve original sample rate (default: downsample to 16kHz)')
    parser_split.add_argument('--format', type=str, choices=['opus', 'wav'], default='opus',
                              help='Output format: opus (16kbps, default) or wav')
    parser_split.add_argument('--single-instance', action='store_true',
                              help='Prevent multiple instances from running simultaneously')
    add_vad_arguments(parser_split)

    args = parser.parse_args()

    try:
        lock = acquire_lock(args.action) if args.single_instance else None
        try:
            if lock:
                lock.acquire()

            if args.action == 'transcribe':
                if args.device is not None and args.backend == 'moonshine':
                    parser.error('--device is not supported by the moonshine backend')

                if args.threads is not None:
                    num_threads = args.threads
                elif args.backend == 'moonshine':
                    num_threads = min(2, os.cpu_count() or 1)
                else:
                    num_threads = None
                if num_threads is not None:
                    logger.info("Using %d thread(s)", num_threads)

                # Handle stdin mode separately (no validation, always VAD, always stdout)
                if getattr(args, 'stdin', False):
                    transcriber = create_transcriber(
                        args.language, args.model, args.backend,
                        args.chinese_conversion,
                        num_threads=num_threads,
                        condition=False if args.no_condition else None,
                        sub_timestamps=not args.no_sub_timestamps,
                        device=args.device,
                    )
                    segment_count = stream_transcribe_stdin_with_vad(
                        transcriber,
                        hard_limit_seconds=transcriber.hard_limit_seconds,
                        soft_limit_seconds=transcriber.soft_limit_seconds,
                        clip_repetitions=args.clip_repetitions,
                    )
                    logger.info("Transcribed %d segments from stdin", segment_count)
                else:
                    # File mode
                    audio_source = args.file
                    validate_audio_source(audio_source)

                    transcriber = create_transcriber(
                        args.language, args.model, args.backend,
                        args.chinese_conversion,
                        num_threads=num_threads,
                        condition=False if args.no_condition else None,
                        sub_timestamps=not args.no_sub_timestamps,
                        device=args.device,
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
                        segment_count = stream_transcribe_with_vad(
                            audio_source, transcriber, output_file,
                            hard_limit_seconds=transcriber.hard_limit_seconds,
                            soft_limit_seconds=transcriber.soft_limit_seconds,
                            clip_repetitions=args.clip_repetitions,
                        )

                        if args.output:
                            logger.info("Transcript written to %s (%d segments)", args.output, segment_count)
                    finally:
                        if args.output:
                            output_file.close()

            elif args.action == 'split':
                audio_source = args.file if args.file else args.url
                validate_audio_source(audio_source)
                segment_count = split_by_vad(
                    audio_source, args.preserve_sample_rate, args.format,
                    min_speech_seconds=args.min_speech_seconds,
                    soft_limit_seconds=args.soft_limit_seconds,
                    speech_threshold=args.speech_threshold,
                    min_silence_duration_ms=args.min_silence_duration_ms,
                    look_back_seconds=args.look_back_seconds,
                )
                base_name = os.path.splitext(os.path.basename(audio_source))[0]
                output_dir = os.path.join("tmp", base_name)
                logger.info("Saved %d segments to %s", segment_count, output_dir)
        finally:
            if lock:
                lock.release()

    except LockError as e:
        logger.error("%s", e)
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
