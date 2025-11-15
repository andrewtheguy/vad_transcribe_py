"""Audio file saving functionality for transcribed audio segments."""

import os
import subprocess
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from whisper_transcribe_py.vad_processor import AudioSegment

TARGET_SAMPLE_RATE = 16000

AudioSegmentCallback = Callable[[AudioSegment], None]


def create_audio_file_saver(show_name: str, directory: str = "./tmp/speech") -> AudioSegmentCallback:
    """Create a callback that saves audio segments to disk as Opus files.

    Args:
        show_name: Name of the show (used for organizing files)
        directory: Base directory for saving audio files

    Returns:
        Callback function that accepts AudioSegment and saves it to disk
    """
    # Base directory for this show
    base_show_directory = os.path.join(directory, show_name)

    def _save(segment: AudioSegment):
        audio = segment.audio

        # Use wall clock timestamps for livestream mode, relative timestamps for file mode
        if segment.wall_clock_start is not None:
            # Livestream mode: use yyyymmddhhmmss.microseconds UTC format
            start_dt = datetime.fromtimestamp(segment.wall_clock_start, timezone.utc)
            start_timestamp = start_dt.strftime("%Y%m%d%H%M%S.%f")

            end_ts = segment.wall_clock_start + len(audio) / TARGET_SAMPLE_RATE
            end_dt = datetime.fromtimestamp(end_ts, timezone.utc)
            end_timestamp = end_dt.strftime("%Y%m%d%H%M%S.%f")

            # Organize by date: tmp/speech/showname/yyyy/mm/dd/
            date_path = start_dt.strftime("%Y/%m/%d")
            target_directory = os.path.join(base_show_directory, date_path)
        else:
            # File mode: use relative timestamps, no date subdirectories
            start_timestamp = f"{segment.start:08.3f}"
            end_timestamp = f"{(segment.start + len(audio) / TARGET_SAMPLE_RATE):08.3f}"
            target_directory = base_show_directory

        # Create target directory if it doesn't exist
        os.makedirs(target_directory, exist_ok=True)

        # Convert float32 audio to int16 for opus encoding
        max_int16 = np.iinfo(np.int16).max
        audio_int16 = (audio * max_int16).astype(np.int16)

        # Create final opus file path
        final_path = os.path.join(target_directory, f"{start_timestamp}-{end_timestamp}.opus")

        # Create temporary file path with proper extension for atomic save
        temp_path = f"{final_path}.tmp"

        # Use ffmpeg to encode to opus 8kbps with 8kHz sample rate
        command = [
            "ffmpeg",
            "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
            "-acodec", "pcm_s16le",  # Input codec
            "-ac", "1",  # Number of channels (1 = mono)
            "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
            "-i", "pipe:",  # Input from stdin
            "-c:a", "libopus",  # Audio codec: Opus
            "-b:a", "8k",  # Audio bitrate: 8kbps
            #"-ar", "8000",  # Output sample rate: 8kHz
            "-f", "ogg",  # Explicitly specify container format
            "-y",  # Overwrite output file if it exists
            temp_path
        ]

        # # Create final mp4 file path
        # final_path = os.path.join(directory, f"{start_timestamp}-{end_timestamp}.m4a")

        # # Create temporary file path with proper extension for atomic save
        # temp_path = f"{final_path}.tmp"

        # # Use ffmpeg to encode to opus 8kbps
        # command = [
        #     "ffmpeg",
        #     "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
        #     "-acodec", "pcm_s16le",  # Input codec
        #     "-ac", "1",  # Number of channels (1 = mono)
        #     "-ar", str(TARGET_SAMPLE_RATE),  # Sample rate
        #     "-i", "pipe:",  # Input from stdin
        #     "-c:a", "aac",  # Audio codec: AAC
        #     "-b:a", "8k",  # Audio bitrate: 8kbps
        #     "-f", "ipod",  # Explicitly specify container format
        #     "-y",  # Overwrite output file if it exists
        #     temp_path
        # ]

        process = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Write audio data to ffmpeg stdin
            process.stdin.write(audio_int16.tobytes())
            process.stdin.close()

            # Wait for ffmpeg to finish
            returncode = process.wait()
            if returncode != 0:
                stderr_output = process.stderr.read().decode('utf-8', errors='replace')
                raise ValueError(f"ffmpeg opus encoding failed with return code {returncode}. Error: {stderr_output}")

            # Atomically rename temp file to final file
            os.rename(temp_path, final_path)

        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
            raise RuntimeError(f"Failed to save opus audio: {e}")
        finally:
            if process and process.stderr:
                process.stderr.close()

    return _save
