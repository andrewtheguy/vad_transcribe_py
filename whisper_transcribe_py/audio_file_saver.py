"""Audio file saving functionality for transcribed audio segments."""

import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from whisper_transcribe_py.vad_processor import AudioSegment

TARGET_SAMPLE_RATE = 16000
OUTPUT_FORMAT = "wav"  # Options: "opus", "wav"
STORAGE_TYPE = "sqlite"  # Options: "file", "sqlite"

AudioSegmentCallback = Callable[[AudioSegment], None]


def _create_file_saver(show_name: str, directory: str) -> AudioSegmentCallback:
    """Create a callback that saves audio segments to individual files.

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

        # Convert float32 audio to int16 for encoding
        max_int16 = np.iinfo(np.int16).max
        audio_int16 = (audio * max_int16).astype(np.int16)

        # Create final audio file path with format-specific extension
        final_path = os.path.join(target_directory, f"{start_timestamp}-{end_timestamp}.{OUTPUT_FORMAT}")

        # Create temporary file path with proper extension for atomic save
        temp_path = f"{final_path}.tmp"

        # Build ffmpeg command based on output format
        if OUTPUT_FORMAT == "wav":
            # WAV: Uncompressed PCM format
            command = [
                "ffmpeg",
                "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
                "-acodec", "pcm_s16le",  # Input codec
                "-ac", "1",  # Number of channels (1 = mono)
                "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
                "-i", "pipe:",  # Input from stdin
                "-c:a", "pcm_s16le",  # Output codec: uncompressed PCM
                "-f", "wav",  # Container format: WAV
                "-y",  # Overwrite output file if it exists
                temp_path
            ]
        elif OUTPUT_FORMAT == "opus":
            # Opus: Compressed format (8kbps)
            command = [
                "ffmpeg",
                "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
                "-acodec", "pcm_s16le",  # Input codec
                "-ac", "1",  # Number of channels (1 = mono)
                "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
                "-i", "pipe:",  # Input from stdin
                "-c:a", "libopus",  # Audio codec: Opus
                "-b:a", "8k",  # Audio bitrate: 8kbps
                "-f", "ogg",  # Container format: OGG
                "-y",  # Overwrite output file if it exists
                temp_path
            ]
        else:
            raise ValueError(f"Unsupported OUTPUT_FORMAT: {OUTPUT_FORMAT}. Supported formats: 'opus', 'wav'")

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
                raise ValueError(f"ffmpeg {OUTPUT_FORMAT} encoding failed with return code {returncode}. Error: {stderr_output}")

            # Atomically rename temp file to final file
            os.rename(temp_path, final_path)

        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors
            raise RuntimeError(f"Failed to save {OUTPUT_FORMAT} audio: {e}")
        finally:
            if process and process.stderr:
                process.stderr.close()

    return _save


def _encode_to_raw_opus(audio_int16: np.ndarray) -> bytes:
    """Encode audio to raw Opus packets using ffmpeg.

    Args:
        audio_int16: Audio samples as int16 numpy array

    Returns:
        Raw Opus packet data (without Ogg container)

    Raises:
        RuntimeError: If ffmpeg encoding fails
    """
    command = [
        "ffmpeg",
        "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
        "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
        "-ac", "1",  # Number of channels (1 = mono)
        "-i", "pipe:",  # Input from stdin
        "-c:a", "libopus",  # Audio codec: Opus
        "-b:a", "8k",  # Audio bitrate: 8kbps
        "-f", "data",  # Output raw data stream
        "pipe:1"  # Output to stdout
    ]

    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Write audio data to ffmpeg stdin and get output
        opus_data, stderr_output = process.communicate(input=audio_int16.tobytes())

        if process.returncode != 0:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            raise ValueError(f"ffmpeg opus encoding failed with return code {process.returncode}. Error: {stderr_text}")

        return opus_data

    except Exception as e:
        raise RuntimeError(f"Failed to encode to raw Opus: {e}")


def _create_sqlite_saver(show_name: str) -> AudioSegmentCallback:
    """Create a callback that saves audio segments to an SQLite database.

    Args:
        show_name: Name of the show (used for database filename)

    Returns:
        Callback function that accepts AudioSegment and saves it to SQLite
    """
    # Create database directory
    db_dir = "./tmp/speech_sqlite"
    os.makedirs(db_dir, exist_ok=True)

    # Database file path includes format to distinguish different encodings
    db_path = os.path.join(db_dir, f"{show_name}_{OUTPUT_FORMAT}.sqlite")

    # Initialize database and schema
    conn = sqlite3.connect(db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS speech (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_ts TEXT NOT NULL,
        end_ts TEXT NOT NULL,
        audio_data BLOB NOT NULL
    )''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_start_ts ON speech(start_ts)')
    conn.commit()

    def _save(segment: AudioSegment):
        audio = segment.audio

        # Calculate timestamps in ISO 8601 format
        if segment.wall_clock_start is not None:
            # Livestream mode: use wall clock timestamps
            start_dt = datetime.fromtimestamp(segment.wall_clock_start, timezone.utc)
            end_ts_float = segment.wall_clock_start + len(audio) / TARGET_SAMPLE_RATE
            end_dt = datetime.fromtimestamp(end_ts_float, timezone.utc)
        else:
            # File mode: use relative timestamps (convert to datetime from start of epoch)
            # Note: This is less common for SQLite storage, mainly for livestream
            start_dt = datetime.fromtimestamp(segment.start, timezone.utc)
            end_ts_float = segment.start + len(audio) / TARGET_SAMPLE_RATE
            end_dt = datetime.fromtimestamp(end_ts_float, timezone.utc)

        # Format as ISO 8601 strings
        start_ts_str = start_dt.isoformat()
        end_ts_str = end_dt.isoformat()

        # Convert float32 audio to int16
        max_int16 = np.iinfo(np.int16).max
        audio_int16 = (audio * max_int16).astype(np.int16)

        # Encode audio data based on OUTPUT_FORMAT
        if OUTPUT_FORMAT == "wav":
            # WAV mode: store raw PCM samples (no WAV header, just audio data)
            audio_data = audio_int16.tobytes()

        elif OUTPUT_FORMAT == "opus":
            # Opus mode: encode to raw Opus packets
            audio_data = _encode_to_raw_opus(audio_int16)

        else:
            raise ValueError(f"Unsupported OUTPUT_FORMAT for SQLite: {OUTPUT_FORMAT}. Supported: 'opus', 'wav'")

        # Insert into database
        try:
            conn.execute(
                'INSERT INTO speech (start_ts, end_ts, audio_data) VALUES (?, ?, ?)',
                (start_ts_str, end_ts_str, audio_data)
            )
            conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to save audio segment to SQLite: {e}")

    return _save


def create_audio_file_saver(show_name: str, directory: str = "./tmp/speech") -> AudioSegmentCallback:
    """Create a callback that saves audio segments.

    Storage backend is determined by the STORAGE_TYPE module constant:
    - "file": Save segments as individual files (opus or wav)
    - "sqlite": Save segments to SQLite database (wav only for now)

    Output format is determined by the OUTPUT_FORMAT module constant:
    - "opus": Compressed Opus format (8kbps)
    - "wav": Uncompressed PCM format

    For SQLite storage:
    - Database location: ./tmp/speech_sqlite/{show_name}_{format}.sqlite
    - Table: speech (id, start_ts, end_ts, audio_data)
    - Timestamps: ISO 8601 format with timezone
    - Audio data: Raw PCM for wav
    - Note: Opus format not currently supported with SQLite storage

    Args:
        show_name: Name of the show (used for organizing files/database)
        directory: Base directory for file storage (ignored for sqlite mode)

    Returns:
        Callback function that accepts AudioSegment and saves it

    Raises:
        ValueError: If STORAGE_TYPE or OUTPUT_FORMAT is unsupported or incompatible
    """
    # Validate STORAGE_TYPE and OUTPUT_FORMAT combination
    if STORAGE_TYPE == "sqlite" and OUTPUT_FORMAT == "opus":
        raise ValueError(
            "STORAGE_TYPE='sqlite' with OUTPUT_FORMAT='opus' is not currently supported. "
            "Please use OUTPUT_FORMAT='wav' with SQLite storage, or use STORAGE_TYPE='file' for Opus format."
        )

    if STORAGE_TYPE == "file":
        return _create_file_saver(show_name, directory)
    elif STORAGE_TYPE == "sqlite":
        return _create_sqlite_saver(show_name)
    else:
        raise ValueError(f"Unsupported STORAGE_TYPE: {STORAGE_TYPE}. Supported: 'file', 'sqlite'")
