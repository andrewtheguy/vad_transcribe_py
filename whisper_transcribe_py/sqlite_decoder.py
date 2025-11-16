"""SQLite database decoder for exporting and transcribing speech segments."""

import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Iterator, Tuple

import numpy as np


TARGET_SAMPLE_RATE = 16000


def read_database_metadata(db_path: str) -> Tuple[str, str, str]:
    """Read metadata from SQLite database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Tuple of (audio_format, database_id, show_name)

    Raises:
        FileNotFoundError: If database doesn't exist
        ValueError: If metadata is missing or invalid
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'version'")
        result = cursor.fetchone()
        if not result:
            raise ValueError("version not found in metadata table")
        version = result[0]

        if version != "1":
            raise ValueError(
                f"Database version mismatch: database was created with version '{version}' "
                f"but current version is '1'. "
                f"Please upgrade the database or use a compatible version of the application."
            )

        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'audio_format'")
        result = cursor.fetchone()
        if not result:
            raise ValueError("audio_format not found in metadata table")
        audio_format = result[0]

        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'database_id'")
        result = cursor.fetchone()
        if not result:
            raise ValueError("database_id not found in metadata table")
        database_id = result[0]

        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'show_name'")
        result = cursor.fetchone()
        if not result:
            raise ValueError("show_name not found in metadata table")
        show_name = result[0]

        if audio_format not in ('wav', 'm4a'):
            raise ValueError(f"Invalid audio_format in metadata: {audio_format}")

        return audio_format, database_id, show_name
    finally:
        conn.close()


def get_max_speech_id(conn: sqlite3.Connection) -> int:
    """Get maximum speech ID from database (snapshot).

    Args:
        conn: SQLite database connection

    Returns:
        Maximum ID from speech table

    Raises:
        ValueError: If speech table is empty
    """
    cursor = conn.execute("SELECT MAX(id) FROM speech")
    result = cursor.fetchone()
    if result is None or result[0] is None:
        raise ValueError("speech table is empty")
    return result[0]


def read_speech_segments(conn: sqlite3.Connection, max_id: int) -> Iterator[Tuple[int, str, str, bytes]]:
    """Read speech segments from database up to max_id.

    Args:
        conn: SQLite database connection
        max_id: Maximum ID to read (inclusive)

    Yields:
        Tuples of (id, start_ts, end_ts, audio_data)
    """
    cursor = conn.execute(
        "SELECT id, start_ts, end_ts, audio_data FROM speech WHERE id <= ? ORDER BY id",
        (max_id,)
    )
    for row in cursor:
        yield row


def decode_audio_segment_wav(audio_data: bytes) -> np.ndarray:
    """Decode WAV audio segment from database.

    Args:
        audio_data: Raw PCM int16 samples

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0]
    """
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    return audio_int16.astype(np.float32) / np.iinfo(np.int16).max


def decode_audio_segment_m4a(audio_data: bytes) -> np.ndarray:
    """Decode M4A (ADTS AAC) audio segment from database.

    Args:
        audio_data: Raw ADTS AAC stream

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0]

    Raises:
        RuntimeError: If ffmpeg decoding fails
    """
    command = [
        "ffmpeg",
        "-f", "aac",  # Input format: raw AAC (ADTS)
        "-i", "pipe:0",  # Input from stdin
        "-f", "s16le",  # Output format: raw PCM signed 16-bit little-endian
        "-ar", str(TARGET_SAMPLE_RATE),  # Output sample rate
        "-ac", "1",  # Output channels: mono
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

        pcm_data, stderr_output = process.communicate(input=audio_data)

        if process.returncode != 0:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            raise RuntimeError(f"ffmpeg M4A decoding failed: {stderr_text}")

        # Convert PCM to float32
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
        return audio_int16.astype(np.float32) / np.iinfo(np.int16).max

    except Exception as e:
        raise RuntimeError(f"Failed to decode M4A audio segment: {e}")


def decode_audio_segment(audio_data: bytes, audio_format: str) -> np.ndarray:
    """Decode audio segment based on format.

    Args:
        audio_data: Raw audio data from database
        audio_format: 'wav' or 'm4a'

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0]

    Raises:
        ValueError: If audio_format is unsupported
        RuntimeError: If decoding fails
    """
    if audio_format == 'wav':
        return decode_audio_segment_wav(audio_data)
    elif audio_format == 'm4a':
        return decode_audio_segment_m4a(audio_data)
    else:
        raise ValueError(f"Unsupported audio_format: {audio_format}")


def concatenate_and_save_audio_wav(audio_segments: list[bytes], output_path: str):
    """Concatenate WAV segments and save to file.

    Args:
        audio_segments: List of raw PCM int16 audio data
        output_path: Path to save concatenated WAV file

    Raises:
        RuntimeError: If ffmpeg encoding fails
    """
    # Concatenate all PCM data
    all_pcm = b''.join(audio_segments)

    command = [
        "ffmpeg",
        "-f", "s16le",  # Input format: raw PCM signed 16-bit little-endian
        "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
        "-ac", "1",  # Input channels: mono
        "-i", "pipe:0",  # Input from stdin
        "-c:a", "pcm_s16le",  # Output codec: PCM
        "-f", "wav",  # Output format: WAV
        "-y",  # Overwrite output file
        output_path
    ]

    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        _, stderr_output = process.communicate(input=all_pcm)

        if process.returncode != 0:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            raise RuntimeError(f"ffmpeg WAV encoding failed: {stderr_text}")

    except Exception as e:
        raise RuntimeError(f"Failed to save concatenated WAV: {e}")


def concatenate_and_save_audio_m4a(audio_segments: list[bytes], output_path: str):
    """Concatenate M4A (ADTS AAC) segments and save to file.

    Args:
        audio_segments: List of raw ADTS AAC streams
        output_path: Path to save concatenated M4A file

    Raises:
        RuntimeError: If ffmpeg encoding fails
    """
    # Concatenate all ADTS streams (ADTS format supports direct concatenation)
    all_adts = b''.join(audio_segments)

    command = [
        "ffmpeg",
        "-f", "aac",  # Input format: raw AAC (ADTS)
        "-i", "pipe:0",  # Input from stdin
        "-c:a", "copy",  # Copy codec (no re-encoding)
        "-f", "ipod",  # Output format: M4A (iPod)
        "-y",  # Overwrite output file
        output_path
    ]

    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        _, stderr_output = process.communicate(input=all_adts)

        if process.returncode != 0:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            raise RuntimeError(f"ffmpeg M4A encoding failed: {stderr_text}")

    except Exception as e:
        raise RuntimeError(f"Failed to save concatenated M4A: {e}")


def concatenate_and_save_audio(audio_segments: list[bytes], output_path: str, audio_format: str):
    """Concatenate audio segments and save to file.

    Args:
        audio_segments: List of raw audio data from database
        output_path: Path to save concatenated audio file
        audio_format: 'wav' or 'm4a'

    Raises:
        ValueError: If audio_format is unsupported
        RuntimeError: If encoding fails
    """
    if audio_format == 'wav':
        concatenate_and_save_audio_wav(audio_segments, output_path)
    elif audio_format == 'm4a':
        concatenate_and_save_audio_m4a(audio_segments, output_path)
    else:
        raise ValueError(f"Unsupported audio_format: {audio_format}")
