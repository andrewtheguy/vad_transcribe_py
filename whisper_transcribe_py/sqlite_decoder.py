"""SQLite database decoder for exporting and transcribing speech segments."""

import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Iterator, Tuple, Optional

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

    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'version'")
        result = cursor.fetchone()
        if not result:
            raise ValueError("version not found in metadata table")
        version = result[0]

        if version != "2":
            raise ValueError(
                f"Database version mismatch: database was created with version '{version}' "
                f"but current version is '2'. "
                f"No backward compatibility - please create a new database."
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


def read_speech_segments(conn: sqlite3.Connection, max_id: int) -> Iterator[Tuple[int, str, str, Optional[bytes], Optional[str]]]:
    """Read speech segments from database up to max_id.

    Args:
        conn: SQLite database connection
        max_id: Maximum ID to read (inclusive)

    Yields:
        Tuples of (id, start_ts, end_ts, audio_data, text)
        - audio_data is None for notice entries (text-only)
        - text is None for audio entries (audio-only)
    """
    cursor = conn.execute(
        "SELECT id, start_ts, end_ts, audio_data, text FROM speech WHERE id <= ? ORDER BY id",
        (max_id,)
    )
    for row in cursor:
        yield row


def decode_audio_segment_wav(audio_data: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode WAV audio segment from database.

    Args:
        audio_data: Raw PCM int16 samples, or None for notice entries

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0], or None if audio_data is None
    """
    if audio_data is None:
        return None
    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
    return audio_int16.astype(np.float32) / np.iinfo(np.int16).max


def decode_audio_segment_m4a(audio_data: Optional[bytes]) -> Optional[np.ndarray]:
    """Decode M4A (ADTS AAC) audio segment from database.

    Args:
        audio_data: Raw ADTS AAC stream, or None for notice entries

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0], or None if audio_data is None

    Raises:
        RuntimeError: If ffmpeg decoding fails
    """
    if audio_data is None:
        return None

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


def decode_audio_segment(audio_data: Optional[bytes], audio_format: str) -> Optional[np.ndarray]:
    """Decode audio segment based on format.

    Args:
        audio_data: Raw audio data from database, or None for notice entries
        audio_format: 'wav' or 'm4a'

    Returns:
        Float32 numpy array normalized to [-1.0, 1.0], or None if audio_data is None

    Raises:
        ValueError: If audio_format is unsupported
        RuntimeError: If decoding fails
    """
    if audio_data is None:
        return None

    if audio_format == 'wav':
        return decode_audio_segment_wav(audio_data)
    elif audio_format == 'm4a':
        return decode_audio_segment_m4a(audio_data)
    else:
        raise ValueError(f"Unsupported audio_format: {audio_format}")


def concatenate_and_save_audio_wav(audio_segments: list[Optional[bytes]], output_path: str, timestamps: Optional[list[Optional[tuple[str, str]]]] = None) -> list[str]:
    """Concatenate WAV segments and save to file(s).

    None entries (notices) cause the audio to be split into multiple files.
    Files are named with timestamps if provided, otherwise: base_1.wav, base_2.wav, etc.

    Args:
        audio_segments: List of raw PCM int16 audio data (None entries cause file splits)
        output_path: Base path for output files (e.g., "output.wav" -> "output_1.wav", "output_2.wav")
        timestamps: Optional list of (start_ts, end_ts) tuples for each segment (None entries for notices)

    Returns:
        List of created file paths

    Raises:
        RuntimeError: If ffmpeg encoding fails
    """
    # Split segments into groups separated by None entries, tracking timestamps
    groups = []
    group_timestamps = []
    current_group = []
    current_group_start_ts = None
    current_group_end_ts = None

    for idx, seg in enumerate(audio_segments):
        if seg is None:
            # None entry - save current group and start a new one
            if current_group:
                groups.append(current_group)
                group_timestamps.append((current_group_start_ts, current_group_end_ts))
                current_group = []
                current_group_start_ts = None
                current_group_end_ts = None
        else:
            current_group.append(seg)
            # Track timestamps for this group
            if timestamps and idx < len(timestamps) and timestamps[idx]:
                start_ts, end_ts = timestamps[idx]
                if current_group_start_ts is None:
                    current_group_start_ts = start_ts
                current_group_end_ts = end_ts

    # Add final group
    if current_group:
        groups.append(current_group)
        group_timestamps.append((current_group_start_ts, current_group_end_ts))

    if not groups:
        return []

    # Generate output file paths
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'wav'

    created_files = []

    for i, group in enumerate(groups, start=1):
        # Concatenate PCM data for this group
        all_pcm = b''.join(group)

        # Generate output path
        if len(groups) == 1:
            # Only one group - use original path with timestamps if available
            if timestamps and group_timestamps[0][0] and group_timestamps[0][1]:
                start_ts, end_ts = group_timestamps[0]
                # Sanitize timestamps for filename (replace : with -)
                start_ts_safe = start_ts.replace(':', '-').replace('.', '-')
                end_ts_safe = end_ts.replace(':', '-').replace('.', '-')
                file_path = f"{base_path}_{start_ts_safe}_to_{end_ts_safe}.{extension}"
            else:
                file_path = output_path
        else:
            # Multiple groups - add suffix with timestamps if available
            if timestamps and group_timestamps[i-1][0] and group_timestamps[i-1][1]:
                start_ts, end_ts = group_timestamps[i-1]
                # Sanitize timestamps for filename (replace : with -)
                start_ts_safe = start_ts.replace(':', '-').replace('.', '-')
                end_ts_safe = end_ts.replace(':', '-').replace('.', '-')
                file_path = f"{base_path}_{i}_{start_ts_safe}_to_{end_ts_safe}.{extension}"
            else:
                file_path = f"{base_path}_{i}.{extension}"

        command = [
            "ffmpeg",
            "-f", "s16le",  # Input format: raw PCM signed 16-bit little-endian
            "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
            "-ac", "1",  # Input channels: mono
            "-i", "pipe:0",  # Input from stdin
            "-c:a", "pcm_s16le",  # Output codec: PCM
            "-f", "wav",  # Output format: WAV
            "-y",  # Overwrite output file
            file_path
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

            created_files.append(file_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save concatenated WAV: {e}")

    return created_files


def concatenate_and_save_audio_m4a(audio_segments: list[Optional[bytes]], output_path: str, timestamps: Optional[list[Optional[tuple[str, str]]]] = None) -> list[str]:
    """Concatenate M4A (ADTS AAC) segments and save to file(s).

    None entries (notices) cause the audio to be split into multiple files.
    Files are named with timestamps if provided, otherwise: base_1.m4a, base_2.m4a, etc.

    Args:
        audio_segments: List of raw ADTS AAC streams (None entries cause file splits)
        output_path: Base path for output files (e.g., "output.m4a" -> "output_1.m4a", "output_2.m4a")
        timestamps: Optional list of (start_ts, end_ts) tuples for each segment (None entries for notices)

    Returns:
        List of created file paths

    Raises:
        RuntimeError: If ffmpeg encoding fails
    """
    # Split segments into groups separated by None entries, tracking timestamps
    groups = []
    group_timestamps = []
    current_group = []
    current_group_start_ts = None
    current_group_end_ts = None

    for idx, seg in enumerate(audio_segments):
        if seg is None:
            # None entry - save current group and start a new one
            if current_group:
                groups.append(current_group)
                group_timestamps.append((current_group_start_ts, current_group_end_ts))
                current_group = []
                current_group_start_ts = None
                current_group_end_ts = None
        else:
            current_group.append(seg)
            # Track timestamps for this group
            if timestamps and idx < len(timestamps) and timestamps[idx]:
                start_ts, end_ts = timestamps[idx]
                if current_group_start_ts is None:
                    current_group_start_ts = start_ts
                current_group_end_ts = end_ts

    # Add final group
    if current_group:
        groups.append(current_group)
        group_timestamps.append((current_group_start_ts, current_group_end_ts))

    if not groups:
        return []

    # Generate output file paths
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    extension = output_path.rsplit('.', 1)[1] if '.' in output_path else 'm4a'

    created_files = []

    for i, group in enumerate(groups, start=1):
        # Concatenate ADTS data for this group (ADTS format supports direct concatenation)
        all_adts = b''.join(group)

        # Generate output path
        if len(groups) == 1:
            # Only one group - use original path with timestamps if available
            if timestamps and group_timestamps[0][0] and group_timestamps[0][1]:
                start_ts, end_ts = group_timestamps[0]
                # Sanitize timestamps for filename (replace : with -)
                start_ts_safe = start_ts.replace(':', '-').replace('.', '-')
                end_ts_safe = end_ts.replace(':', '-').replace('.', '-')
                file_path = f"{base_path}_{start_ts_safe}_to_{end_ts_safe}.{extension}"
            else:
                file_path = output_path
        else:
            # Multiple groups - add suffix with timestamps if available
            if timestamps and group_timestamps[i-1][0] and group_timestamps[i-1][1]:
                start_ts, end_ts = group_timestamps[i-1]
                # Sanitize timestamps for filename (replace : with -)
                start_ts_safe = start_ts.replace(':', '-').replace('.', '-')
                end_ts_safe = end_ts.replace(':', '-').replace('.', '-')
                file_path = f"{base_path}_{i}_{start_ts_safe}_to_{end_ts_safe}.{extension}"
            else:
                file_path = f"{base_path}_{i}.{extension}"

        command = [
            "ffmpeg",
            "-f", "aac",  # Input format: raw AAC (ADTS)
            "-i", "pipe:0",  # Input from stdin
            "-c:a", "copy",  # Copy codec (no re-encoding)
            "-f", "ipod",  # Output format: M4A (iPod)
            "-y",  # Overwrite output file
            file_path
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

            created_files.append(file_path)

        except Exception as e:
            raise RuntimeError(f"Failed to save concatenated M4A: {e}")

    return created_files


def concatenate_and_save_audio(audio_segments: list[Optional[bytes]], output_path: str, audio_format: str, timestamps: Optional[list[Optional[tuple[str, str]]]] = None) -> list[str]:
    """Concatenate audio segments and save to file(s).

    None entries (notices) cause the audio to be split into multiple files.

    Args:
        audio_segments: List of raw audio data from database (None entries cause file splits)
        output_path: Base path for output files
        audio_format: 'wav' or 'm4a'
        timestamps: Optional list of (start_ts, end_ts) tuples for each segment (None entries for notices)

    Returns:
        List of created file paths

    Raises:
        ValueError: If audio_format is unsupported
        RuntimeError: If encoding fails
    """
    if audio_format == 'wav':
        return concatenate_and_save_audio_wav(audio_segments, output_path, timestamps)
    elif audio_format == 'm4a':
        return concatenate_and_save_audio_m4a(audio_segments, output_path, timestamps)
    else:
        raise ValueError(f"Unsupported audio_format: {audio_format}")
