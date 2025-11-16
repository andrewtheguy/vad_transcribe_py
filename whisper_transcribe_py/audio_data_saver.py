"""Audio data saving functionality for transcribed audio segments."""

import os
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Callable, TYPE_CHECKING, Optional, Union

import numpy as np

from whisper_transcribe_py.vad_processor import AudioSegment

if TYPE_CHECKING:
    from __main__ import ThreadExceptionHandler
    from whisper_transcribe_py.audio_transcriber import TranscriptionNotice

TARGET_SAMPLE_RATE = 16000
OUTPUT_FORMAT = "m4a"  # Options: "wav", "m4a"

# Callback accepts either AudioSegment or TranscriptionNotice
if TYPE_CHECKING:
    AudioSegmentCallback = Callable[[Union[AudioSegment, 'TranscriptionNotice']], None]
else:
    AudioSegmentCallback = Callable[[Union[AudioSegment, object]], None]


def _get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    """Get a metadata value from the database.

    Args:
        conn: SQLite database connection
        key: Metadata key to retrieve

    Returns:
        Metadata value or None if key doesn't exist
    """
    cursor = conn.execute('SELECT value FROM metadata WHERE key = ?', (key,))
    result = cursor.fetchone()
    return result[0] if result else None


def _set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set a metadata value in the database.

    Args:
        conn: SQLite database connection
        key: Metadata key to set
        value: Metadata value to set
    """
    conn.execute(
        'INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)',
        (key, value)
    )


def _initialize_metadata(conn: sqlite3.Connection, audio_format: str, show_name: str) -> None:
    """Initialize metadata table with version, format, show name, and database ID.

    Args:
        conn: SQLite database connection
        audio_format: Audio format being used ('wav' or 'm4a')
        show_name: Show name for this database

    Raises:
        ValueError: If existing version, format, or show name doesn't match current settings
    """
    # Create metadata table if it doesn't exist
    conn.execute('''CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )''')

    # Check if metadata already exists
    existing_version = _get_metadata(conn, 'version')
    existing_format = _get_metadata(conn, 'audio_format')
    existing_show_name = _get_metadata(conn, 'show_name')

    if existing_version is not None:
        # Metadata exists - validate version, format, and show name match
        if existing_version != "2":
            raise ValueError(
                f"Database version mismatch: database was created with version '{existing_version}' "
                f"but current version is '2'. "
                f"No backward compatibility - please create a new database."
            )
        if existing_format != audio_format:
            raise ValueError(
                f"Database format mismatch: database was created with format '{existing_format}' "
                f"but current OUTPUT_FORMAT is '{audio_format}'. "
                f"Please use the correct format or create a new database."
            )
        if existing_show_name != show_name:
            raise ValueError(
                f"Database show name mismatch: database was created for show '{existing_show_name}' "
                f"but current show name is '{show_name}'. "
                f"Please use the correct show name or create a new database."
            )
        # Version, format, and show name match - no action needed, database_id already exists
    else:
        # No metadata exists - initialize it
        database_id = str(uuid.uuid4())
        _set_metadata(conn, 'version', '2')
        _set_metadata(conn, 'audio_format', audio_format)
        _set_metadata(conn, 'show_name', show_name)
        _set_metadata(conn, 'database_id', database_id)


def _encode_to_raw_aac(audio_int16: np.ndarray) -> bytes:
    """Encode audio to raw AAC stream using ffmpeg.

    Args:
        audio_int16: Audio samples as int16 numpy array

    Returns:
        Raw AAC stream data (ADTS format for concatenation)

    Raises:
        RuntimeError: If ffmpeg encoding fails
    """
    command = [
        "ffmpeg",
        "-f", "s16le",  # Input format: raw PCM, signed 16-bit little-endian
        "-ar", str(TARGET_SAMPLE_RATE),  # Input sample rate
        "-ac", "1",  # Number of channels (1 = mono)
        "-i", "pipe:",  # Input from stdin
        "-c:a", "aac",  # Audio codec: AAC
        "-b:a", "8k",  # Audio bitrate: 8kbps
        "-f", "adts",  # Output format: ADTS (raw AAC with headers for concatenation)
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
        aac_data, stderr_output = process.communicate(input=audio_int16.tobytes())

        if process.returncode != 0:
            stderr_text = stderr_output.decode('utf-8', errors='replace')
            raise ValueError(f"ffmpeg AAC encoding failed with return code {process.returncode}. Error: {stderr_text}")

        return aac_data

    except Exception as e:
        raise RuntimeError(f"Failed to encode to raw AAC: {e}")


def _save_notice_to_sqlite(conn: sqlite3.Connection, notice) -> None:
    """Save a TranscriptionNotice to the SQLite database.

    Args:
        conn: SQLite database connection
        notice: TranscriptionNotice object to save

    Raises:
        RuntimeError: If saving fails
    """
    # Calculate timestamp - notice.timestamp is wall clock time
    if notice.timestamp is not None:
        start_dt = datetime.fromtimestamp(notice.timestamp, timezone.utc)
    else:
        # If no timestamp provided, use current time
        start_dt = datetime.now(timezone.utc)

    # Format as ISO 8601 string
    start_ts_str = start_dt.isoformat()
    end_ts_str = start_ts_str  # Same as start_ts for notices (instantaneous event)

    # Insert into database with NULL audio_data
    try:
        conn.execute(
            'INSERT INTO speech (start_ts, end_ts, audio_data, text) VALUES (?, ?, NULL, ?)',
            (start_ts_str, end_ts_str, notice.text)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save TranscriptionNotice to SQLite: {e}")


def create_audio_data_saver(
    show_name: str,
    exception_handler: Optional['ThreadExceptionHandler'] = None
) -> AudioSegmentCallback:
    """Create a callback that saves audio segments to an SQLite database.

    Output format is determined by the OUTPUT_FORMAT module constant:
    - "wav": Uncompressed PCM format
    - "m4a": AAC compressed format (8kbps)

    Database details:
    - Location: ./tmp/speech_sqlite/{show_name}_{format}.sqlite
    - Table: speech (id, start_ts, end_ts, audio_data, text)
    - Table: metadata (key, value) - stores version, audio_format, show_name, and database_id
    - Timestamps: ISO 8601 format with timezone
    - Audio data: Raw PCM for wav, raw AAC (ADTS) for m4a
    - Validation: On startup, verifies version, audio_format, and show_name in metadata match current settings

    Args:
        show_name: Name of the show (used for database filename)
        exception_handler: Optional handler to capture exceptions for main thread

    Returns:
        Callback function that accepts AudioSegment or TranscriptionNotice and saves it

    Raises:
        ValueError: If OUTPUT_FORMAT is unsupported
    """
    # Create database directory
    db_dir = "./tmp/speech_sqlite"
    os.makedirs(db_dir, exist_ok=True)

    # Database file path includes format to distinguish different encodings
    db_path = os.path.join(db_dir, f"{show_name}_{OUTPUT_FORMAT}.sqlite")

    # Initialize database and schema
    # Set timeout for database locks and enable WAL mode for concurrent access
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")

    # Enable WAL mode for better concurrent read/write access
    conn.execute('PRAGMA journal_mode=WAL')

    # Initialize metadata table and validate format and show name
    _initialize_metadata(conn, OUTPUT_FORMAT, show_name)

    # Create speech table
    conn.execute('''CREATE TABLE IF NOT EXISTS speech (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_ts TEXT NOT NULL,
        end_ts TEXT NOT NULL,
        audio_data BLOB,
        text TEXT,
        CHECK (audio_data IS NOT NULL OR text IS NOT NULL)
    )''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_start_ts ON speech(start_ts)')

    def _save_impl(segment: AudioSegment):
        audio = segment.audio

        # Calculate timestamps in ISO 8601 format
        if segment.wall_clock_start is not None:
            # Livestream mode: use wall clock timestamps
            start_dt = datetime.fromtimestamp(segment.wall_clock_start, timezone.utc)
            end_ts_float = segment.wall_clock_start + len(audio) / TARGET_SAMPLE_RATE
            end_dt = datetime.fromtimestamp(end_ts_float, timezone.utc)
        else:
            # Prerecorded mode: use relative timestamps (convert to datetime from start of epoch)
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

        elif OUTPUT_FORMAT == "m4a":
            # M4A mode: encode to raw AAC stream (ADTS format)
            audio_data = _encode_to_raw_aac(audio_int16)

        else:
            raise ValueError(f"Unsupported OUTPUT_FORMAT for SQLite: {OUTPUT_FORMAT}. Supported: 'wav', 'm4a'")

        # Insert into database
        try:
            conn.execute(
                'INSERT INTO speech (start_ts, end_ts, audio_data, text) VALUES (?, ?, ?, NULL)',
                (start_ts_str, end_ts_str, audio_data)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to save audio segment to SQLite: {e}")

    def _save(item):
        """Wrapper that captures exceptions for the exception handler.

        Accepts either AudioSegment or TranscriptionNotice.
        """
        try:
            if isinstance(item, AudioSegment):
                _save_impl(item)
            elif type(item).__name__ == 'TranscriptionNotice':
                _save_notice_to_sqlite(conn, item)
            else:
                raise TypeError(f"Expected AudioSegment or TranscriptionNotice, got {type(item)}")
        except Exception as e:
            if exception_handler:
                exception_handler.set_exception(e)
            raise

    return _save
