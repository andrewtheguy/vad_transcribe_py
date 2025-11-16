"""Audio data saving functionality for transcribed audio segments."""

import logging
import os
import sqlite3
import subprocess
import threading
import time
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

# Module-level setup for periodic backup
logger = logging.getLogger(__name__)
_backup_lock = threading.Lock()
_active_show_name = None
_active_db_path = None
_backup_thread = None
_backup_thread_lock = threading.Lock()


class NonBlockingLock:
    """Context manager for non-blocking lock acquisition."""

    def __init__(self, lock: threading.Lock):
        self.lock = lock
        self.acquired = False

    def __enter__(self) -> bool:
        """Try to acquire lock without blocking. Returns True if acquired."""
        self.acquired = self.lock.acquire(blocking=False)
        return self.acquired

    def __exit__(self, *args):
        """Release lock if it was acquired."""
        if self.acquired:
            self.lock.release()


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


def _create_db_connection(db_path: str) -> sqlite3.Connection:
    """Create a configured SQLite connection for the database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Configured SQLite connection with WAL mode and timeouts
    """
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


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

    # Initialize database connection with WAL mode and timeouts
    conn = _create_db_connection(db_path)

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

    # Register database for periodic backup
    global _active_show_name, _active_db_path
    _active_show_name = show_name
    _active_db_path = db_path
    _ensure_backup_thread_started()

    return _save

def periodic_backup(show_name: str, db_path: str) -> None:
    """
    Backup a specific SQLite database and upload to remote storage.

    Creates a new database connection in the backup thread to avoid
    SQLite thread-safety issues.

    Process:
    1. Create backup using SQLite's backup API
    2. Determine max id from the backup file (ensures consistent snapshot)
    3. Upload backup to remote using rclone
    4. If upload succeeds, delete old records from source database up to the
       max id that was actually backed up

    Args:
        show_name: Name of the show (used for organizing backups)
        db_path: Path to the SQLite database file
    """
    # Check if periodic backup is enabled
    if os.getenv("PERIODIC_UPLOAD_ENABLED") != "yes":
        logger.debug(f"Skipping backup for {show_name} - periodic backup disabled")
        return

    # Get destination directory from environment
    dest_dir = os.getenv("SQLITE_BACKUP_DEST_DIR")
    if not dest_dir:
        logger.error("SQLITE_BACKUP_DEST_DIR not set in environment")
        return

    # Try to acquire lock (non-blocking)
    with NonBlockingLock(_backup_lock) as lock_acquired:
        if not lock_acquired:
            logger.debug(f"Skipping backup for {show_name} - another backup in progress")
            return

        logger.info(f"Starting backup for {show_name}")

        # Create a new connection for this thread (SQLite thread-safety requirement)
        db_conn = None
        try:
            db_conn = _create_db_connection(db_path)
        except Exception as e:
            logger.error(f"Failed to create database connection for {show_name}: {e}")
            return

        # Step 1: Create backup file
        backup_dir = f"./tmp/speech_sqlite_backup/{show_name}"
        os.makedirs(backup_dir, exist_ok=True)

        now = datetime.now(timezone.utc)

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{show_name}_{timestamp}.sqlite"
        backup_path = os.path.join(backup_dir, backup_filename)

        backup_conn = None

        try:
            # Create backup connection
            backup_conn = sqlite3.connect(backup_path)

            # Use SQLite's backup API to copy database
            db_conn.backup(backup_conn)

            backup_conn.close()
            backup_conn = None

            logger.info(f"Backup created: {backup_path}")

        except Exception as e:
            logger.error(f"Failed to create backup for {show_name}: {e}")
            if backup_conn:
                backup_conn.close()
            if db_conn:
                db_conn.close()
            # Clean up partial backup file
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except:
                    pass
            return

        # Step 2: Determine max id from backup to avoid races with new inserts
        backup_max_id = None
        backup_max_id_conn = None
        try:
            backup_max_id_conn = sqlite3.connect(backup_path)
            cursor = backup_max_id_conn.execute("SELECT MAX(id) FROM speech")
            result = cursor.fetchone()
            backup_max_id = result[0] if result and result[0] is not None else None
            if backup_max_id is None:
                logger.info(f"No records to backup for {show_name}")
                if db_conn:
                    db_conn.close()
                # Remove empty backup file
                try:
                    os.remove(backup_path)
                except OSError:
                    pass
                return
            logger.info(f"Max ID in backup for {show_name}: {backup_max_id}")
        except Exception as e:
            logger.error(f"Failed to read max id from backup for {show_name}: {e}")
            if backup_max_id_conn:
                backup_max_id_conn.close()
            if db_conn:
                db_conn.close()
            return
        finally:
            if backup_max_id_conn:
                backup_max_id_conn.close()

        # Step 3: Upload with rclone
        remote = "remote"
        remote_dest = f"{remote}:{dest_dir}/{show_name}/{now.strftime('%Y/%m/%d')}/"

        try:
            result = subprocess.run(
                ['rclone', '--config', '/notfound', '-v', 'move', backup_dir, remote_dest],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"rclone upload failed for {show_name}: {result.stderr}")
                # Keep backup file for retry
                if db_conn:
                    db_conn.close()
                return

            logger.info(f"Upload success for {show_name}: {backup_filename}")

        except subprocess.TimeoutExpired:
            logger.error(f"rclone upload timed out for {show_name}")
            if db_conn:
                db_conn.close()
            return
        except Exception as e:
            logger.error(f"rclone upload error for {show_name}: {e}")
            if db_conn:
                db_conn.close()
            return

        # Step 4: Delete old records from source database (only if upload succeeded)
        try:
            cursor = db_conn.execute("DELETE FROM speech WHERE id <= ?", (backup_max_id,))
            deleted_count = cursor.rowcount
            db_conn.commit()

            logger.info(
                f"Deleted {deleted_count} records from {show_name} (id <= {backup_max_id})"
            )

        except Exception as e:
            logger.error(f"Failed to delete old records from {show_name}: {e}")
            # Don't fail the backup - records will accumulate and get backed up next time
        finally:
            # Always close the connection created in this thread
            if db_conn:
                db_conn.close()


def _test_remote_connection() -> bool:
    """Test rclone remote connection.

    Returns:
        True if connection test succeeds, False otherwise
    """
    try:
        # Test connection by listing remote directory
        result = subprocess.run(
            ['rclone', '--config', '/notfound', 'lsd', 'remote:'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            logger.error(f"Remote connection test failed: {result.stderr}")
            return False

        logger.info("Remote connection test successful")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Remote connection test timed out")
        return False
    except Exception as e:
        logger.error(f"Remote connection test error: {e}")
        return False


def _backup_scheduler_thread():
    """Background thread that runs periodic backups at configured interval."""
    # Get backup interval from environment variable (default: 3600 seconds = 1 hour)
    try:
        interval = int(os.getenv("SQLITE_BACKUP_INTERVAL_SECONDS", "3600"))
        if interval < 1:
            logger.warning(f"Invalid SQLITE_BACKUP_INTERVAL_SECONDS={interval}, using default 3600")
            interval = 3600
    except ValueError:
        logger.warning(f"Invalid SQLITE_BACKUP_INTERVAL_SECONDS, using default 3600")
        interval = 3600

    logger.info(f"Backup scheduler thread started (interval: {interval}s)")

    while True:
        try:
            time.sleep(interval)

            # Check if backups are enabled
            if os.getenv("PERIODIC_UPLOAD_ENABLED") != "yes":
                continue

            # Backup the active database if one is registered
            if _active_show_name and _active_db_path:
                try:
                    # Check if database file still exists
                    if not os.path.exists(_active_db_path):
                        logger.warning(f"Database no longer exists: {_active_db_path}")
                        continue

                    periodic_backup(_active_show_name, _active_db_path)

                except Exception as e:
                    logger.error(f"Backup failed for {_active_show_name}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Backup scheduler error: {e}", exc_info=True)


def _ensure_backup_thread_started():
    """Ensure the backup scheduler thread is running (start it if not).

    Tests configuration and remote connection before starting the thread to fail fast.

    Raises:
        RuntimeError: If backup is enabled but configuration is invalid or remote connection fails
    """
    global _backup_thread

    # Check if backups are enabled
    if os.getenv("PERIODIC_UPLOAD_ENABLED") != "yes":
        return

    with _backup_thread_lock:
        if _backup_thread is None or not _backup_thread.is_alive():
            # Check required environment variables (fail fast)
            dest_dir = os.getenv("SQLITE_BACKUP_DEST_DIR")
            if not dest_dir:
                raise RuntimeError(
                    "SQLITE_BACKUP_DEST_DIR environment variable is not set. "
                    "Please set it to the remote backup destination directory, "
                    "or set PERIODIC_UPLOAD_ENABLED=no to disable backups."
                )

            # Test remote connection before starting backup thread (fail fast)
            logger.info("Testing remote connection before starting backup thread...")
            if not _test_remote_connection():
                raise RuntimeError(
                    "Failed to connect to remote storage. "
                    "Please check rclone configuration and remote connectivity. "
                    "Set PERIODIC_UPLOAD_ENABLED=no to disable backups."
                )

            _backup_thread = threading.Thread(
                target=_backup_scheduler_thread,
                daemon=True,
                name="backup-scheduler"
            )
            _backup_thread.start()
            logger.info("Started backup scheduler thread")
