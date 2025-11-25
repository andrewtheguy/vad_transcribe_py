"""
Cross-platform file locking to prevent multiple instances of the program.

Supports both Unix (fcntl) and Windows (msvcrt) locking mechanisms.
"""

import os
import sys
import tempfile
import re
from pathlib import Path
from typing import Optional

# Platform-specific imports
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl


class LockError(Exception):
    """Raised when a lock cannot be acquired."""
    pass


class FileLock:
    """
    Context manager for exclusive file locking.

    Usage:
        with FileLock('mic_web') as lock:
            # Your code here
            pass
    """

    def __init__(self, lock_name: str, show_name: Optional[str] = None):
        """
        Initialize file lock.

        Args:
            lock_name: Type of lock ('file')
            show_name: Unused (kept for compatibility)
        """
        self.lock_name = lock_name
        self.show_name = show_name
        self.lock_file_path = self._get_lock_file_path()
        self.lock_file = None
        self.acquired = False

    def _get_lock_file_path(self) -> Path:
        """Generate lock file path based on lock type."""
        temp_dir = tempfile.gettempdir()

        # File transcription only
        filename = f"whisper_transcribe_{self.lock_name}.lock"
        return Path(temp_dir) / filename

    def _read_lock_pid(self) -> Optional[int]:
        """Read PID from lock file if it exists and is currently held."""
        try:
            if self.lock_file_path.exists():
                with open(self.lock_file_path, 'r') as f:
                    content = f.read().strip()
                    if content.isdigit():
                        return int(content)
        except Exception:
            pass
        return None

    def _acquire_lock_unix(self):
        """Acquire lock using fcntl (Unix/Linux/macOS)."""
        try:
            # Try to acquire exclusive non-blocking lock
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = True
        except (IOError, OSError):
            raise LockError(f"Lock already held by another process")

    def _acquire_lock_windows(self):
        """Acquire lock using msvcrt (Windows)."""
        try:
            # Try to lock the first byte of the file
            msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            self.acquired = True
        except (IOError, OSError):
            raise LockError(f"Lock already held by another process")

    def _get_action_description(self) -> str:
        """Get human-readable description of the action being locked."""
        if self.lock_name == 'file':
            return "file transcription"
        return self.lock_name

    def acquire(self):
        """
        Acquire the lock.

        The OS-level file lock (fcntl/msvcrt) is automatically released when a process
        exits or crashes, so we don't need to check for stale locks.
        """
        # Open lock file for writing
        try:
            self.lock_file = open(self.lock_file_path, 'w')
        except Exception as e:
            raise LockError(f"Failed to create lock file: {e}")

        # Try to acquire platform-specific lock
        try:
            if sys.platform == 'win32':
                self._acquire_lock_windows()
            else:
                self._acquire_lock_unix()

            # Write our PID to the lock file
            self.lock_file.write(str(os.getpid()))
            self.lock_file.flush()

        except LockError as e:
            # Failed to acquire lock
            self.lock_file.close()

            # Read the PID that holds the lock
            holding_pid = self._read_lock_pid()
            action_desc = self._get_action_description()

            error_msg = f"Cannot acquire lock for {action_desc}: another instance is already running"
            if holding_pid:
                error_msg += f" (PID: {holding_pid})"
            error_msg += f"\nLock file: {self.lock_file_path}"

            raise LockError(error_msg)

    def release(self):
        """Release the lock."""
        if self.lock_file:
            try:
                if sys.platform == 'win32' and self.acquired:
                    # Unlock on Windows
                    try:
                        msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception:
                        pass

                self.lock_file.close()

                # Remove lock file
                self.lock_file_path.unlink(missing_ok=True)

            except Exception:
                pass
            finally:
                self.lock_file = None
                self.acquired = False

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def acquire_lock(lock_type: str, show_name: Optional[str] = None) -> FileLock:
    """
    Convenience function to create and return a FileLock.

    Args:
        lock_type: Type of lock ('file')
        show_name: Unused (kept for compatibility)

    Returns:
        FileLock instance (use as context manager)

    Example:
        with acquire_lock('file'):
            # Your code here
            pass
    """
    return FileLock(lock_type, show_name)
