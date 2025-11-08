"""Database helpers shared between CLI and web server."""

from __future__ import annotations

import os
import psycopg

from whisper_transcribe_py.speech_detector import TranscribedSegment


def connect_to_database():
    """Create a psycopg connection using env vars.

    Raises:
        RuntimeError: If DATABASE_URL is missing.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is required for database operations")
    timeout = int(os.environ.get("DATABASE_TIMEOUT", "10"))
    return psycopg.connect(db_url, autocommit=True, connect_timeout=timeout)


def build_database_writer(conn, show_name: str):
    """Return a callback that persists transcribed segments."""

    def _persist(segment: TranscribedSegment):
        if segment.start_timestamp is None:
            raise ValueError("Database writes require wall clock timestamps.")
        with conn.cursor() as cur:
            cur.execute(
                '''INSERT INTO transcripts (show_name,"timestamp", content)
                   VALUES (%s, %s, %s)
                   RETURNING id''',
                (
                    show_name,
                    segment.start_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    segment.text,
                ),
            )
            row = cur.fetchone()
        return row[0] if row else None

    return _persist
