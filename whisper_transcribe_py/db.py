"""Database helpers shared between CLI and web server."""

from __future__ import annotations

import os
import psycopg

from whisper_transcribe_py.audio_transcriber import TranscribedSegment


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


def initialize_database_schema(conn):
    """Initialize the database schema if it doesn't exist.

    Args:
        conn: psycopg connection object
    """
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id bigserial PRIMARY KEY,
            show_name varchar(255) NOT NULL,
            start_timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            end_timestamp TIMESTAMP WITHOUT TIME ZONE,
            content TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS transcript_show_name_idx ON transcripts (show_name);
        """)


def build_database_writer(conn, show_name: str):
    """Return a callback that persists transcribed segments."""

    def _persist(segment: TranscribedSegment):
        if segment.start_timestamp is None:
            raise ValueError("Database writes require wall clock timestamps.")
        with conn.cursor() as cur:
            cur.execute(
                '''INSERT INTO transcripts (show_name, start_timestamp, end_timestamp, content)
                   VALUES (%s, %s, %s, %s)
                   RETURNING id''',
                (
                    show_name,
                    segment.start_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    segment.end_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f') if segment.end_timestamp else None,
                    segment.text,
                ),
            )
            row = cur.fetchone()
        return row[0] if row else None

    return _persist
