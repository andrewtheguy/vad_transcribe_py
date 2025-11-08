"""Utilities for handling lightweight streaming transcription sessions."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple

import numpy.typing as npt

from whisper_transcribe_py.speech_detector import (
    AudioSegment,
    SpeechDetector,
    TranscriptPersistenceCallback,
)

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base class for streaming session errors."""


class SessionRevokedError(SessionError):
    """Raised when an API call references a revoked session."""


@dataclass
class StreamingSession:
    """Holds the state required to stream audio into a SpeechDetector."""

    session_id: str
    language: str
    input_sample_rate: int
    queue: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    detector: SpeechDetector | None = None
    thread: threading.Thread | None = None
    wall_clock_reference: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    transcript_persistence_callback: Optional[TranscriptPersistenceCallback] = None
    persistence_cleanup: Optional[Callable[[], None]] = None
    first_transcript_id: Optional[int] = None

    def ensure_running(self) -> None:
        if self.detector is not None:
            return
        self.detector = SpeechDetector(
            audio_input_queue=self.queue,
            language=self.language,
            timestamp_strategy="wall_clock",
            stop_event=self.stop_event,
            wall_clock_reference=self.wall_clock_reference,
            transcript_persistence_callback=self.transcript_persistence_callback,
        )
        self.thread = threading.Thread(
            target=self.detector.process_input,
            args=(self.input_sample_rate,),
            daemon=True,
        )
        self.thread.start()

    def enqueue(self, audio: npt.NDArray, start_ts: float, approx_wall_clock: Optional[float]) -> None:
        self.last_activity = time.time()
        if approx_wall_clock is not None and self.detector is not None and self.detector.wall_clock_reference is None:
            self.detector.wall_clock_reference = approx_wall_clock
        elif approx_wall_clock is not None and self.detector is None:
            self.wall_clock_reference = approx_wall_clock

        self.queue.put(AudioSegment(audio=audio, start=start_ts))

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.queue.put(None)
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.persistence_cleanup is not None:
            try:
                self.persistence_cleanup()
            finally:
                self.persistence_cleanup = None


PersistenceFactory = Callable[
    [],
    Tuple[Optional[TranscriptPersistenceCallback], Optional[Callable[[], None]]],
]


class StreamingSessionManager:
    """Tracks active streaming sessions."""

    def __init__(self, persistence_factory: Optional[PersistenceFactory] = None) -> None:
        self._sessions: Dict[str, StreamingSession] = {}
        self._lock = threading.Lock()
        self._active_session_id: Optional[str] = None
        self._revoked_session_ids: Set[str] = set()
        self._persistence_factory = persistence_factory

    def set_persistence_factory(self, factory: Optional[PersistenceFactory]) -> None:
        with self._lock:
            self._persistence_factory = factory

    def _create_session(self, *, session_id: str, language: str, sample_rate: int) -> StreamingSession:
        persistence_callback = None
        cleanup_callback = None
        if self._persistence_factory is not None:
            try:
                persistence_callback, cleanup_callback = self._persistence_factory()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to initialize persistence for session %s: %s", session_id, exc, exc_info=True)

        session = StreamingSession(
            session_id=session_id,
            language=language,
            input_sample_rate=sample_rate,
            persistence_cleanup=cleanup_callback,
        )

        if persistence_callback is not None:
            def _wrapped(segment, *, _session=session, _writer=persistence_callback):
                inserted_id = _writer(segment)
                if inserted_id is not None and _session.first_transcript_id is None:
                    _session.first_transcript_id = inserted_id
                return inserted_id

            session.transcript_persistence_callback = _wrapped

        self._sessions[session_id] = session
        session.ensure_running()
        return session

    def get_or_create(self, *, session_id: str, language: str, sample_rate: int) -> StreamingSession:
        with self._lock:
            if self._active_session_id is None:
                self._active_session_id = session_id
                self._revoked_session_ids.discard(session_id)
                return self._create_session(session_id=session_id, language=language, sample_rate=sample_rate)

            if session_id == self._active_session_id:
                session = self._sessions.get(session_id)
                if session is None:
                    session = self._create_session(session_id=session_id, language=language, sample_rate=sample_rate)
                return session

            if session_id in self._revoked_session_ids:
                raise SessionRevokedError(f"Session '{session_id}' is no longer active")

            # Accept the new session id and revoke the previous one.
            previous_session_id = self._active_session_id
            previous_session = self._sessions.pop(previous_session_id, None)
            if previous_session is not None:
                previous_session.close()
            self._revoked_session_ids.add(previous_session_id)

            # Avoid unbounded growth.
            if len(self._revoked_session_ids) > 16:
                self._revoked_session_ids.pop()

            self._active_session_id = session_id
            self._revoked_session_ids.discard(session_id)
            return self._create_session(session_id=session_id, language=language, sample_rate=sample_rate)

    def stop(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session_id == self._active_session_id:
                self._active_session_id = None
            if session_id in self._revoked_session_ids:
                self._revoked_session_ids.remove(session_id)
        if session is None:
            return False
        session.close()
        self._revoked_session_ids.add(session_id)
        return True

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def cleanup_inactive(self, ttl_seconds: float = 300.0) -> None:
        """Remove idle sessions to avoid leaking resources."""
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_activity > ttl_seconds:
                    expired.append(session_id)
            for session_id in expired:
                session = self._sessions.pop(session_id, None)
                if session is not None:
                    session.close()
                if session_id == self._active_session_id:
                    self._active_session_id = None
                self._revoked_session_ids.add(session_id)


streaming_sessions = StreamingSessionManager()
