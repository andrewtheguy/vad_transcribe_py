"""Utilities for handling lightweight streaming transcription sessions."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set, Tuple

import numpy.typing as npt

from whisper_transcribe_py.audio_transcriber import (
    AudioSegment,
    AudioTranscriber,
    QueueBacklogLimiter,
    TranscriptionCallback,
)

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base class for streaming session errors."""


class SessionRevokedError(SessionError):
    """Raised when an API call references a revoked session."""

# Queue backlog limit must be at least 2x the max speech segment duration (default 60s)
DEFAULT_SESSION_QUEUE_TIME_LIMIT_SECONDS = 120.0


@dataclass
class StreamingSession:
    """Holds the state required to stream audio into an AudioTranscriber."""

    session_id: str
    language: str
    input_sample_rate: int
    queue: queue.Queue = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    detector: AudioTranscriber | None = None
    thread: threading.Thread | None = None
    wall_clock_reference: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    transcription_callback: Optional[TranscriptionCallback] = None
    persistence_cleanup: Optional[Callable[[], None]] = None
    first_transcript_id: Optional[int] = None
    n_threads: int = 1
    queue_limiter: Optional[QueueBacklogLimiter] = None

    def ensure_running(self) -> None:
        if self.detector is not None:
            return
        self.detector = AudioTranscriber(
            audio_input_queue=self.queue,
            language=self.language,
            mode='livestream',  # Web API uses livestream mode
            stop_event=self.stop_event,
            wall_clock_reference=self.wall_clock_reference,
            transcription_callback=self.transcription_callback,
            queue_backlog_limiter=self.queue_limiter,
            n_threads=self.n_threads,
        )
        self.thread = threading.Thread(
            target=self.detector.process_input,
            args=(self.input_sample_rate,),
            daemon=True,
        )
        self.thread.start()

    def enqueue(self, audio: npt.NDArray, start_ts: float, approx_wall_clock: Optional[float]) -> None:
        self.last_activity = time.time()

        # Fail fast if wall_clock is missing - all callers must provide it
        if approx_wall_clock is None:
            raise ValueError(
                "approx_wall_clock is required for all audio segments. "
                "All streaming API calls must provide wall clock timestamps."
            )

        if self.detector is not None and self.detector.wall_clock_reference is None:
            self.detector.wall_clock_reference = approx_wall_clock
        elif self.detector is None:
            self.wall_clock_reference = approx_wall_clock

        duration_seconds = len(audio) / self.input_sample_rate if self.input_sample_rate else 0.0
        if self.queue_limiter and duration_seconds > 0:
            if not self.queue_limiter.try_add(duration_seconds, chunk_wall_clock=approx_wall_clock):
                logger.warning(
                    "Dropped %.2fs chunk for session %s because backlog exceeded %.0fs cap",
                    duration_seconds,
                    self.session_id,
                    self.queue_limiter.max_seconds if self.queue_limiter.max_seconds is not None else 0.0,
                )
                return

        self.queue.put(AudioSegment(audio=audio, start=start_ts, wall_clock_start=approx_wall_clock))

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
    Tuple[Optional[TranscriptionCallback], Optional[Callable[[], None]]],
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

        limiter = None
        if DEFAULT_SESSION_QUEUE_TIME_LIMIT_SECONDS and DEFAULT_SESSION_QUEUE_TIME_LIMIT_SECONDS > 0:
            limiter = QueueBacklogLimiter(
                DEFAULT_SESSION_QUEUE_TIME_LIMIT_SECONDS,
                source_label=f"session:{session_id}",
            )

        session = StreamingSession(
            session_id=session_id,
            language=language,
            input_sample_rate=sample_rate,
            persistence_cleanup=cleanup_callback,
            queue_limiter=limiter,
        )

        if persistence_callback is not None:
            def _wrapped(segments, *, _session=session, _writer=persistence_callback):
                for segment in segments:
                    inserted_id = _writer(segment)
                    if inserted_id is not None and _session.first_transcript_id is None:
                        _session.first_transcript_id = inserted_id

            session.transcription_callback = _wrapped

        self._sessions[session_id] = session
        session.ensure_running()
        return session

    def start_new_session(self, *, session_id: str, language: str, sample_rate: int) -> StreamingSession:
        previous_session: Optional[StreamingSession] = None
        with self._lock:
            if self._active_session_id is not None and self._active_session_id in self._sessions:
                previous_session = self._sessions.pop(self._active_session_id)
                self._revoked_session_ids.add(self._active_session_id)
            self._active_session_id = session_id
            self._revoked_session_ids.discard(session_id)
            session = self._create_session(session_id=session_id, language=language, sample_rate=sample_rate)
        if previous_session is not None:
            previous_session.close()
        return session

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
