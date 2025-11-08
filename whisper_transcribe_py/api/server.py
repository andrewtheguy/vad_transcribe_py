"""FastAPI server for Whisper Transcribe web interface."""

import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from whisper_transcribe_py.api.streaming import SessionRevokedError, streaming_sessions
from whisper_transcribe_py.db import build_database_writer, connect_to_database
from whisper_transcribe_py.speech_detector import pcm_s16le_to_float32

logger = logging.getLogger(__name__)


def create_app(dev_mode: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        dev_mode: If True, enables CORS for development with Vite dev server

    Returns:
        Configured FastAPI application
    """
    # Check environment variable for dev mode (used when factory=True in uvicorn)
    if not dev_mode:
        dev_mode = os.environ.get("WHISPER_DEV_MODE", "").lower() == "true"

    app = FastAPI(
        title="Whisper Transcribe API",
        description="AI-powered speech transcription with voice activity detection",
        version="0.1.0",
    )

    # Configure persistence factory (database writes)
    streaming_sessions.set_persistence_factory(_build_persistence_factory())

    # CORS middleware for development
    if dev_mode:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],  # Vite dev server
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Health check endpoint
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "service": "whisper-transcribe"}

    # Placeholder API endpoints for future implementation
    @app.post("/api/transcribe/file")
    async def transcribe_file():
        """Placeholder: Upload and transcribe audio file."""
        raise HTTPException(
            status_code=501,
            detail="File transcription not yet implemented"
        )

    @app.post("/api/transcribe/stream")
    async def ingest_audio_chunk(
        request: Request,
        session_id: str = Query(..., description="Client provided session identifier"),
        start: float = Query(..., description="Relative start timestamp (seconds) for this chunk"),
        sample_rate: int = Query(16000, gt=0, description="Sample rate of the provided audio"),
        language: str = Query("en", description="Language code for transcription"),
    ):
        """Accept a small PCM chunk and enqueue it for transcription."""
        payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="Audio chunk is empty")
        if len(payload) % 2 != 0:
            raise HTTPException(
                status_code=400,
                detail="Audio chunk length must align to 16-bit samples",
            )

        try:
            session = streaming_sessions.get_or_create(
                session_id=session_id,
                language=language,
                sample_rate=sample_rate,
            )
        except SessionRevokedError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        audio = pcm_s16le_to_float32(payload)
        approx_reference = max(time.time() - max(start, 0.0), 0.0)
        session.enqueue(audio=audio, start_ts=start, approx_wall_clock=approx_reference)

        return {
            "status": "queued",
            "session_id": session_id,
            "samples": len(audio),
        }

    @app.delete("/api/transcribe/stream/{session_id}")
    async def stop_stream(session_id: str):
        """Stop an active streaming transcription session."""
        if not streaming_sessions.stop(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "stopped", "session_id": session_id}

    @app.get("/api/transcribe/stream/{session_id}/transcripts")
    async def fetch_transcripts(
        session_id: str,
        limit: int = Query(1000, ge=1, le=2000, description="Max number of transcripts to return"),
    ):
        """Return up to `limit` transcripts; if more exist, return the latest ones for the session."""
        session = streaming_sessions.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or no longer active")

        if not os.environ.get("DATABASE_URL"):
            raise HTTPException(status_code=503, detail="Database not configured")

        if session.first_transcript_id is None:
            return {"first_id": None, "transcripts": []}

        latest_id = None
        start_id = session.first_transcript_id

        try:
            with connect_to_database() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT MAX(id) FROM transcripts
                        WHERE show_name = %s AND id >= %s
                        ''',
                        ("web_recording", session.first_transcript_id),
                    )
                    max_row = cur.fetchone()
                    if not max_row or max_row[0] is None:
                        return {"first_id": session.first_transcript_id, "transcripts": []}

                    latest_id = max_row[0]
                    start_id = max(session.first_transcript_id, latest_id - limit + 1)

                    cur.execute(
                        '''
                        SELECT id, "timestamp", content
                        FROM transcripts
                        WHERE show_name = %s AND id BETWEEN %s AND %s
                        ORDER BY id ASC
                        ''',
                        ("web_recording", start_id, latest_id),
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            logger.error("Failed to fetch transcripts for session %s: %s", session_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to fetch transcripts") from exc

        return {
            "first_id": session.first_transcript_id,
            "latest_id": latest_id,
            "start_id": start_id,
            "transcripts": [
                {"id": row[0], "timestamp": row[1].isoformat(), "content": row[2]}
                for row in rows
            ],
        }

    @app.get("/api/sessions")
    async def list_sessions():
        """Placeholder: Get list of transcription sessions."""
        return JSONResponse(
            content={
                "sessions": [],
                "message": "Session management not yet implemented"
            },
            status_code=200
        )

    @app.get("/api/shows")
    async def list_shows():
        """List all distinct show names with metadata."""
        if not os.environ.get("DATABASE_URL"):
            raise HTTPException(status_code=503, detail="Database not configured")

        try:
            with connect_to_database() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        '''
                        SELECT
                            show_name,
                            COUNT(*) as transcript_count,
                            MAX("timestamp") as latest_timestamp,
                            MIN("timestamp") as earliest_timestamp
                        FROM transcripts
                        GROUP BY show_name
                        ORDER BY MAX("timestamp") DESC
                        '''
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            logger.error("Failed to fetch shows: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to fetch shows") from exc

        return {
            "shows": [
                {
                    "name": row[0],
                    "transcript_count": row[1],
                    "latest_timestamp": row[2].isoformat() if row[2] else None,
                    "earliest_timestamp": row[3].isoformat() if row[3] else None,
                }
                for row in rows
            ]
        }

    @app.get("/api/shows/{show_name}/transcripts")
    async def fetch_show_transcripts(
        show_name: str,
        offset: int = Query(0, ge=0, description="Number of transcripts to skip"),
        limit: int = Query(50, ge=1, le=200, description="Max number of transcripts to return"),
    ):
        """Fetch transcripts for a specific show in reverse chronological order."""
        if not os.environ.get("DATABASE_URL"):
            raise HTTPException(status_code=503, detail="Database not configured")

        try:
            with connect_to_database() as conn:
                with conn.cursor() as cur:
                    # Get total count
                    cur.execute(
                        'SELECT COUNT(*) FROM transcripts WHERE show_name = %s',
                        (show_name,)
                    )
                    total_count = cur.fetchone()[0]

                    if total_count == 0:
                        return {
                            "show_name": show_name,
                            "total": 0,
                            "offset": offset,
                            "limit": limit,
                            "transcripts": []
                        }

                    # Fetch transcripts in reverse chronological order
                    cur.execute(
                        '''
                        SELECT id, "timestamp", content
                        FROM transcripts
                        WHERE show_name = %s
                        ORDER BY "timestamp" DESC, id DESC
                        LIMIT %s OFFSET %s
                        ''',
                        (show_name, limit, offset)
                    )
                    rows = cur.fetchall()
        except Exception as exc:
            logger.error("Failed to fetch transcripts for show %s: %s", show_name, exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to fetch transcripts") from exc

        return {
            "show_name": show_name,
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "transcripts": [
                {"id": row[0], "timestamp": row[1].isoformat(), "content": row[2]}
                for row in rows
            ],
        }

    # Serve static files in production mode
    if not dev_mode:
        # Get the frontend build directory
        project_root = Path(__file__).parent.parent.parent
        frontend_dist = project_root / "frontend" / "dist"

        if frontend_dist.exists():
            # Serve static assets
            app.mount(
                "/assets",
                StaticFiles(directory=frontend_dist / "assets"),
                name="assets"
            )

            # Serve index.html for all non-API routes (SPA routing)
            @app.get("/{full_path:path}")
            async def serve_frontend(full_path: str):
                """Serve the frontend application."""
                # Don't serve frontend for API routes
                if full_path.startswith("api/"):
                    raise HTTPException(status_code=404, detail="Not found")

                # Serve index.html for all other routes
                index_file = frontend_dist / "index.html"
                if index_file.exists():
                    return FileResponse(index_file)
                else:
                    raise HTTPException(
                        status_code=500,
                        detail="Frontend not built. Run 'npm run build' in frontend directory."
                    )
        else:
            @app.get("/")
            async def no_frontend():
                """Inform user that frontend is not built."""
                return {
                    "error": "Frontend not built",
                    "message": "Please run 'npm run build' in the frontend directory",
                    "path": str(frontend_dist)
                }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, dev: bool = False):
    """Run the FastAPI server with uvicorn.

    Args:
        host: Host to bind to
        port: Port to bind to
        dev: Enable development mode with hot reload and CORS
    """
    import uvicorn

    if dev:
        # Set environment variable so create_app knows we're in dev mode
        os.environ["WHISPER_DEV_MODE"] = "true"

        # Use import string for reload to work
        uvicorn.run(
            "whisper_transcribe_py.api.server:create_app",
            host=host,
            port=port,
            reload=True,
            factory=True,
            log_level="info",
        )
    else:
        # Production mode: create app directly
        app = create_app(dev_mode=False)
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
        )


def _build_persistence_factory():
    """Create a factory that provides database writers for streaming sessions."""
    if not os.environ.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set; streaming transcripts will not be persisted")
        return None

    def factory():
        try:
            conn = connect_to_database()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to connect to database for streaming session: %s", exc, exc_info=True)
            return None, None

        writer = build_database_writer(conn, show_name="web_recording")

        def cleanup():
            conn.close()

        return writer, cleanup

    return factory
