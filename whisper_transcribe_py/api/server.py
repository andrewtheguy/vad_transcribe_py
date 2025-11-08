"""FastAPI server for Whisper Transcribe web interface."""

import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from whisper_transcribe_py.api.streaming import SessionRevokedError, streaming_sessions
from whisper_transcribe_py.speech_detector import pcm_s16le_to_float32


def create_app(dev_mode: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        dev_mode: If True, enables CORS for development with Vite dev server

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Whisper Transcribe API",
        description="AI-powered speech transcription with voice activity detection",
        version="0.1.0",
    )

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

    uvicorn.run(
        "whisper_transcribe_py.api.server:create_app",
        host=host,
        port=port,
        reload=dev,
        factory=True,
        log_level="info" if dev else "warning",
    )
