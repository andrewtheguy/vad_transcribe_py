"""FastAPI server for Whisper Transcribe web interface."""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse


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

    @app.websocket("/api/transcribe/stream")
    async def transcribe_stream():
        """Placeholder: WebSocket endpoint for streaming audio transcription."""
        raise HTTPException(
            status_code=501,
            detail="Streaming transcription not yet implemented"
        )

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
