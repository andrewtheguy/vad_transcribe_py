"""
Unit tests for FastAPI server, including view-only mode functionality.

Tests cover:
- Transcription configuration endpoint
- View-only mode behavior (503 responses)
- Normal mode functionality
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from whisper_transcribe_py.api.server import create_app


class TestTranscriptionConfig:
    """Tests for /api/transcribe/config endpoint."""

    def test_config_with_transcription_enabled(self):
        """Test config endpoint returns transcription_enabled=True by default."""
        app = create_app(dev_mode=False, no_transcribe=False)
        client = TestClient(app)

        response = client.get("/api/transcribe/config")

        assert response.status_code == 200
        data = response.json()
        assert data["transcription_enabled"] is True
        assert data["alternate_api_url"] is None

    def test_config_with_transcription_disabled(self):
        """Test config endpoint returns transcription_enabled=False when disabled."""
        app = create_app(dev_mode=False, no_transcribe=True)
        client = TestClient(app)

        response = client.get("/api/transcribe/config")

        assert response.status_code == 200
        data = response.json()
        assert data["transcription_enabled"] is False
        assert data["alternate_api_url"] is None

    def test_config_with_alternate_api_url(self):
        """Test config endpoint returns alternate_api_url when provided."""
        app = create_app(
            dev_mode=False,
            no_transcribe=True,
            transcribe_api_url="https://example.com/transcribe"
        )
        client = TestClient(app)

        response = client.get("/api/transcribe/config")

        assert response.status_code == 200
        data = response.json()
        assert data["transcription_enabled"] is False
        assert data["alternate_api_url"] == "https://example.com/transcribe"


class TestViewOnlyMode:
    """Tests for view-only mode (--no-transcribe flag)."""

    @pytest.fixture
    def app_view_only(self):
        """Create app in view-only mode."""
        return create_app(dev_mode=False, no_transcribe=True)

    @pytest.fixture
    def client_view_only(self, app_view_only):
        """Create test client for view-only app."""
        return TestClient(app_view_only)

    def test_health_endpoint_works_in_view_only(self, client_view_only):
        """Test health endpoint works in view-only mode."""
        response = client_view_only.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "whisper-transcribe"}

    def test_file_transcribe_disabled_in_view_only(self, client_view_only):
        """Test /api/transcribe/file returns 503 in view-only mode."""
        response = client_view_only.post("/api/transcribe/file")

        assert response.status_code == 503
        assert "view-only mode" in response.json()["detail"].lower()

    def test_start_session_disabled_in_view_only(self, client_view_only):
        """Test /api/transcribe/stream/session returns 503 in view-only mode."""
        response = client_view_only.post(
            "/api/transcribe/stream/session",
            json={"language": "en", "sample_rate": 16000}
        )

        assert response.status_code == 503
        assert "view-only mode" in response.json()["detail"].lower()

    def test_ingest_audio_disabled_in_view_only(self, client_view_only):
        """Test /api/transcribe/stream returns 503 in view-only mode."""
        audio_data = b'\x00' * 1024  # Mock PCM audio data

        response = client_view_only.post(
            "/api/transcribe/stream?session_id=test-id&start=0.0",
            content=audio_data,
            headers={"Content-Type": "application/octet-stream"}
        )

        assert response.status_code == 503
        assert "view-only mode" in response.json()["detail"].lower()

    def test_stop_stream_disabled_in_view_only(self, client_view_only):
        """Test DELETE /api/transcribe/stream/{session_id} returns 503 in view-only mode."""
        response = client_view_only.delete("/api/transcribe/stream/test-session-id")

        assert response.status_code == 503
        assert "view-only mode" in response.json()["detail"].lower()

    def test_fetch_session_transcripts_disabled_in_view_only(self, client_view_only):
        """Test /api/transcribe/stream/{session_id}/transcripts returns 503 in view-only mode."""
        response = client_view_only.get("/api/transcribe/stream/test-session-id/transcripts")

        assert response.status_code == 503
        assert "view-only mode" in response.json()["detail"].lower()

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"})
    @patch("whisper_transcribe_py.api.server.connect_to_database")
    def test_shows_endpoint_works_in_view_only(self, mock_connect, client_view_only):
        """Test /api/shows endpoint works in view-only mode (read-only)."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn

        response = client_view_only.get("/api/shows")

        assert response.status_code == 200
        assert "shows" in response.json()

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost/test"})
    @patch("whisper_transcribe_py.api.server.connect_to_database")
    def test_show_transcripts_works_in_view_only(self, mock_connect, client_view_only):
        """Test /api/shows/{show_name}/transcripts works in view-only mode (read-only)."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (0,)  # total count
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value.__enter__.return_value = mock_conn

        response = client_view_only.get("/api/shows/test-show/transcripts")

        assert response.status_code == 200
        data = response.json()
        assert data["show_name"] == "test-show"
        assert "transcripts" in data


class TestNormalMode:
    """Tests for normal mode (transcription enabled)."""

    @pytest.fixture
    def app_normal(self):
        """Create app in normal mode."""
        return create_app(dev_mode=False, no_transcribe=False)

    @pytest.fixture
    def client_normal(self, app_normal):
        """Create test client for normal mode app."""
        return TestClient(app_normal)

    def test_health_endpoint_works(self, client_normal):
        """Test health endpoint works in normal mode."""
        response = client_normal.get("/api/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "whisper-transcribe"}

    def test_file_transcribe_returns_501_not_implemented(self, client_normal):
        """Test /api/transcribe/file returns 501 (not implemented) in normal mode."""
        response = client_normal.post("/api/transcribe/file")

        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"].lower()

    @patch("whisper_transcribe_py.api.server.streaming_sessions")
    def test_start_session_works_in_normal_mode(self, mock_sessions, client_normal):
        """Test /api/transcribe/stream/session works in normal mode."""
        # Mock the streaming_sessions manager
        mock_sessions.start_new_session = MagicMock()

        response = client_normal.post(
            "/api/transcribe/stream/session",
            json={"language": "en", "sample_rate": 16000}
        )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        mock_sessions.start_new_session.assert_called_once()

    @patch("whisper_transcribe_py.api.server.streaming_sessions")
    def test_ingest_audio_works_in_normal_mode(self, mock_sessions, client_normal):
        """Test /api/transcribe/stream works in normal mode when session exists."""
        # Mock session
        mock_session = MagicMock()
        mock_session.enqueue = MagicMock()
        mock_sessions.get_session.return_value = mock_session

        audio_data = b'\x00' * 1024  # Mock PCM audio data

        response = client_normal.post(
            "/api/transcribe/stream?session_id=test-id&start=0.0",
            content=audio_data,
            headers={"Content-Type": "application/octet-stream"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        assert data["session_id"] == "test-id"

    @patch("whisper_transcribe_py.api.server.streaming_sessions")
    def test_stop_stream_works_in_normal_mode(self, mock_sessions, client_normal):
        """Test DELETE /api/transcribe/stream/{session_id} works in normal mode."""
        mock_sessions.stop.return_value = True

        response = client_normal.delete("/api/transcribe/stream/test-session-id")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        mock_sessions.stop.assert_called_once_with("test-session-id")


class TestEnvironmentVariableConfig:
    """Tests for configuration via environment variables (dev mode with factory)."""

    def test_no_transcribe_from_env_var(self):
        """Test that WHISPER_NO_TRANSCRIBE environment variable is respected."""
        with patch.dict(os.environ, {"WHISPER_NO_TRANSCRIBE": "true"}):
            app = create_app()
            client = TestClient(app)

            response = client.get("/api/transcribe/config")
            assert response.json()["transcription_enabled"] is False

    def test_transcribe_api_url_from_env_var(self):
        """Test that WHISPER_TRANSCRIBE_API_URL environment variable is respected."""
        with patch.dict(os.environ, {
            "WHISPER_NO_TRANSCRIBE": "true",
            "WHISPER_TRANSCRIBE_API_URL": "https://alternate.com/api"
        }):
            app = create_app()
            client = TestClient(app)

            response = client.get("/api/transcribe/config")
            data = response.json()
            assert data["transcription_enabled"] is False
            assert data["alternate_api_url"] == "https://alternate.com/api"

    def test_dev_mode_from_env_var(self):
        """Test that WHISPER_DEV_MODE environment variable enables dev mode."""
        with patch.dict(os.environ, {"WHISPER_DEV_MODE": "true"}):
            app = create_app()
            # In dev mode, CORS middleware should be added
            # Check if any middleware wraps CORSMiddleware
            has_cors = any(
                'CORSMiddleware' in str(m.cls) if hasattr(m, 'cls') else 'CORSMiddleware' in type(m).__name__
                for m in app.user_middleware
            )
            assert has_cors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
