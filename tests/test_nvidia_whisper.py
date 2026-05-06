"""Tests for the nvidia-whisper backend (Riva gRPC stubbed)."""

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import vad_transcribe_py.audio_transcriber as audio_transcriber
from vad_transcribe_py.backends.nvidia_whisper import (
    NVIDIA_RIVA_URI,
    NVIDIA_WHISPER_FUNCTION_ID,
    NvidiaWhisperBackend,
)
from vad_transcribe_py.vad_processor import (
    NVIDIA_WHISPER_HARD_LIMIT_SECONDS,
    NVIDIA_WHISPER_SOFT_LIMIT_SECONDS,
)


@dataclass
class _StubAuth:
    uri: str | None = None
    use_ssl: bool = False
    metadata_args: list[list[str]] = field(default_factory=list)

    def __init__(self, *, uri=None, use_ssl=False, metadata_args=None, **_kwargs):
        self.uri = uri
        self.use_ssl = use_ssl
        self.metadata_args = metadata_args or []


@dataclass
class _StubRecognitionConfig:
    language_code: str = ""
    max_alternatives: int = 0
    enable_automatic_punctuation: bool = False
    enable_word_time_offsets: bool = False

    def __init__(self, *, language_code, max_alternatives, enable_automatic_punctuation,
                 enable_word_time_offsets):
        self.language_code = language_code
        self.max_alternatives = max_alternatives
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enable_word_time_offsets = enable_word_time_offsets


class _StubASRService:
    last_instance: "_StubASRService | None" = None

    def __init__(self, auth):
        self.auth = auth
        self.calls: list[dict[str, object]] = []
        self._next_text = "hello world"
        _StubASRService.last_instance = self

    def offline_recognize(self, audio_bytes, config):
        self.calls.append({"audio_bytes": audio_bytes, "config": config})
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    alternatives=[SimpleNamespace(transcript=self._next_text)]
                )
            ]
        )


def _install_stub_riva(monkeypatch):
    """Install a fake riva.client module so the backend can construct end-to-end."""
    riva_pkg = ModuleType("riva")
    riva_client = ModuleType("riva.client")
    riva_client.Auth = _StubAuth  # type: ignore[attr-defined]
    riva_client.ASRService = _StubASRService  # type: ignore[attr-defined]
    riva_client.RecognitionConfig = _StubRecognitionConfig  # type: ignore[attr-defined]
    riva_pkg.client = riva_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "riva", riva_pkg)
    monkeypatch.setitem(sys.modules, "riva.client", riva_client)
    _StubASRService.last_instance = None


def test_missing_api_key_raises(monkeypatch):
    """No NVIDIA_API_KEY → clear ValueError pointing at .env."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NvidiaWhisperBackend(language="en")


def test_init_wires_auth_metadata(monkeypatch):
    """Auth gets the right URI, SSL flag, function-id and bearer token."""
    _install_stub_riva(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")

    backend = NvidiaWhisperBackend(language="en")

    assert _StubASRService.last_instance is not None
    auth = _StubASRService.last_instance.auth
    assert auth.uri == NVIDIA_RIVA_URI
    assert auth.use_ssl is True
    metadata = {entry[0]: entry[1] for entry in auth.metadata_args}
    assert metadata["function-id"] == NVIDIA_WHISPER_FUNCTION_ID
    assert metadata["authorization"] == "Bearer nvapi-test-token"
    assert backend.hard_limit_seconds == NVIDIA_WHISPER_HARD_LIMIT_SECONDS
    assert backend.soft_limit_seconds == NVIDIA_WHISPER_SOFT_LIMIT_SECONDS
    assert NVIDIA_WHISPER_HARD_LIMIT_SECONDS == 30  # decoupled from local whisper


def test_transcribe_concatenates_segments_and_frames_audio(monkeypatch):
    """Multi-result responses concatenate; segment timestamps span the audio."""
    _install_stub_riva(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")

    backend = NvidiaWhisperBackend(language="en")
    assert _StubASRService.last_instance is not None
    _StubASRService.last_instance._next_text = "hi there"

    audio = np.zeros(16000, dtype=np.float32)  # 1.0s at 16kHz
    segments = backend.transcribe(audio, start_offset=2.5)

    assert len(segments) == 1
    assert segments[0].text == "hi there"
    assert segments[0].start == 2.5
    assert segments[0].end == pytest.approx(3.5)

    call = _StubASRService.last_instance.calls[0]
    assert isinstance(call["audio_bytes"], bytes)
    assert call["audio_bytes"][:4] == b"RIFF"  # WAV header
    assert call["config"].language_code == "en"
    assert call["config"].enable_automatic_punctuation is True


def test_transcribe_language_none_maps_to_multi(monkeypatch):
    """language=None → language_code='multi' (Riva auto-detect)."""
    _install_stub_riva(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")

    backend = NvidiaWhisperBackend(language=None)
    backend.transcribe(np.zeros(8000, dtype=np.float32))

    assert _StubASRService.last_instance is not None
    assert _StubASRService.last_instance.calls[0]["config"].language_code == "multi"


def test_factory_creates_nvidia_whisper(monkeypatch):
    """create_transcriber('nvidia-whisper') returns a backend implementing the protocol."""
    _install_stub_riva(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")

    transcriber = audio_transcriber.create_transcriber(
        language="en",
        backend="nvidia-whisper",
    )
    assert isinstance(transcriber, NvidiaWhisperBackend)
    assert isinstance(transcriber, audio_transcriber.AudioTranscriber)


def test_factory_rejects_condition_true(monkeypatch):
    """condition=True is incompatible with the stateless hosted endpoint."""
    _install_stub_riva(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")

    with pytest.raises(ValueError, match="condition=True is not supported"):
        audio_transcriber.create_transcriber(
            language="en",
            backend="nvidia-whisper",
            condition=True,
        )


def test_model_device_threads_warn_but_accept(monkeypatch, caplog):
    """model/device/num_threads are ignored with a warning, not an error."""
    import logging

    _install_stub_riva(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")
    caplog.set_level(logging.WARNING)

    _ = NvidiaWhisperBackend(
        language="en", model="some/other-model", device="cuda", num_threads=4,
    )
    messages = " ".join(r.message for r in caplog.records)
    assert "model=" in messages
    assert "device=" in messages
    assert "num_threads=" in messages
