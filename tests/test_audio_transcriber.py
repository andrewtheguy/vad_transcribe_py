import numpy as np
import pytest

import whisper_transcribe_py.audio_transcriber as audio_transcriber


@pytest.fixture(autouse=True)
def stub_whisper_cpp(monkeypatch):
    """Stub out whisper.cpp model loading."""
    monkeypatch.setattr(audio_transcriber.WhisperTranscriber, "_load_whisper_cpp", lambda self: None)


def test_transcriber_initialization():
    """Test that transcriber initializes with correct parameters."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="en",
        model="large-v3-turbo",
        backend="whisper_cpp",
        n_threads=4,
    )

    assert transcriber.language == "en"
    assert transcriber.model == "large-v3-turbo"
    assert transcriber.backend == "whisper_cpp"
    assert transcriber.n_threads == 4


def test_create_transcriber_factory():
    """Test the factory function creates transcriber correctly."""
    transcriber = audio_transcriber.create_transcriber(
        language="zh",
        model="large-v3",
        backend="whisper_cpp",
        n_threads=2,
    )

    assert transcriber.language == "zh"
    assert transcriber.model == "large-v3"
    assert transcriber.backend == "whisper_cpp"
    assert transcriber.n_threads == 2


def test_pcm_conversion_functions():
    """Test PCM audio conversion functions."""
    # Test pcm_int16_to_float32
    int16_audio = np.array([0, 16384, -16384, 32767], dtype=np.int16)
    float32_audio = audio_transcriber.pcm_int16_to_float32(int16_audio)

    assert float32_audio.dtype == np.float32
    assert float32_audio[0] == 0.0
    assert -1.0 <= float32_audio[3] <= 1.0

    # Test pcm_s16le_to_float32
    pcm_bytes = int16_audio.tobytes()
    float32_from_bytes = audio_transcriber.pcm_s16le_to_float32(pcm_bytes)

    assert float32_from_bytes.dtype == np.float32
    assert len(float32_from_bytes) == len(int16_audio)


def test_get_window_size_samples():
    """Test window size calculation."""
    window_size = audio_transcriber.get_window_size_samples()
    assert window_size == 512  # For 16kHz sample rate


def test_transcribed_segment_dataclass():
    """Test TranscribedSegment dataclass."""
    segment = audio_transcriber.TranscribedSegment(
        text="Hello world",
        start=1.5,
        end=3.0,
    )

    assert segment.text == "Hello world"
    assert segment.start == 1.5
    assert segment.end == 3.0


def test_process_text_chinese():
    """Test that Chinese text is converted to Traditional Chinese."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="zh",
        model="large-v3-turbo",
        backend="whisper_cpp",
    )

    # zhconv converts simplified to traditional
    result = transcriber._process_text("简体中文")
    assert result == "簡體中文"


def test_process_text_cantonese():
    """Test that Cantonese text is converted to Traditional Chinese."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="yue",
        model="large-v3-turbo",
        backend="whisper_cpp",
    )

    result = transcriber._process_text("简体中文")
    assert result == "簡體中文"


def test_process_text_english():
    """Test that English text is not modified."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="en",
        model="large-v3-turbo",
        backend="whisper_cpp",
    )

    result = transcriber._process_text("Hello world")
    assert result == "Hello world"


def test_unsupported_backend():
    """Test that unsupported backend raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        audio_transcriber.WhisperTranscriber(
            language="en",
            model="large-v3-turbo",
            backend="unsupported_backend",
        )
