import pytest

import whisper_transcribe_py.audio_transcriber as audio_transcriber


@pytest.fixture(autouse=True)
def stub_whisper(monkeypatch):
    """Stub out Whisper model loading."""
    monkeypatch.setattr(audio_transcriber.WhisperTranscriber, "_load_whisper", lambda _self: None)


def test_transcriber_initialization():
    """Test that transcriber initializes with correct parameters."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="en",
        model="large-v3-turbo",
        backend="whisper",
    )

    assert transcriber.language == "en"
    assert transcriber.model == "large-v3-turbo"
    assert transcriber.backend == "whisper"


def test_create_transcriber_factory():
    """Test the factory function creates transcriber correctly."""
    transcriber = audio_transcriber.create_transcriber(
        language="zh",
        model="large-v3",
        backend="whisper",
    )

    assert transcriber.language == "zh"
    assert transcriber.model == "large-v3"
    assert transcriber.backend == "whisper"


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


def test_process_text_chinese_no_conversion():
    """Test that Chinese text is not converted by default."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="zh",
        model="large-v3-turbo",
        backend="whisper",
    )

    # Default: no conversion
    result = transcriber._process_text("简体中文")
    assert result == "简体中文"


def test_process_text_chinese_to_traditional():
    """Test that Chinese text is converted to Traditional Chinese."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="zh",
        model="large-v3-turbo",
        backend="whisper",
        chinese_conversion="traditional",
    )

    result = transcriber._process_text("简体中文")
    assert result == "簡體中文"


def test_process_text_chinese_to_simplified():
    """Test that Chinese text is converted to Simplified Chinese."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="zh",
        model="large-v3-turbo",
        backend="whisper",
        chinese_conversion="simplified",
    )

    result = transcriber._process_text("簡體中文")
    assert result == "简体中文"


def test_process_text_cantonese_no_conversion():
    """Test that Cantonese text is not converted by default."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="yue",
        model="large-v3-turbo",
        backend="whisper",
    )

    # Default: no conversion
    result = transcriber._process_text("简体中文")
    assert result == "简体中文"


def test_process_text_cantonese_to_traditional():
    """Test that Cantonese text is converted to Traditional Chinese."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="yue",
        model="large-v3-turbo",
        backend="whisper",
        chinese_conversion="traditional",
    )

    result = transcriber._process_text("简体中文")
    assert result == "簡體中文"


def test_process_text_english():
    """Test that English text is not modified."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="en",
        model="large-v3-turbo",
        backend="whisper",
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


def test_hard_limit_seconds_whisper():
    """Test that whisper backend reports correct hard limit."""
    transcriber = audio_transcriber.WhisperTranscriber(
        language="en",
        backend="whisper",
    )
    assert transcriber.hard_limit_seconds == audio_transcriber.WHISPER_HARD_LIMIT_SECONDS


def test_hard_limit_seconds_moonshine(monkeypatch):
    """Test that moonshine backend reports correct hard limit."""
    monkeypatch.setattr(audio_transcriber.WhisperTranscriber, "_load_moonshine", lambda _self: None)
    transcriber = audio_transcriber.WhisperTranscriber(
        language="zh",
        model="moonshine-tiny-zh",
        backend="moonshine",
    )
    assert transcriber.hard_limit_seconds == audio_transcriber.MOONSHINE_HARD_LIMIT_SECONDS


def test_resolve_model_id_whisper():
    """Test model ID resolution for whisper backend."""
    assert audio_transcriber.resolve_model_id("large-v3-turbo", "whisper") == "openai/whisper-large-v3-turbo"
    assert audio_transcriber.resolve_model_id("openai/whisper-large-v3", "whisper") == "openai/whisper-large-v3"


def test_resolve_model_id_moonshine():
    """Test model ID resolution for moonshine backend."""
    assert audio_transcriber.resolve_model_id("moonshine-tiny-zh", "moonshine") == "UsefulSensors/moonshine-tiny-zh"
    assert audio_transcriber.resolve_model_id("UsefulSensors/moonshine-tiny-ja", "moonshine") == "UsefulSensors/moonshine-tiny-ja"
