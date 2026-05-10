"""
Integration tests for CLI split_by_vad function.

Tests both default mode (16kHz downsampling) and --preserve-sample-rate mode
using real speech audio (samples/samples_jfk.wav).
"""

import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vad_transcribe_py.audio_transcriber import TranscribedSegment
from vad_transcribe_py.cli import (
    get_audio_properties,
    resample_to_16k,
    split_by_vad,
    write_jsonl_segment,
)


# Path to JFK speech sample (16kHz mono, ~11 seconds)
SAMPLES_DIR = Path(__file__).parent.parent / "samples"
JFK_SAMPLE = SAMPLES_DIR / "samples_jfk.wav"


def get_audio_file_properties(filepath: str) -> dict:
    """Get audio properties of an audio file using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,codec_name",
         "-of", "json", filepath],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No audio stream found in {filepath}")

    stream = data["streams"][0]
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "codec": stream["codec_name"],
    }


def convert_sample_rate(input_path: str, output_path: str, sample_rate: int) -> None:
    """Convert audio file to a different sample rate using ffmpeg."""
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",  # Keep mono
        output_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")


@pytest.fixture(scope="module")
def jfk_16k():
    """Path to original JFK sample (16kHz mono)."""
    assert JFK_SAMPLE.exists(), f"JFK sample not found at {JFK_SAMPLE}"
    return str(JFK_SAMPLE)


@pytest.fixture(scope="module")
def jfk_48k(tmp_path_factory):
    """Create a 48kHz version of JFK sample for testing preserve mode."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    output_path = tmp_dir / "jfk_48k.wav"
    convert_sample_rate(str(JFK_SAMPLE), str(output_path), 48000)
    return str(output_path)


@pytest.fixture(scope="module")
def jfk_44k(tmp_path_factory):
    """Create a 44.1kHz version of JFK sample."""
    tmp_dir = tmp_path_factory.mktemp("audio")
    output_path = tmp_dir / "jfk_44k.wav"
    convert_sample_rate(str(JFK_SAMPLE), str(output_path), 44100)
    return str(output_path)


class TestGetAudioProperties:
    """Tests for get_audio_properties function."""

    def test_get_properties_16k_mono(self, jfk_16k):
        """Test getting properties from 16kHz mono WAV."""
        props = get_audio_properties(jfk_16k)

        assert props["sample_rate"] == 16000
        assert props["channels"] == 1

    def test_get_properties_48k(self, jfk_48k):
        """Test getting properties from 48kHz WAV."""
        props = get_audio_properties(jfk_48k)

        assert props["sample_rate"] == 48000
        assert props["channels"] == 1

    def test_get_properties_44k(self, jfk_44k):
        """Test getting properties from 44.1kHz WAV."""
        props = get_audio_properties(jfk_44k)

        assert props["sample_rate"] == 44100
        assert props["channels"] == 1

    def test_get_properties_nonexistent_file(self):
        """Test that nonexistent file raises error."""
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            get_audio_properties("/nonexistent/path/audio.wav")


class TestSplitByVadDefault:
    """Integration tests for split_by_vad with default mode (16kHz downsampling)."""

    def test_split_default_detects_speech(self, jfk_16k, tmp_path, monkeypatch):
        """Test that default split detects speech segments in JFK sample."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_16k, preserve_sample_rate=False)

        # JFK sample has clear speech - must detect at least 1 segment
        assert segment_count >= 1, "VAD should detect speech in JFK sample"

    def test_split_default_creates_opus_files(self, jfk_16k, tmp_path, monkeypatch):
        """Test that default split creates Opus segment files."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_16k, preserve_sample_rate=False)

        base_name = os.path.splitext(os.path.basename(jfk_16k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = list(output_dir.glob("*.opus"))

        assert output_dir.exists()
        assert len(opus_files) == segment_count
        assert segment_count >= 1

    def test_split_default_output_is_mono_opus(self, jfk_16k, tmp_path, monkeypatch):
        """Test that default split produces mono Opus files."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_16k, preserve_sample_rate=False)
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_16k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = list(output_dir.glob("*.opus"))

        for opus_file in opus_files:
            props = get_audio_file_properties(str(opus_file))
            assert props["channels"] == 1
            assert props["codec"] == "opus"

    def test_split_48k_default_downsamples(self, jfk_48k, tmp_path, monkeypatch):
        """Test that 48kHz input is downsampled in default mode."""
        monkeypatch.chdir(tmp_path)

        # Verify input is 48kHz
        input_props = get_audio_properties(jfk_48k)
        assert input_props["sample_rate"] == 48000

        segment_count = split_by_vad(jfk_48k, preserve_sample_rate=False)
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_48k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = list(output_dir.glob("*.opus"))

        # Output should be valid Opus (mono)
        for opus_file in opus_files:
            props = get_audio_file_properties(str(opus_file))
            assert props["channels"] == 1
            assert props["codec"] == "opus"


class TestSplitByVadPreserveSampleRate:
    """Integration tests for split_by_vad with --preserve-sample-rate mode."""

    def test_preserve_detects_speech(self, jfk_48k, tmp_path, monkeypatch):
        """Test that preserve mode detects speech segments."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_48k, preserve_sample_rate=True)

        assert segment_count >= 1, "VAD should detect speech in JFK sample"

    def test_preserve_creates_opus_files(self, jfk_48k, tmp_path, monkeypatch):
        """Test that preserve mode creates Opus segment files."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_48k, preserve_sample_rate=True)

        base_name = os.path.splitext(os.path.basename(jfk_48k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = list(output_dir.glob("*.opus"))

        assert output_dir.exists()
        assert len(opus_files) == segment_count
        assert segment_count >= 1

    def test_preserve_output_is_mono_opus(self, jfk_48k, tmp_path, monkeypatch):
        """Test that preserve mode produces mono Opus files."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_48k, preserve_sample_rate=True)
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_48k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = list(output_dir.glob("*.opus"))

        for opus_file in opus_files:
            props = get_audio_file_properties(str(opus_file))
            assert props["channels"] == 1
            assert props["codec"] == "opus"

    def test_preserve_with_44k_input(self, jfk_44k, tmp_path, monkeypatch):
        """Test preserve mode with 44.1kHz input."""
        monkeypatch.chdir(tmp_path)

        # Verify input is 44.1kHz
        input_props = get_audio_properties(jfk_44k)
        assert input_props["sample_rate"] == 44100

        segment_count = split_by_vad(jfk_44k, preserve_sample_rate=True)
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_44k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = list(output_dir.glob("*.opus"))

        for opus_file in opus_files:
            props = get_audio_file_properties(str(opus_file))
            assert props["channels"] == 1
            assert props["codec"] == "opus"

    def test_preserve_48k_wav_verifies_sample_rate(self, jfk_48k, tmp_path, monkeypatch):
        """Test that WAV output preserves 48kHz sample rate."""
        monkeypatch.chdir(tmp_path)

        # Verify input is 48kHz
        input_props = get_audio_properties(jfk_48k)
        assert input_props["sample_rate"] == 48000

        segment_count = split_by_vad(jfk_48k, preserve_sample_rate=True, output_format="wav")
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_48k))[0]
        output_dir = tmp_path / "tmp" / base_name
        wav_files = list(output_dir.glob("*.wav"))

        assert len(wav_files) == segment_count

        # WAV output should preserve the exact sample rate
        for wav_file in wav_files:
            props = get_audio_file_properties(str(wav_file))
            assert props["sample_rate"] == 48000, f"Expected 48kHz, got {props['sample_rate']}Hz"
            assert props["channels"] == 1
            assert props["codec"] == "pcm_s16le"

    def test_preserve_44k_wav_verifies_sample_rate(self, jfk_44k, tmp_path, monkeypatch):
        """Test that WAV output preserves 44.1kHz sample rate."""
        monkeypatch.chdir(tmp_path)

        # Verify input is 44.1kHz
        input_props = get_audio_properties(jfk_44k)
        assert input_props["sample_rate"] == 44100

        segment_count = split_by_vad(jfk_44k, preserve_sample_rate=True, output_format="wav")
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_44k))[0]
        output_dir = tmp_path / "tmp" / base_name
        wav_files = list(output_dir.glob("*.wav"))

        assert len(wav_files) == segment_count

        # WAV output should preserve the exact sample rate
        for wav_file in wav_files:
            props = get_audio_file_properties(str(wav_file))
            assert props["sample_rate"] == 44100, f"Expected 44.1kHz, got {props['sample_rate']}Hz"
            assert props["channels"] == 1
            assert props["codec"] == "pcm_s16le"

    def test_default_mode_wav_is_16k(self, jfk_48k, tmp_path, monkeypatch):
        """Test that default mode (no preserve) outputs 16kHz WAV."""
        monkeypatch.chdir(tmp_path)

        # Input is 48kHz but should be downsampled
        input_props = get_audio_properties(jfk_48k)
        assert input_props["sample_rate"] == 48000

        segment_count = split_by_vad(jfk_48k, preserve_sample_rate=False, output_format="wav")
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_48k))[0]
        output_dir = tmp_path / "tmp" / base_name
        wav_files = list(output_dir.glob("*.wav"))

        # Default mode should downsample to 16kHz
        for wav_file in wav_files:
            props = get_audio_file_properties(str(wav_file))
            assert props["sample_rate"] == 16000, f"Expected 16kHz, got {props['sample_rate']}Hz"
            assert props["channels"] == 1


class TestSplitByVadComparison:
    """Compare default vs preserve modes to ensure both detect same segments."""

    def test_both_modes_same_segment_count(self, jfk_48k, tmp_path):
        """Both modes should detect the same number of speech segments."""
        # Run default mode
        default_dir = tmp_path / "default"
        default_dir.mkdir()
        os.chdir(default_dir)
        default_count = split_by_vad(jfk_48k, preserve_sample_rate=False)

        # Run preserve mode
        preserve_dir = tmp_path / "preserve"
        preserve_dir.mkdir()
        os.chdir(preserve_dir)
        preserve_count = split_by_vad(jfk_48k, preserve_sample_rate=True)

        # Both should detect speech
        assert default_count >= 1
        assert preserve_count >= 1

        # Both should produce the same number of segments
        assert default_count == preserve_count

    def test_segment_filenames_match_pattern(self, jfk_16k, tmp_path, monkeypatch):
        """Test that segment filenames follow expected pattern."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_16k, preserve_sample_rate=False)
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_16k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = sorted(output_dir.glob("*.opus"))

        for i, opus_file in enumerate(opus_files):
            # Pattern: segment_XXXX_XXXms_XXXms.opus
            name = opus_file.name
            assert name.startswith(f"segment_{i:04d}_")
            assert name.endswith(".opus")
            assert "ms_" in name

    def test_segment_timestamps_reasonable(self, jfk_16k, tmp_path, monkeypatch):
        """Test that segment timestamps are within audio duration."""
        monkeypatch.chdir(tmp_path)

        segment_count = split_by_vad(jfk_16k, preserve_sample_rate=False)
        assert segment_count >= 1

        base_name = os.path.splitext(os.path.basename(jfk_16k))[0]
        output_dir = tmp_path / "tmp" / base_name
        opus_files = sorted(output_dir.glob("*.opus"))

        # JFK sample is ~11 seconds
        max_duration_ms = 12000  # Allow some margin

        for opus_file in opus_files:
            # Extract timestamps from filename: segment_XXXX_STARTms_ENDms.opus
            name = opus_file.stem  # Remove .opus
            parts = name.split("_")
            start_ms = int(parts[2].replace("ms", ""))
            end_ms = int(parts[3].replace("ms", ""))

            assert start_ms >= 0
            assert end_ms > start_ms
            assert end_ms <= max_duration_ms


class TestResampleTo16k:
    """Tests for the resample_to_16k function."""

    def test_resample_same_rate_passthrough(self):
        """Test that 16kHz input passes through unchanged."""
        audio = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=16000)

        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, audio)

    def test_resample_48k_to_16k(self):
        """Test resampling from 48kHz to 16kHz."""
        # 4800 samples at 48kHz = 0.1 seconds
        # Should become 1600 samples at 16kHz
        audio = np.sin(2 * np.pi * 440 * np.arange(4800) / 48000).astype(np.float32)
        result = resample_to_16k(audio, orig_sr=48000)

        assert result.dtype == np.float32
        assert len(result) == 1600

    def test_resample_44100_to_16k(self):
        """Test resampling from 44.1kHz to 16kHz."""
        # 4410 samples at 44.1kHz = 0.1 seconds
        # Should become 1600 samples at 16kHz
        audio = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100).astype(np.float32)
        result = resample_to_16k(audio, orig_sr=44100)

        assert result.dtype == np.float32
        assert len(result) == 1600

    def test_resample_preserves_signal_shape(self):
        """Test that resampling preserves general signal characteristics."""
        # Create a simple signal with known characteristics
        orig_sr = 48000
        duration = 0.1
        freq = 440
        t = np.arange(int(orig_sr * duration)) / orig_sr
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

        result = resample_to_16k(audio, orig_sr=orig_sr)

        # Check that max amplitude is preserved (approximately)
        assert abs(np.max(np.abs(result)) - 1.0) < 0.1


class TestCLIIntegration:
    """Test CLI argument parsing and execution."""

    def test_split_help_shows_preserve_option(self):
        """Test that --preserve-sample-rate appears in help."""
        result = subprocess.run(
            ["uv", "run", "vad-transcribe-py", "split", "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--preserve-sample-rate" in result.stdout

    def test_split_command_default_with_real_speech(self, jfk_16k, tmp_path):
        """Test split command with real speech audio."""
        result = subprocess.run(
            ["uv", "run", "vad-transcribe-py", "split", "--file", jfk_16k],
            capture_output=True, text=True,
            cwd=tmp_path
        )

        assert result.returncode == 0
        assert "Saved" in result.stderr
        assert "segment" in result.stderr.lower()

    def test_split_command_preserve_with_real_speech(self, jfk_48k, tmp_path):
        """Test split command with --preserve-sample-rate and real speech."""
        result = subprocess.run(
            ["uv", "run", "vad-transcribe-py", "split",
             "--file", jfk_48k, "--preserve-sample-rate"],
            capture_output=True, text=True,
            cwd=tmp_path
        )

        assert result.returncode == 0
        assert "preserving sample rate" in result.stderr
        assert "Saved" in result.stderr

    def test_split_command_nonexistent_file(self, tmp_path):
        """Test split command with nonexistent file."""
        result = subprocess.run(
            ["uv", "run", "vad-transcribe-py", "split",
             "--file", "/nonexistent/audio.wav"],
            capture_output=True, text=True,
            cwd=tmp_path
        )

        assert result.returncode != 0
        assert "Error" in result.stderr or "error" in result.stderr.lower()


class TestWriteJsonlSegmentFlags:
    """JSONL transcript writer omits boundary flags by default and emits when set."""

    def _write_and_parse(self, segment: TranscribedSegment) -> dict:
        buf = io.StringIO()
        write_jsonl_segment(segment, buf)
        return json.loads(buf.getvalue())

    def test_flags_omitted_when_false(self):
        payload = self._write_and_parse(
            TranscribedSegment(text="Hello world.", start=0.0, end=1.0)
        )
        assert payload["type"] == "transcript"
        assert payload["text"] == "Hello world."
        assert "continued_from_prior" not in payload
        assert "ends_mid_sentence" not in payload

    def test_continued_from_prior_emitted_when_true(self):
        payload = self._write_and_parse(
            TranscribedSegment(
                text="continuation",
                start=5.0,
                end=6.0,
                continued_from_prior=True,
            )
        )
        assert payload["continued_from_prior"] is True
        assert "ends_mid_sentence" not in payload

    def test_ends_mid_sentence_emitted_when_true(self):
        payload = self._write_and_parse(
            TranscribedSegment(
                text="cut off",
                start=2.0,
                end=3.0,
                ends_mid_sentence=True,
            )
        )
        assert payload["ends_mid_sentence"] is True
        assert "continued_from_prior" not in payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
