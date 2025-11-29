# Whisper Transcribe - File Transcription Tool

**Whisper Transcribe** is a file-based audio transcription tool that combines voice activity detection (VAD) with AI-powered transcription. It uses [silero-vad](https://github.com/snakers4/silero-vad) to intelligently detect speech segments and offers two transcription backends:

- [whispercpp](https://github.com/absadiki/pywhispercpp) (default) - Python binding for whisper.cpp (Faster on Mac with MPS support)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2-based implementation for faster GPU/CPU inference

## Features

- **Streaming Audio Processing**: Audio is streamed from ffmpeg - never loads full file into memory
- **Smart Voice Detection**: Uses Silero VAD to detect speech segments
- **Flexible Backend Selection**: Choose between whisper.cpp (default) or faster-whisper backends
- **File Transcription**: Process audio files to JSON transcripts
- **Split Mode**: Save detected speech segments as Opus files
- **Multi-language Support**: Supports all languages available in OpenAI Whisper models
- **CLI-Based**: Simple command-line interface for batch processing

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- ffmpeg - For audio format conversion

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg
```

### Installation

The tool has two installation modes:

| Installation | Commands Available | Use Case |
|-------------|-------------------|----------|
| Base | `split` only | VAD-based audio splitting without transcription |
| With `[transcribe]` | `split` + `transcribe` | Full transcription with Whisper models |

```bash
# Clone the repository
git clone https://github.com/andrewtheguy/whisper_transcribe_py.git
cd whisper_transcribe_py

# Full installation with transcription support (recommended)
uv sync --extra transcribe

# Or minimal installation (split command only, no transcription)
uv sync
```

### Running (Development)

For development, run commands with `uv run`:

```bash
# Requires [transcribe] extra
uv run whisper-transcribe-py transcribe --file audio.wav --lang en

# Works with base installation
uv run whisper-transcribe-py split --file audio.wav
```

### Installing as a Tool (Global)

To install globally and run `whisper-transcribe-py` directly:

```bash
# Full installation from GitHub (recommended)
uv tool install "whisper-transcribe-py[transcribe] @ git+https://github.com/andrewtheguy/whisper_transcribe_py.git@ref(tag or branch)"

# VAD split only (no transcription)
uv tool install "whisper-transcribe-py @ git+https://github.com/andrewtheguy/whisper_transcribe_py.git@ref(tag or branch)"

# Or from local clone
uv tool install ".[transcribe]"  # with transcription
uv tool install "."              # split only
```

After installation:

```bash
whisper-transcribe-py transcribe --file audio.wav --lang en  # requires [transcribe]
whisper-transcribe-py split --file audio.wav                 # always available
```

---

## Transcribe Command

Transcribe audio to text using Whisper models with optional VAD segmentation.

```bash
whisper-transcribe-py transcribe (--file PATH | --stdin) [OPTIONS]
```

### Transcribe Options

- `--file PATH`: Path to audio file (mutually exclusive with --stdin)
- `--stdin`: Read WAV audio from stdin (mutually exclusive with --file). Always uses VAD, always outputs JSONL to stdout.
- `--output PATH`: Output path for JSONL transcript (default: stdout)
- `--lang LANG`: Language code for transcription (default: `en`)
- `--model MODEL`: Whisper model to use (default: `large-v3-turbo`)
- `--backend {whisper_cpp, faster_whisper}`: Transcription backend (default: `whisper_cpp`)
- `--n-threads N`: Number of threads for transcription (default: 1)
- `--vad / --no-vad`: Use VAD segmentation (default: enabled). `--no-vad` has a 2-hour limit.
- `--chinese-conversion {none, simplified, traditional}`: Chinese character conversion for zh/yue languages (default: none)

**VAD tuning options** (only apply when VAD is enabled):
- `--min-speech-seconds FLOAT`: Minimum speech duration in seconds (default: 3.0)
- `--max-speech-seconds FLOAT`: Maximum speech duration in seconds (default: 60.0)
- `--speech-threshold FLOAT`: VAD speech detection threshold 0.0-1.0 (default: 0.5)
- `--min-silence-duration-ms INT`: Minimum silence duration in ms to end segment (default: 2000)
- `--look-back-seconds FLOAT`: Look-back buffer in seconds for segment start (default: 0.5)

### Transcribe Examples

**Transcribe audio file with VAD to stdout (streaming JSONL):**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --lang en
```

**Transcribe to file:**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --output transcript.jsonl --lang en
```

**Transcribe without VAD (max 2 hours):**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --output transcript.jsonl --no-vad
```

**Transcribe from stdin (WAV format, always uses VAD, outputs to stdout in JSONL format):**
```bash
# Pipe WAV audio from ffmpeg
ffmpeg -i video.mp4 -f wav - | uv run whisper-transcribe-py transcribe --stdin --lang en

# Or from a file
cat audio.wav | uv run whisper-transcribe-py transcribe --stdin --lang en
```

**Use faster-whisper backend:**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --backend faster_whisper
```

**Use different Whisper model with more threads:**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --model large-v3 --n-threads 4
```

### Transcribe Output Format (JSONL)

Transcription outputs streaming JSONL (one JSON object per line). Each entry has a `type` field:

**With VAD (default):** Includes segment boundaries and transcriptions:
```jsonl
{"type": "segment_start", "timestamp": 0.5, "timestamp_formatted": "00:00:00.500"}
{"type": "transcription", "start": 0.5, "start_formatted": "00:00:00.500", "end": 2.3, "end_formatted": "00:00:02.300", "text": "Hello world"}
{"type": "segment_end", "timestamp": 2.3, "timestamp_formatted": "00:00:02.300"}
{"type": "segment_start", "timestamp": 3.1, "timestamp_formatted": "00:00:03.100"}
{"type": "transcription", "start": 3.1, "start_formatted": "00:00:03.100", "end": 5.8, "end_formatted": "00:00:05.800", "text": "This is a test"}
{"type": "segment_end", "timestamp": 5.8, "timestamp_formatted": "00:00:05.800"}
```

- `segment_start`: Marks the beginning of a VAD-detected speech segment
- `transcription`: Contains the transcribed text with start/end timestamps
- `segment_end`: Marks the end of a VAD-detected speech segment
- `*_formatted`: Human-readable timestamps in `hh:mm:ss.ms` format

**Without VAD (`--no-vad`):** Only transcriptions (no segment boundaries):
```jsonl
{"type": "transcription", "start": 0.0, "start_formatted": "00:00:00.000", "end": 2.3, "end_formatted": "00:00:02.300", "text": "Hello world"}
{"type": "transcription", "start": 2.3, "start_formatted": "00:00:02.300", "end": 5.8, "end_formatted": "00:00:05.800", "text": "This is a test"}
```

---

## Split Command

Split audio into separate files based on VAD-detected speech segments (no transcription).

```bash
whisper-transcribe-py split (--file PATH | --url URL) [OPTIONS]
```

### Split Options

- `--file PATH`: Path to audio file (mutually exclusive with --url)
- `--url URL`: URL to audio file (mutually exclusive with --file). Live streams not supported because there is no real use case for this.
- `--preserve-sample-rate`: Preserve original sample rate (default: downsample to 16kHz)
- `--format {opus, wav}`: Output format (default: opus)

**VAD tuning options:**
- `--min-speech-seconds FLOAT`: Minimum speech duration in seconds (default: 3.0)
- `--max-speech-seconds FLOAT`: Maximum speech duration in seconds (default: 60.0)
- `--speech-threshold FLOAT`: VAD speech detection threshold 0.0-1.0 (default: 0.5)
- `--min-silence-duration-ms INT`: Minimum silence duration in ms to end segment (default: 2000)
- `--look-back-seconds FLOAT`: Look-back buffer in seconds for segment start (default: 0.5)

### Split Examples

**Split audio by VAD into Opus segments:**
```bash
uv run whisper-transcribe-py split --file audio.wav
# Outputs to tmp/audio/
```

**Split with preserved sample rate:**
```bash
uv run whisper-transcribe-py split --file audio.wav --preserve-sample-rate
```

**Split to WAV format:**
```bash
uv run whisper-transcribe-py split --file audio.wav --format wav
```

### Split Output Format

Detected speech segments are saved as Opus files (16kbps mono) to `tmp/(filename)/`:

```
tmp/audio/
  segment_0000_500ms_2300ms.opus
  segment_0001_3100ms_5800ms.opus
  ...
```

**Why Opus at 16kbps?** The split command is designed for speech processing workflows (e.g., feeding segments to transcription APIs, archiving spoken content, or reviewing detected speech). Opus with `-application voip` mode is optimized specifically for speech, delivering clear and intelligible audio at just 16kbps mono. This is not intended for high-fidelity audio preservation—use `--format wav` if you need lossless output.

---

## Supported Languages

The tool supports all languages available in OpenAI Whisper models. Common language codes:

- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh` - Chinese (Simplified/Traditional)
- `yue` - Cantonese
- `ja` - Japanese
- `ru` - Russian

**Chinese Character Conversion:** For Chinese language codes (`zh` and `yue`), you can optionally convert characters using `--chinese-conversion`:
- `none` (default): No conversion, output as-is from Whisper
- `simplified`: Convert to Simplified Chinese (zh-Hans)
- `traditional`: Convert to Traditional Chinese (zh-Hant)

Conversion is powered by [zhconv-rs](https://github.com/Xmader/zhconv-rs).

## Performance Notes

- **whisper.cpp** backend: Optimized for Mac (MPS support), good for CPU-only systems
- **faster-whisper** backend: Better for GPU or CPU-only Linux systems
- Larger models (e.g., `large-v3`) provide better accuracy but require more memory and time
- Use `--n-threads` to speed up transcription on multi-core systems

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest
```

## Project Structure

```
whisper_transcribe_py/
  ├── __init__.py
  ├── cli.py                    # CLI entry point
  ├── audio_transcriber.py      # Core transcription logic
  ├── vad_processor.py          # Voice activity detection
  ├── file_lock.py              # File locking for exclusive access

pyproject.toml                  # Project configuration
```

## Requirements

- OpenAI Whisper model files (~1-3GB depending on model size) - downloaded automatically on first use

## Known Limitations

- File-based transcription only (no real-time/live transcription)
- Live streams not supported (URLs must have fixed duration)
- `--no-vad` mode limited to 2 hours to prevent memory issues
- VAD mode has a 1-hour hard cap per speech segment (aborts if exceeded, indicating a VAD bug)
- No database persistence (outputs to JSONL files or Opus segments)
- No web interface

## License

See LICENSE file for details.
