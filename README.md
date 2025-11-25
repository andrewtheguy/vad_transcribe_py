# Whisper Transcribe - File Transcription Tool

**Whisper Transcribe** is a file-based audio transcription tool that combines voice activity detection (VAD) with AI-powered transcription. It uses [silero-vad](https://github.com/snakers4/silero-vad) to intelligently detect speech segments and offers two transcription backends:

- [whispercpp](https://github.com/absadiki/pywhispercpp) (default) - Python binding for whisper.cpp (Faster on Mac with MPS support)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2-based implementation for faster GPU/CPU inference

## Features

- **Smart Voice Detection**: Uses Silero VAD to detect speech segments
- **Flexible Backend Selection**: Choose between whisper.cpp (default) or faster-whisper backends
- **File Transcription**: Process audio files to JSON transcripts
- **VAD-Only Mode**: Save detected speech segments as WAV files without transcription
- **Multi-language Support**: Supports all languages available in OpenAI Whisper models
- **CLI-Based**: Simple command-line interface for batch processing

## Quick Start

### Installation

**Full installation with transcription:**
```bash
uv pip install -e '.[transcribe]'
```

**Minimal installation (VAD-only, no transcription):**
```bash
uv pip install -e .
```

### Usage

**Transcribe audio file to JSON:**
```bash
uv run python main.py --lang en file --file audio.wav --output transcript.json
```

**VAD-only mode (save speech segments as WAV files):**
```bash
uv run python main.py file --file audio.wav --no-transcribe
```

**Transcribe with faster-whisper backend:**
```bash
uv run python main.py --backend faster_whisper file --file audio.wav --output transcript.json
```

**Use different Whisper model:**
```bash
uv run python main.py --model large-v3 file --file audio.wav --output transcript.json
```

## Command-Line Options

### Global Options

- `--model MODEL_NAME`: Whisper model to use (default: `large-v3-turbo`)
- `--backend {whisper_cpp, faster_whisper}`: Transcription backend (default: `whisper_cpp`)
- `--n-threads N`: Number of threads for transcription (default: 1)

### File Command Options

- `--file PATH`: Path to audio file (required)
- `--output PATH`: Output path for JSON transcript (required for `--transcribe`)
- `--lang LANG`: Language code for transcription (default: `en`)
- `--transcribe/--no-transcribe`: Enable/disable transcription (default: enabled)

## Output Format

### Transcription Output (JSON)

When transcribing, output is saved as JSON:

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 2.3,
      "text": "Hello world"
    },
    {
      "start": 3.1,
      "end": 5.8,
      "text": "This is a test"
    }
  ]
}
```

### VAD-Only Mode

In VAD-only mode (`--no-transcribe`), detected speech segments are saved as WAV files to `~/whisper_segments/` directory.

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

## Performance Notes

- **whisper.cpp** backend: Optimized for Mac (MPS support), good for CPU-only systems
- **faster-whisper** backend: Better for GPU or CPU-only Linux systems
- Larger models (e.g., `large-v3`) provide better accuracy but require more memory and time

## Testing

Run tests with pytest:

```bash
uv run pytest
```

## Project Structure

```
whisper_transcribe_py/
  ├── __init__.py
  ├── audio_transcriber.py       # Core transcription logic
  ├── vad_processor.py           # Voice activity detection

main.py                          # CLI entry point
file_lock.py                     # File locking for exclusive access
pyproject.toml                   # Project configuration
```

## Requirements

- Python 3.12+
- ffmpeg (for audio format conversion)
- OpenAI Whisper model files (~1-3GB depending on model size)

## Known Limitations

- File-based transcription only (no real-time/live transcription)
- No database persistence (outputs to JSON files or WAV segments)
- No web interface
- Requires local audio files (no URL streaming)

## License

See LICENSE file for details.
