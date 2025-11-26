# Whisper Transcribe - File Transcription Tool

**Whisper Transcribe** is a file-based audio transcription tool that combines voice activity detection (VAD) with AI-powered transcription. It uses [silero-vad](https://github.com/snakers4/silero-vad) to intelligently detect speech segments and offers two transcription backends:

- [whispercpp](https://github.com/absadiki/pywhispercpp) (default) - Python binding for whisper.cpp (Faster on Mac with MPS support)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2-based implementation for faster GPU/CPU inference

## Features

- **Streaming Audio Processing**: Audio is streamed from ffmpeg - never loads full file into memory
- **Smart Voice Detection**: Uses Silero VAD to detect speech segments
- **Flexible Backend Selection**: Choose between whisper.cpp (default) or faster-whisper backends
- **File Transcription**: Process audio files to JSON transcripts
- **Split Mode**: Save detected speech segments as WAV files
- **Multi-language Support**: Supports all languages available in OpenAI Whisper models
- **CLI-Based**: Simple command-line interface for batch processing

## Quick Start

### Installation

**Full installation with transcription:**
```bash
uv sync --extra transcribe
```

**Minimal installation (VAD split only, no transcription):**
```bash
uv sync
```

### Usage

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

**Split audio by VAD into Opus segments:**
```bash
uv run whisper-transcribe-py split --file audio.wav
# Outputs to tmp/audio/
```

**Use faster-whisper backend:**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --backend faster_whisper
```

**Use different Whisper model with more threads:**
```bash
uv run whisper-transcribe-py transcribe --file audio.wav --model large-v3 --n-threads 4
```

## Command-Line Options

### Transcribe Command

```bash
whisper-transcribe-py transcribe --file PATH [OPTIONS]
```

- `--file PATH`: Path to audio file (required)
- `--output PATH`: Output path for JSONL transcript (default: stdout)
- `--lang LANG`: Language code for transcription (default: `en`)
- `--model MODEL`: Whisper model to use (default: `large-v3-turbo`)
- `--backend {whisper_cpp, faster_whisper}`: Transcription backend (default: `whisper_cpp`)
- `--n-threads N`: Number of threads for transcription (default: 1)
- `--vad / --no-vad`: Use VAD segmentation (default: enabled). `--no-vad` has a 2-hour limit.

### Split Command

```bash
whisper-transcribe-py split --file PATH
```

- `--file PATH`: Path to audio file (required)
- Output directory: `tmp/(filename without extension)/`

## Output Format

### Transcription Output (JSONL)

Transcription outputs streaming JSONL (one JSON object per line). Each segment is output as soon as it's transcribed:

```jsonl
{"start": 0.5, "end": 2.3, "text": "Hello world"}
{"start": 3.1, "end": 5.8, "text": "This is a test"}
```

### Split Mode

In split mode, detected speech segments are saved as Opus files (16kbps mono) to `tmp/(filename)/`:

```
tmp/audio/
  segment_0000_500ms_2300ms.opus
  segment_0001_3100ms_5800ms.opus
  ...
```

**Why Opus?** Opus is chosen because the output is expected to be speech, and Opus excels at encoding speech at low bitrates. The encoder uses `-application voip` mode which optimizes for speech content, providing better quality than general audio encoding at the same bitrate. At 16kbps mono, Opus delivers intelligible speech while keeping file sizes minimal.

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
- Use `--n-threads` to speed up transcription on multi-core systems

## Testing

Run tests with pytest:

```bash
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

- Python 3.12+
- ffmpeg (for audio format conversion)
- OpenAI Whisper model files (~1-3GB depending on model size)

## Known Limitations

- File-based transcription only (no real-time/live transcription)
- `--no-vad` mode limited to 2 hours to prevent memory issues
- VAD mode has a 1-hour hard cap per speech segment (aborts if exceeded, indicating a VAD bug)
- No database persistence (outputs to JSON files or WAV segments)
- No web interface
- Requires local audio files (no URL streaming)

## License

See LICENSE file for details.
