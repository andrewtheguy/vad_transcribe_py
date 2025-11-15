# Feature Flags

This document explains the optional dependency system for Whisper Transcribe.

## Overview

Whisper Transcribe uses Python's optional dependencies (extras) to allow modular installation based on your needs. This is similar to Rust's feature flags.

## Installation Options

### Full Installation (transcription + microphone)

```bash
uv pip install -e '.[transcribe,mic]'
```

**Includes:**
- All base dependencies
- Transcription backends (`pywhispercpp`, `faster-whisper`)
- PostgreSQL database adapter (`psycopg[binary]`)
- Microphone recording (`sounddevice`)

**Use when:**
- You need full transcription capabilities
- You want to use `mic` command with transcription
- You're on a desktop platform (Windows, Mac, Linux)

### Transcription Only (no microphone)

```bash
uv pip install -e '.[transcribe]'
```

**Includes:**
- All base dependencies
- `pywhispercpp` - Whisper.cpp Python bindings (~1GB)
- `faster-whisper` - CTranslate2-based Whisper (~1GB)
- `psycopg[binary]` - PostgreSQL database adapter

**Excludes:** sounddevice (microphone recording)

**Use when:**
- You need transcription for `stream`, `file`, or `web` commands
- You don't need microphone input
- You're deploying to a server without audio hardware

### Microphone Only (no transcription)

```bash
uv pip install -e '.[mic]'
```

**Includes:**
- All base dependencies
- `sounddevice` - Microphone audio capture

**Excludes:** pywhispercpp, faster-whisper, psycopg (PostgreSQL)

**Saves:** ~2GB disk space, no database required

**Use when:**
- You only need microphone recording with `--no-transcribe` mode
- You want to save audio segments without transcription
- You're on a desktop platform

### Lightweight Installation (VAD-only)

```bash
uv pip install -e .
```

**Includes:**
- All base dependencies (silero-vad, scipy, fastapi, etc.)

**Excludes:**
- pywhispercpp, faster-whisper (transcription)
- psycopg (PostgreSQL)
- sounddevice (microphone)

**Saves:** ~2GB disk space, faster installation, no database or audio hardware required

**Use when:**
- You only need voice activity detection (VAD)
- You only process audio streams with `--no-transcribe` mode
- You're deploying to resource-constrained environments
- You're deploying to servers without audio hardware
- You want faster CI/CD builds

## Usage Examples

### With Lightweight Installation (no mic, no transcribe)

```bash
# Stream mode only - saves VAD-detected speech segments
uv run python main.py stream --config configs/mystream.toml --no-transcribe
```

### With `[mic]` Installation Only

```bash
# Mic mode: save speech segments from microphone (no transcription)
uv run python main.py mic --lang en --no-transcribe
```

### With `[transcribe]` Installation Only

```bash
# File mode: transcribe audio file (no microphone needed)
uv run python main.py file --file audio.mp3 --lang en --output output.json

# Stream mode: transcribe stream
uv run python main.py stream --config configs/mystream.toml
```

### With Both `[transcribe,mic]` Installation

```bash
# Full functionality
uv run python main.py mic --lang en  # Transcribe microphone input
```

## Error Messages

### Attempting Transcription Without Dependencies

If you try to use transcription without installing the `[transcribe]` extra, you'll get a helpful error:

```
ImportError: pywhispercpp is not installed.
To use transcription, install with: uv pip install -e '.[transcribe]'
or use --no-transcribe flag for VAD-only mode.
```

### Attempting Microphone Recording Without Dependencies

If you try to use microphone recording without installing the `[mic]` extra, you'll get a helpful error:

```
ImportError: sounddevice is not installed.
To use microphone recording, install with: uv pip install -e '.[mic]'
Note: Microphone recording is only supported on desktop platforms (Windows, Mac, Linux).
```

## How It Works

### pyproject.toml Configuration

```toml
[project]
dependencies = [
    "silero-vad>=5.1.2,<6",
    "scipy>=1.14.1,<2",
    # ... other base dependencies
    # NOTE: transcription, database, and mic dependencies are optional
]

[project.optional-dependencies]
transcribe = [
    "pywhispercpp>=1.3.3,<1.4",
    "faster-whisper>=1.0.0,<2",
    "psycopg[binary]>=3.2.3,<4",
]
mic = [
    "sounddevice>=0.5.1,<0.6",
]
dev = [
    "pytest>=9.0.0,<10",
    "httpx>=0.28.1,<1",
]
```

### Lazy Loading with Error Handling

Dependencies are imported lazily inside methods/modules with try/except blocks:

**Transcription backends:**
```python
def _load_whisper_cpp(self):
    try:
        from pywhispercpp.model import Model
    except ImportError:
        raise ImportError(
            "pywhispercpp is not installed. "
            "To use transcription, install with: uv pip install -e '.[transcribe]' "
            "or use --no-transcribe flag for VAD-only mode."
        )
    # ... continue loading
```

**Microphone recording:**
```python
try:
    import sounddevice as sd
except ImportError:
    raise ImportError(
        "sounddevice is not installed. "
        "To use microphone recording, install with: uv pip install -e '.[mic]' "
        "Note: Microphone recording is only supported on desktop platforms."
    )
```

This ensures:
1. Import errors only occur when features are actually attempted
2. Users get clear, actionable error messages
3. Minimal installations work without optional dependencies

## Development

When developing, install with all extras:

```bash
uv pip install -e '.[transcribe,mic,dev]'
```

This includes:
- Transcription backends (pywhispercpp, faster-whisper)
- PostgreSQL database adapter (psycopg)
- Microphone recording (sounddevice)
- Development tools (pytest, httpx)

## CI/CD Optimization

For CI/CD pipelines that only test VAD functionality:

```bash
# Fast installation without transcription backends
uv pip install -e '.[dev]'

# Run VAD-only tests
uv run pytest tests/test_vad_processor.py
```

For full test coverage:

```bash
# Full installation
uv pip install -e '.[transcribe,mic,dev]'

# Run all tests
uv run pytest
```

## Comparison to Rust Feature Flags

| Rust | Python (this project) |
|------|----------------------|
| `cargo build --features transcribe` | `uv pip install -e '.[transcribe]'` |
| `cargo build --no-default-features` | `uv pip install -e .` |
| Compile-time selection | Runtime dependency selection |
| Binary size reduction | Disk space & install time reduction |

## Benefits

1. **Reduced Disk Usage**: Save ~2GB by not installing transcription models
2. **Faster Installation**: Skip compilation of native extensions (pywhispercpp, faster-whisper, psycopg)
3. **No Database Required**: VAD-only mode doesn't need PostgreSQL
4. **Flexible Deployment**: Deploy only what you need
5. **Clear Error Messages**: Users know exactly what to install when needed
6. **Backward Compatible**: Existing full installations continue to work
