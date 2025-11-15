# Feature Flags

This document explains the optional dependency system for Whisper Transcribe.

## Overview

Whisper Transcribe uses Python's optional dependencies (extras) to allow modular installation based on your needs. This is similar to Rust's feature flags.

## Installation Options

### Full Installation (all features)

```bash
uv pip install -e '.[transcribe,mic,web]'
```

**Includes:**
- All base dependencies
- Transcription backends (`pywhispercpp`, `faster-whisper`)
- PostgreSQL database adapter (`psycopg[binary]`)
- Microphone recording (`sounddevice`)
- Web server (`fastapi`, `uvicorn`)

**Use when:**
- You want all features: CLI transcription, microphone, database, and web server
- Development environment with full capabilities

### CLI-only Installation (transcription + microphone + database)

```bash
uv pip install -e '.[transcribe,mic]'
```

**Includes:**
- All base dependencies
- Transcription backends (`pywhispercpp`, `faster-whisper`)
- PostgreSQL database adapter (`psycopg[binary]`)
- Microphone recording (`sounddevice`)

**Excludes:** fastapi, uvicorn (web server)

**Use when:**
- You only need CLI tools (no web interface)
- Server deployment for processing audio streams
- Desktop environment for transcribing microphone or files to database

### Web Server (full features)

```bash
uv pip install -e '.[web,transcribe,mic]'
```

**Includes:**
- All base dependencies
- Web server (`fastapi`, `uvicorn`, `psycopg[binary]`)
- Transcription backends (`pywhispercpp`, `faster-whisper`)
- Microphone recording (`sounddevice`)

**Use when:**
- You want web interface with full transcription capabilities
- Running web server that can record and transcribe from browser
- Need both viewing and creating transcripts via web UI

### Web Server (view-only)

```bash
uv pip install -e '.[web]'
```

**Includes:**
- All base dependencies
- Web server (`fastapi`, `uvicorn`, `psycopg[binary]`)

**Excludes:** pywhispercpp, faster-whisper, sounddevice

**Saves:** ~2GB disk space (no transcription models)

**Use when:**
- You only need to view existing transcripts via web UI
- Separate read-only web server for viewing transcripts
- Don't need transcription capabilities on this instance

### Transcription Only (file processing to JSON)

```bash
uv pip install -e '.[transcribe]'
```

**Includes:**
- All base dependencies
- Transcription backends (`pywhispercpp`, `faster-whisper`)
- PostgreSQL database adapter (`psycopg[binary]`)

**Excludes:** sounddevice, fastapi, uvicorn

**Use when:**
- Processing audio files to JSON output only (file mode doesn't use database)
- CLI transcription to database (mic/stream modes)
- Don't need microphone or web interface

### Microphone Only (no transcription, no database)

```bash
uv pip install -e '.[mic]'
```

**Includes:**
- All base dependencies
- Microphone recording (`sounddevice`)

**Excludes:** pywhispercpp, faster-whisper, psycopg, fastapi, uvicorn

**Saves:** ~2GB disk space, no database required

**Use when:**
- Only need microphone recording with `--no-transcribe` mode
- Want to save audio segments without transcription
- Desktop platform only

### Lightweight Installation (VAD-only)

```bash
uv pip install -e .
```

**Includes:**
- All base dependencies (silero-vad, scipy, torch, numpy, etc.)

**Excludes:**
- pywhispercpp, faster-whisper (transcription)
- psycopg (PostgreSQL)
- sounddevice (microphone)
- fastapi, uvicorn (web server)

**Saves:** ~2GB disk space, faster installation, no database or audio hardware required

**Use when:**
- You only need voice activity detection (VAD)
- Processing audio streams with `--no-transcribe` mode
- Deploying to resource-constrained environments
- Deploying to servers without audio hardware
- Want faster CI/CD builds

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
# File mode: transcribe audio file to JSON (no database needed)
uv run python main.py file --file audio.mp3 --lang en --output output.json
```

### With `[transcribe,mic]` Installation (CLI full)

```bash
# Mic mode: transcribe microphone input to database
uv run python main.py mic --lang en

# Stream mode: transcribe stream to database
uv run python main.py stream --config configs/mystream.toml
```

### With `[web]` Installation Only (view-only)

```bash
# Web server: view existing transcripts only
uv run python main.py web --no-transcribe
```

### With `[web,transcribe,mic]` Installation (web full)

```bash
# Web server: full functionality with transcription
uv run python main.py web
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

### Attempting Database Access Without Dependencies

If you try to use database features without installing the `[transcribe]` or `[web]` extra, you'll get a helpful error:

```
ImportError: psycopg is not installed.
To use database features with CLI, install with: uv pip install -e '.[transcribe]'
```

### Attempting Web Server Without Dependencies

If you try to start the web server without installing the `[web]` extra, you'll get a helpful error:

```
ImportError: Web server dependencies are not installed.
To use the web server, install with: uv pip install -e '.[web]'
```

## How It Works

### pyproject.toml Configuration

```toml
[project]
dependencies = [
    "silero-vad>=5.1.2,<6",
    "scipy>=1.14.1,<2",
    "torch>=2.5.1,<3",
    "numpy>=2.2.1,<3",
    # ... other base dependencies
    # NOTE: transcription, database, mic, and web dependencies are optional
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
web = [
    "fastapi>=0.115.0,<1",
    "uvicorn[standard]>=0.32.0,<1",
    "psycopg[binary]>=3.2.3,<4",
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
uv pip install -e '.[transcribe,mic,web,dev]'
```

This includes:
- Transcription backends (pywhispercpp, faster-whisper)
- PostgreSQL database adapter (psycopg)
- Microphone recording (sounddevice)
- Web server (fastapi, uvicorn)
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
uv pip install -e '.[transcribe,mic,web,dev]'

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
2. **Faster Installation**: Skip compilation of native extensions when not needed
3. **Modular Architecture**: Separate CLI, web server, database, and transcription components
4. **Flexible Deployment**:
   - CLI-only servers without web dependencies
   - Web-only servers for viewing transcripts
   - VAD-only mode doesn't need PostgreSQL, transcription, or web server
5. **Clear Error Messages**: Users know exactly what to install when needed
6. **Platform-Specific**: Microphone recording only installed on desktop platforms
7. **Resource Optimization**: Install only the components needed for your use case
