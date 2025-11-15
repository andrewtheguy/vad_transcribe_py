# Feature Flags

This document explains the optional dependency system for Whisper Transcribe.

## Overview

Whisper Transcribe uses Python's optional dependencies (extras) to allow installation without heavy transcription backends. This is similar to Rust's feature flags.

## Installation Options

### Full Installation (with transcription)

```bash
uv pip install -e '.[transcribe]'
```

**Includes:**
- All base dependencies
- `pywhispercpp` - Whisper.cpp Python bindings (~1GB)
- `faster-whisper` - CTranslate2-based Whisper (~1GB)

**Use when:**
- You need full transcription capabilities
- You want to use `mic`, `stream`, `file`, or `web` commands WITH transcription
- You need language-specific text output

### Lightweight Installation (VAD-only)

```bash
uv pip install -e .
```

**Includes:**
- All base dependencies (silero-vad, sounddevice, scipy, etc.)
- **Excludes:** pywhispercpp, faster-whisper

**Saves:** ~2GB disk space, faster installation

**Use when:**
- You only need voice activity detection (VAD)
- You only want to save audio segments (`--no-transcribe` mode)
- You're deploying to resource-constrained environments
- You want faster CI/CD builds

## Usage Examples

### With Lightweight Installation

```bash
# File mode: save VAD-detected speech segments without transcription
uv run python main.py file --file audio.mp3 --lang en --no-transcribe

# Mic mode: save speech segments from microphone
uv run python main.py mic --lang en --no-transcribe

# Stream mode: save speech segments from stream
uv run python main.py stream --config configs/mystream.toml --no-transcribe
```

### Attempting Transcription Without Dependencies

If you try to use transcription without installing the `[transcribe]` extra, you'll get a helpful error:

```
ImportError: pywhispercpp is not installed.
To use transcription, install with: uv pip install -e '.[transcribe]'
or use --no-transcribe flag for VAD-only mode.
```

## How It Works

### pyproject.toml Configuration

```toml
[project]
dependencies = [
    "silero-vad>=5.1.2,<6",
    "scipy>=1.14.1,<2",
    # ... other base dependencies
    # NOTE: pywhispercpp and faster-whisper are NOT here
]

[project.optional-dependencies]
transcribe = [
    "pywhispercpp>=1.3.3,<1.4",
    "faster-whisper>=1.0.0,<2",
]
```

### Lazy Loading with Error Handling

The transcription backends are imported lazily inside methods with try/except blocks:

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

This ensures:
1. Import errors only occur when transcription is actually attempted
2. Users get clear, actionable error messages
3. VAD-only functionality works without transcription dependencies

## Development

When developing, install with both transcribe and dev extras:

```bash
uv pip install -e '.[transcribe,dev]'
```

This includes:
- Transcription backends (pywhispercpp, faster-whisper)
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
uv pip install -e '.[transcribe,dev]'

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
2. **Faster Installation**: Skip compilation of native extensions
3. **Flexible Deployment**: Deploy only what you need
4. **Clear Error Messages**: Users know exactly what to install when needed
5. **Backward Compatible**: Existing full installations continue to work
