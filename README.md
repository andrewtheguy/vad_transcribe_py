# VAD Transcribe - File Transcription Tool

**VAD Transcribe** is a file-based audio transcription tool that combines voice activity detection (VAD) with AI-powered transcription. It uses [silero-vad](https://github.com/snakers4/silero-vad) to intelligently detect speech segments and offers two transcription backends:

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) (default) - OpenAI's Whisper large-v3-turbo via HuggingFace Transformers, multilingual ASR
- [Moonshine](https://github.com/usefulsensors/moonshine) - Fast ONNX-based ASR models (English: streaming, Chinese/Spanish: non-streaming). Auto-downloaded on first use.
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Alibaba's Qwen3-ASR via HuggingFace Transformers, 30-language ASR
- [Qwen3-ASR (Rust)](https://github.com/andrewtheguy/qwencandle/releases/tag/v0.0.3) - Qwen3-ASR via qwencandle Rust bindings (PyO3), supports CPU/Metal/CUDA

## Features

- **Streaming Audio Processing**: Audio is streamed from ffmpeg - never loads full file into memory
- **Smart Voice Detection**: Uses Silero VAD to detect speech segments
- **Flexible Backend Selection**: Choose between Whisper (default) or Moonshine backends
- **File Transcription**: Process audio files to JSON transcripts
- **Split Mode**: Save detected speech segments as Opus files
- **Multi-language Support**: Supports all languages available in OpenAI Whisper models
- **CLI-Based**: Simple command-line interface for batch processing

## Quick Start

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- ffmpeg - Only needed for file-based operations (not required for stdin WAV input)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Ubuntu/Debian)
sudo apt install ffmpeg
```

### Installation

```bash
# Clone the repository
git clone https://github.com/andrewtheguy/vad_transcribe_py.git
cd vad_transcribe_py

# Install all dependencies
uv sync
```

### Running (Development)

For development, run commands with `uv run`:

```bash
uv run vad-transcribe-py transcribe --file audio.wav --language en

uv run vad-transcribe-py split --file audio.wav
```

### Installing as a Tool (Global)

To install globally and run `vad-transcribe-py` directly:

```bash
# From GitHub Pages index (recommended)
uv tool install --extra-index-url https://andrewtheguy.github.io/vad_transcribe_py/simple/ 'vad-transcribe-py==VERSION'

# From GitHub release wheel directly
uv tool install "vad-transcribe-py @ https://github.com/andrewtheguy/vad_transcribe_py/releases/download/VERSION/vad_transcribe_py-VERSION-py3-none-any.whl"

# Or from GitHub source directly
uv tool install "vad-transcribe-py @ git+https://github.com/andrewtheguy/vad_transcribe_py.git@(ref, tag or branch)"

# Or from local clone
uv tool install "."
```

After installation:

```bash
vad-transcribe-py transcribe --file audio.wav --language en
vad-transcribe-py split --file audio.wav
```

---

## Transcribe Command

Transcribe audio to text using Whisper models with optional VAD segmentation.

```bash
vad-transcribe-py transcribe (--file PATH | --stdin) [OPTIONS]
```

### Transcribe Options

- `--file PATH`: Path to audio file (mutually exclusive with --stdin)
- `--stdin`: Read WAV audio (mono, 16kHz) from stdin (mutually exclusive with --file). Accepts 16-bit PCM, 32-bit PCM, or 32-bit float WAV. Always uses VAD, always outputs JSONL to stdout.
- `--output PATH`: Output path for JSONL transcript (default: stdout)
- `--language LANG`: Language code for transcription (required for moonshine, optional for other backends which auto-detect when omitted)
- `--model MODEL`: Model name (auto-selected if omitted). For whisper: HuggingFace short name or full ID (default: `large-v3-turbo` → `openai/whisper-large-v3-turbo`). For moonshine: use short names like `small-streaming`, `base`, `tiny` — these map to language-specific variants (e.g., `small-streaming` → `small-streaming-en`). Defaults: `small-streaming` (English), `base` (Chinese/Spanish).
- `--backend {whisper, moonshine, qwen-asr, qwen-asr-rs}`: Transcription backend (default: `whisper`)
- `--chinese-conversion {none, simplified, traditional}`: Chinese character conversion for zh/yue languages (default: none)
- `--threads N`: Number of CPU threads for inference. If omitted, moonshine defaults to `min(2, cpu_count)` and other backends leave threading unset. For whisper and qwen-asr, thread tuning applies to CPU execution and configures PyTorch CPU threading before model load. For moonshine, it configures ONNX Runtime sessions. For qwen-asr-rs, it sets the `RAYON_NUM_THREADS` environment variable.
- `--device {cpu, mps, metal, cuda}`: Device for whisper, qwen-asr, and qwen-asr-rs backends (default: auto-detect cuda > mps > cpu). Not supported by moonshine.
- `--no-condition`: Disable conditioning on previous segment output (whisper, qwen-asr, and qwen-asr-rs backends)
- `--no-sub-timestamps`: Disable sub-sentence timestamp splitting (whisper backend only)

VAD soft/hard limits are set automatically per backend (Whisper: 6s soft / 30s hard, Moonshine streaming: 6s / 60s, Moonshine non-streaming: 6s / 9s, Qwen3-ASR: 6s / 30s). Use the `split` command for manual VAD tuning.

### Transcribe Examples

**Transcribe audio file with VAD to stdout (streaming JSONL):**
```bash
uv run vad-transcribe-py transcribe --file audio.wav --language en
```

**Transcribe to file:**
```bash
uv run vad-transcribe-py transcribe --file audio.wav --output transcript.jsonl --language en
```

**Transcribe from stdin (mono 16kHz WAV — always uses VAD, outputs to stdout in JSONL format):**
```bash
# Pipe WAV audio from ffmpeg (16-bit PCM, 32-bit PCM, or 32-bit float all accepted)
ffmpeg -loglevel error -i video.mp4 -ac 1 -ar 16000 -f wav - | uv run vad-transcribe-py transcribe --stdin --language en

# Float32 WAV — no conversion needed, ffmpeg not required and can be replaced by other commands
ffmpeg -loglevel error -i video.mp4 -ac 1 -ar 16000 -f wav -acodec pcm_f32le - | uv run vad-transcribe-py transcribe --stdin --language en

# Or from an existing WAV file
cat audio.wav | uv run vad-transcribe-py transcribe --stdin --language en
```

**Use Moonshine backend (auto-selects model by language, ONNX models downloaded on first use):**
```bash
# English — uses small-streaming by default
uv run vad-transcribe-py transcribe --file audio.wav --backend moonshine --language en

# Chinese — uses base by default
uv run vad-transcribe-py transcribe --file audio.wav --backend moonshine --language zh

# Spanish — uses base by default
uv run vad-transcribe-py transcribe --file audio.wav --backend moonshine --language es

# Or specify model explicitly
uv run vad-transcribe-py transcribe --file audio.wav --backend moonshine --model tiny --language en
```

**Use Qwen3-ASR backend (auto-detects language when --language is omitted):**
```bash
# Via HuggingFace Transformers
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr

# Via Rust bindings (faster, supports Metal/CUDA)
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr-rs
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr-rs --device metal
```

**Use a different Whisper model:**
```bash
uv run vad-transcribe-py transcribe --file audio.wav --model large-v3
```

### Transcribe Output Format (JSONL)

Transcription outputs streaming JSONL (one JSON object per line). Each entry has a `type` field:

```jsonl
{"type": "stream_start"}
{"type": "segment_start", "timestamp": 0.5, "timestamp_formatted": "00:00:00.500"}
{"type": "transcript", "start": 0.5, "start_formatted": "00:00:00.500", "end": 2.3, "end_formatted": "00:00:02.300", "text": "Hello world"}
{"type": "segment_end", "timestamp": 2.3, "timestamp_formatted": "00:00:02.300"}
{"type": "segment_start", "timestamp": 3.1, "timestamp_formatted": "00:00:03.100"}
{"type": "transcript", "start": 3.1, "start_formatted": "00:00:03.100", "end": 5.8, "end_formatted": "00:00:05.800", "text": "This is a test"}
{"type": "segment_end", "timestamp": 5.8, "timestamp_formatted": "00:00:05.800"}
{"type": "stream_end"}
```

- `stream_start` / `stream_end`: Marks the beginning and end of the transcription stream
- `segment_start`: Marks the beginning of a VAD-detected speech segment
- `transcript`: Contains the transcribed text with start/end timestamps
- `segment_end`: Marks the end of a VAD-detected speech segment
- `*_formatted`: Human-readable timestamps in `hh:mm:ss.ms` format

---

## Split Command

Split audio into separate files based on VAD-detected speech segments (no transcription).

```bash
vad-transcribe-py split (--file PATH | --url URL) [OPTIONS]
```

### Split Options

- `--file PATH`: Path to audio file (mutually exclusive with --url)
- `--url URL`: URL to audio file (mutually exclusive with --file). Live streams not supported because there is no real use case for this.
- `--preserve-sample-rate`: Preserve original sample rate (default: downsample to 16kHz)
- `--format {opus, wav}`: Output format (default: opus)

**VAD tuning options** (split command only — transcribe uses backend-specific defaults):
- `--min-speech-seconds FLOAT`: Minimum speech duration in seconds (default: 3.0)
- `--soft-limit-seconds FLOAT`: Soft limit on speech segment duration in seconds (default: 60.0). Triggers adaptive silence detection.
- `--speech-threshold FLOAT`: VAD speech detection threshold 0.0-1.0 (default: 0.5)
- `--min-silence-duration-ms INT`: Minimum silence duration in ms to end segment (default: 2000)
- `--look-back-seconds FLOAT`: Look-back buffer in seconds for segment start (default: 0.5)

### Split Examples

**Split audio by VAD into Opus segments:**
```bash
uv run vad-transcribe-py split --file audio.wav
# Outputs to tmp/audio/
```

**Split with preserved sample rate:**
```bash
uv run vad-transcribe-py split --file audio.wav --preserve-sample-rate
```

**Split to WAV format:**
```bash
uv run vad-transcribe-py split --file audio.wav --format wav
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

Language support varies by backend:

- **Whisper**: All languages available in OpenAI Whisper models
- **Moonshine**: English (`en`), Chinese (`zh`), Spanish (`es`)
- **Qwen3-ASR / Qwen3-ASR (Rust)**: 30 languages — `zh`, `en`, `yue`, `ar`, `de`, `fr`, `es`, `pt`, `id`, `it`, `ko`, `ru`, `th`, `vi`, `ja`, `tr`, `hi`, `ms`, `nl`, `sv`, `da`, `fi`, `pl`, `cs`, `fil`, `fa`, `el`, `ro`, `hu`, `mk`

**Chinese Character Conversion:** For Chinese language codes (`zh` and `yue`), you can optionally convert characters using `--chinese-conversion`:
- `none` (default): No conversion, output as-is from Whisper
- `simplified`: Convert to Simplified Chinese (zh-Hans)
- `traditional`: Convert to Traditional Chinese (zh-Hant)

Conversion is powered by [zhconv-rs](https://github.com/Xmader/zhconv-rs).

## Performance Notes

- **Whisper** backend: Best for multilingual transcription, 30-second hard limit per segment
- **Moonshine** backend: Fast ONNX inference. English (streaming, 60s hard limit), Chinese/Spanish (non-streaming, 9s hard limit)
- **Qwen3-ASR** backend: 30-language support, 30-second hard limit per segment
- **Qwen3-ASR (Rust)** backend: Same model via Rust bindings, supports CPU/Metal/CUDA via `--device`. Recommended over `qwen-asr` for Metal, as it leaks significantly less memory.
- Device auto-detected for torch-based backends (whisper, qwen-asr, qwen-asr-rs): CUDA > MPS > CPU. Override with `--device`. Moonshine uses ONNX runtime (CUDA or CPU)
- If `--threads` is omitted, moonshine defaults to `min(2, cpu_count)` and other backends use their runtime defaults
- For whisper and qwen-asr on CPU, `--threads` primes common BLAS/OpenMP env vars before import, sets PyTorch intra-op threads, and keeps inter-op threads at 1
- Larger Whisper models (e.g., `large-v3`) provide better accuracy but require more memory
- Moonshine supports English, Chinese, and Spanish only

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --group dev

# Lint
uv run ruff check

# Type check
uv run basedpyright

# Run tests
uv run pytest
```


## Requirements

- Whisper: HuggingFace model files (~1-3GB depending on model size) - downloaded automatically on first use
- Moonshine: ONNX model files (~10-100MB) - downloaded automatically from `download.moonshine.ai` on first use

## Known Limitations

- File-based transcription only (no real-time/live transcription)
- Live streams not supported (URLs must have fixed duration)
- VAD enforces per-backend hard limits on segment duration via force-split (30s Whisper, 60s Moonshine streaming, 9s Moonshine non-streaming, 30s Qwen3-ASR)
- No database persistence (outputs to JSONL files or Opus segments)
- No web interface

## License

See LICENSE file for details.
