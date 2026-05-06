# VAD Transcribe - File Transcription Tool

**VAD Transcribe** is a file-based audio transcription tool that combines voice activity detection (VAD) with AI-powered transcription. It uses [silero-vad](https://github.com/snakers4/silero-vad) to intelligently detect speech segments and offers several transcription backends:

- [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) (default) - OpenAI's Whisper large-v3-turbo via HuggingFace Transformers, multilingual ASR
- [Moonshine](https://github.com/usefulsensors/moonshine) - Fast ONNX-based ASR models (English: streaming, Chinese/Spanish: non-streaming). Auto-downloaded on first use.
- [Qwen3-ASR (Rust)](https://github.com/andrewtheguy/qwencandle/releases/tag/v0.0.3) - Qwen3-ASR via qwencandle Rust bindings (PyO3), supports CPU/Metal/CUDA
- [Qwen3-ASR (MLX)](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16) - Qwen3-ASR via [mlx-audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon (Metal). Default model: `mlx-community/Qwen3-ASR-0.6B-bf16` (best accuracy per published WER benchmark)
- [GLM-ASR](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) - Z.ai GLM-ASR Nano via HuggingFace Transformers. Default model: `zai-org/GLM-ASR-Nano-2512` (1.5B parameters per upstream model card, BF16)
- [GLM-ASR (MLX)](https://huggingface.co/mlx-community/GLM-ASR-Nano-2512-8bit) - GLM-ASR Nano via [mlx-audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon (Metal). Default model: `mlx-community/GLM-ASR-Nano-2512-8bit` (same 1.5B-parameter network as the upstream model, 8-bit MLX quantization, ~2.4 GB on disk)
- [NVIDIA Whisper](https://build.nvidia.com/openai/whisper-large-v3) - OpenAI whisper-large-v3 hosted by NVIDIA via Riva gRPC (no local GPU required). Optional dep: `uv sync --extra nvidia-whisper`. Requires `NVIDIA_API_KEY` in `.env`.

## Features

- **Streaming Audio Processing**: Audio is streamed from ffmpeg - never loads full file into memory
- **Smart Voice Detection**: Uses Silero VAD to detect speech segments
- **Flexible Backend Selection**: Choose between Whisper (default), Moonshine, Qwen3-ASR (Rust/MLX), or GLM-ASR (Transformers/MLX) backends
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

### Development

```bash
# Clone the repository
git clone https://github.com/andrewtheguy/vad_transcribe_py.git
cd vad_transcribe_py

# Install all dependencies
uv sync
```

For development, run commands with `uv run`:

```bash
uv run vad-transcribe-py transcribe --file audio.wav --language en

uv run vad-transcribe-py split --file audio.wav
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
- `--model MODEL`: Model name (auto-selected if omitted). Defaults: `large-v3-turbo` → `openai/whisper-large-v3-turbo` (whisper), `zai-org/GLM-ASR-Nano-2512` (glm-asr), `mlx-community/Qwen3-ASR-0.6B-bf16` (qwen-asr-mlx), `mlx-community/GLM-ASR-Nano-2512-8bit` (glm-asr-mlx); moonshine picks `small-streaming` for English and `base` for Chinese/Spanish.
    - **Local directory paths** are accepted by **whisper**, **qwen-asr-rs**, and **glm-asr**.
    - **GGUF files** are accepted only by **qwen-asr-rs** (qwencandle resolves the format from the path).
- `--backend {whisper, moonshine, qwen-asr-rs, qwen-asr-mlx, glm-asr, glm-asr-mlx, nvidia-whisper}`: Transcription backend (default: `whisper`). `nvidia-whisper` calls the hosted whisper-large-v3 endpoint at `build.nvidia.com` and requires `NVIDIA_API_KEY` in `.env` (see [NVIDIA Whisper backend](#nvidia-whisper-backend)).
- `--chinese-conversion {none, simplified, traditional}`: Chinese character conversion for zh/yue languages (default: none)
- `--threads N`: Number of CPU threads for inference (default: `min(2, cpu_count)` for moonshine, none for other backends). For qwen-asr-rs, sets the `RAYON_NUM_THREADS` environment variable. Ignored by qwen-asr-mlx, glm-asr-mlx, and glm-asr.
- `--device {cpu, mps, metal, cuda}`: Device for whisper, qwen-asr-rs, and glm-asr backends (default: auto-detect cuda > mps > cpu). Not supported by moonshine. The qwen-asr-mlx and glm-asr-mlx backends always use Metal and ignore this flag.
- `--no-condition`: Disable conditioning on previous segment output (whisper, qwen-asr-rs, and qwen-asr-mlx backends; ignored by glm-asr and glm-asr-mlx, which do not support conditioning)
- `--no-sub-timestamps`: Disable sub-sentence timestamp splitting (whisper backend only)

VAD soft/hard limits are set automatically per backend (Whisper: 6s soft / 60s hard, NVIDIA Whisper: 6s / 30s, Moonshine streaming: 6s / 60s, Moonshine non-streaming: 6s / 9s, Qwen3-ASR Rust/MLX: 30s / 60s, GLM-ASR (Transformers/MLX): 30s / 60s). Use the `split` command for manual VAD tuning.

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
# Via Rust bindings (qwencandle), supports CPU/Metal/CUDA
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr-rs
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr-rs --device metal

# Via MLX (Apple Silicon only, default model: mlx-community/Qwen3-ASR-0.6B-bf16)
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr-mlx

# Override MLX quantization (bf16 default, or 8bit/6bit/5bit/4bit for smaller/faster)
uv run vad-transcribe-py transcribe --file audio.wav --backend qwen-asr-mlx \
    --model mlx-community/Qwen3-ASR-0.6B-8bit
```

**Use GLM-ASR backend (auto-detects language when --language is omitted):**
```bash
uv run vad-transcribe-py transcribe --file audio.wav --backend glm-asr
uv run vad-transcribe-py transcribe --file audio.wav --backend glm-asr --language en
uv run vad-transcribe-py transcribe --file audio.wav --backend glm-asr --device cuda
```

**Use GLM-ASR (MLX) backend on Apple Silicon (Metal-only, 8-bit quantized by default):**
```bash
uv run vad-transcribe-py transcribe --file audio.wav --backend glm-asr-mlx
```

**Use a different Whisper model:**
```bash
uv run vad-transcribe-py transcribe --file audio.wav --model large-v3
```

#### NVIDIA Whisper backend

The `nvidia-whisper` backend calls OpenAI whisper-large-v3 hosted on `build.nvidia.com` via Riva gRPC. No local GPU is needed; audio segments are uploaded per VAD chunk.

```bash
# 1. Install the optional dep (pulls in nvidia-riva-client and grpc).
uv sync --extra nvidia-whisper

# 2. Get a key at https://build.nvidia.com/openai/whisper-large-v3 and put it in .env:
cp .env.example .env
# then edit .env and set NVIDIA_API_KEY=nvapi-...

# 3. Transcribe.
uv run vad-transcribe-py transcribe --file audio.wav --backend nvidia-whisper --language en

# Omit --language for Riva auto-detect (sends language_code=multi):
uv run vad-transcribe-py transcribe --file audio.wav --backend nvidia-whisper
```

`--model`, `--device`, and `--threads` are ignored for this backend (server-side execution, fixed function-id).

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
- **Qwen3-ASR (Rust) / Qwen3-ASR (MLX)**: 30 languages — `zh`, `en`, `yue`, `ar`, `de`, `fr`, `es`, `pt`, `id`, `it`, `ko`, `ru`, `th`, `vi`, `ja`, `tr`, `hi`, `ms`, `nl`, `sv`, `da`, `fi`, `pl`, `cs`, `fil`, `fa`, `el`, `ro`, `hu`, `mk`
- **GLM-ASR**: Uses the same explicit language-code set as Qwen3-ASR when `--language` is provided; omit `--language` to use the model's default transcription prompt

**Chinese Character Conversion:** For Chinese language codes (`zh` and `yue`), you can optionally convert characters using `--chinese-conversion`:
- `none` (default): No conversion, output as-is from Whisper
- `simplified`: Convert to Simplified Chinese (zh-Hans)
- `traditional`: Convert to Traditional Chinese (zh-Hant)

Conversion is powered by [zhconv-rs](https://github.com/Xmader/zhconv-rs).

## Performance Notes

- **Whisper** backend: Best for multilingual transcription, 60-second hard limit per segment (HF Transformers handles long-form decoding by chunking the 30s native window internally when `return_timestamps=True`, which is the default)
- **NVIDIA Whisper** backend: Hosted whisper-large-v3 via Riva gRPC, 30-second hard limit per segment to stay under the NVCF 60s request timeout
- **Moonshine** backend: Fast ONNX inference. English (streaming, 60s hard limit), Chinese/Spanish (non-streaming, 9s hard limit)
- **Qwen3-ASR (Rust)** backend: 30-language support via qwencandle Rust bindings, supports CPU/Metal/CUDA via `--device`, 60-second hard limit per segment
- **Qwen3-ASR (MLX)** backend: Same model via mlx-audio on Apple Silicon (Metal only, `--device` ignored). Default `mlx-community/Qwen3-ASR-0.6B-bf16` (2.29 % WER on LibriSpeech test-clean per published benchmark). 8-bit (`-8bit`) is near-lossless; 4-bit (`-4bit`) is fastest but with measurable WER loss.
- **GLM-ASR** backend: Transformers-based GLM-ASR Nano inference, supports CPU/MPS/CUDA via `--device`, 60-second hard limit per segment
- **GLM-ASR (MLX)** backend: Same 1.5B-parameter network as the Transformers backend, 8-bit MLX-quantized, served via mlx-audio on Apple Silicon (Metal only, `--device` ignored). Default `mlx-community/GLM-ASR-Nano-2512-8bit` (~2.4 GB on disk). English and Chinese; the model auto-detects language and `--language` is not forwarded.
- Device auto-detected for torch-based backends (whisper, qwen-asr-rs, glm-asr): CUDA > MPS > CPU. Override with `--device`. Moonshine uses ONNX runtime (CUDA or CPU).
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
- GLM-ASR: HuggingFace model files for `zai-org/GLM-ASR-Nano-2512` - downloaded automatically on first use
- GLM-ASR (MLX): HuggingFace model files for `mlx-community/GLM-ASR-Nano-2512-8bit` (~2.4 GB on disk; same 1.5B-parameter network as the upstream model, 8-bit MLX-quantized) - downloaded automatically on first use (Apple Silicon only)

## Known Limitations

- File-based transcription only (no real-time/live transcription)
- Live streams not supported (URLs must have fixed duration)
- VAD enforces per-backend hard limits on segment duration via force-split (60s Whisper, 30s NVIDIA Whisper, 60s Moonshine streaming, 9s Moonshine non-streaming, 60s Qwen3-ASR Rust/MLX, 60s GLM-ASR Transformers/MLX)
- No database persistence (outputs to JSONL files or Opus segments)
- No web interface

## License

See LICENSE file for details.
