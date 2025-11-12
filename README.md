# Whisper Transcribe

**OpenAI Whisper Transcribe** is a real-time speech transcription system that combines voice activity detection (VAD) with AI-powered transcription. It uses [silero-vad](https://github.com/snakers4/silero-vad) to intelligently detect speech segments and offers two transcription backends:
- [whispercpp](https://github.com/absadiki/pywhispercpp) (default) - Python binding for whisper.cpp (Faster on Mac with MPS support)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2-based implementation for faster GPU/CPU inference (faster for CPU inference without MPS support)

## Key Features

- **Smart Voice Detection**: Uses Silero VAD to detect speech segments, reducing unnecessary processing
- **Flexible Backend Selection**: Choose between whisper.cpp (default) or faster-whisper backends for transcription
- **Multiple Input Sources**: Process audio files, microphone input, or live audio streams
- **Real-time Transcription**: Low-latency transcription for live audio sources
- **Web Interface**: Modern web UI for browser-based microphone recording and transcription
- **Database Persistence**: Stores transcripts in PostgreSQL with timestamps for easy retrieval
- **Multi-language Support**: Supports all languages available in OpenAI Whisper models through cli arguments. Currently English, Spanish, Mandarin, and Cantonese is selectable in the web UI to reduce clutter, but others can be added to the web UI dropdown list.
- **Queue Backlog Management**: Prevents memory overflow during long-running sessions

## Quick Start

### Database Setup (Required for most features)

Most features (web interface, microphone recording, and stream processing) require a PostgreSQL database to store transcripts. Set this up first:

```bash
# Option 1: Create a .env file in the project root (Recommended)
echo 'DATABASE_URL=postgresql://user:password@localhost/dbname' > .env

# Option 2: Export as environment variable
export DATABASE_URL="postgresql://user:password@localhost/dbname"
```

**Note**: The project uses python-dotenv to automatically load environment variables from a `.env` file in the project root. Only the `file` command works without a database.

## Backend Selection

Whisper Transcribe supports two transcription backends with different performance characteristics:

### whisper_cpp (Default)
- Python bindings for the original whisper.cpp implementation
- Good CPU performance with optimized inference
- Lower memory footprint
- Best for: CPU-only environments, embedded systems, or when memory is constrained

### faster-whisper
- Based on CTranslate2 for optimized GPU/CPU inference
- Automatic device detection (uses GPU if CUDA available, falls back to CPU)
- Faster inference speed, especially on GPU
- Best for: GPU-enabled systems, high-throughput processing, or when speed is priority

**Usage examples:**
```bash
# Use default whisper_cpp backend
uv run python main.py file --file audio.mp3 --lang en --output output.json

# Use faster-whisper backend with GPU acceleration
uv run python main.py file --file audio.mp3 --lang en --output output.json --backend faster_whisper

# Set backend in config file for stream processing
echo "backend = 'faster_whisper'" >> configs/mystream.toml
uv run python main.py stream --config configs/mystream.toml
```

**Note:** Both backends use the same custom VAD logic (Silero VAD) to detect speech segments. The faster-whisper backend has its built-in VAD disabled to prevent duplicate processing.

### Web Interface

The easiest way to use Whisper Transcribe is through the web interface:

```bash
# Development mode (hot reload, separate frontend dev server)
uv run python main.py web --dev

# Production mode (serves built frontend)
cd frontend && npm run build && cd ..
uv run python main.py web
```

Access the web UI at `http://localhost:5002` and start recording from your browser microphone.

### Command Line

For batch processing or automation, use the CLI commands below.

## Commands

### 1. Transcribe from file
Process an audio file and save the transcript as JSON.

```commandline
uv run python main.py file --file /path/to/file --lang en --output /path/to/output.json
```

**Behavior:**
- Processes audio from a file
- Outputs JSON transcript to the specified `--output` path
- Does NOT save to database
- Uses relative timestamps (seconds from start of file)

**Required arguments:**
- `--file`: Path to audio file
- `--lang`: Language code (e.g., 'en', 'es', 'zh', 'yue', etc.)
- `--output`: Path for JSON output file

**Optional arguments:**
- `--model`: Whisper model size (default: 'large-v3-turbo')
- `--n-threads`: Number of threads for Whisper (default: 1)
- `--backend`: Transcription backend - `whisper_cpp` or `faster_whisper` (default: 'whisper_cpp')

### 2. Transcribe from microphone
Record from microphone and transcribe in real-time.

```commandline
uv run python main.py mic --lang en
```

**Behavior:**
- Records audio from default microphone
- Transcribes speech segments to console output
- Persists transcripts to the `transcripts` table with show name `microphone`

**Required arguments:**
- `--lang`: Language code (e.g., 'en', 'zh', 'yue')

**Optional arguments:**
- `--n-threads`: Number of threads for Whisper (default: 1)
- `--backend`: Transcription backend - `whisper_cpp` or `faster_whisper` (default: 'whisper_cpp')

**Required environment variables:**
- `DATABASE_URL`: PostgreSQL connection string (e.g., `postgresql://user:pass@localhost/db`)

**Optional environment variables:**
- `DATABASE_TIMEOUT`: Connection timeout in seconds (default: 10)

### 3. Transcribe from audio stream
Stream audio from URL and save transcripts to database.

```commandline
uv run python main.py stream --config configs/rthk2.toml
```

**Behavior:**
- Streams audio from URL specified in config file
- Saves transcripts to PostgreSQL database
- Creates `transcripts` table automatically if it doesn't exist
- Uses wall clock timestamps

**Required arguments:**
- `--config`: Path to TOML configuration file

**Optional arguments:**
- `--model`: Whisper model size (overrides config file, default: 'large-v3-turbo')
- `--n-threads`: Number of threads for Whisper (overrides config file, default: 1)
- `--backend`: Transcription backend - `whisper_cpp` or `faster_whisper` (overrides config file, default: 'whisper_cpp')

**Required environment variables:**
- `DATABASE_URL`: PostgreSQL connection string (e.g., `postgresql://user:pass@localhost/db`)

**Optional environment variables:**
- `DATABASE_TIMEOUT`: Connection timeout in seconds (default: 10)

**Config file format (TOML):**
```toml
# Required fields
url = 'https://example.com/stream.m3u8'  # Stream URL
show_name = 'my_show'                     # Show identifier for database
language = 'en'                           # Language code (e.g., 'en', 'zh', 'yue')

# Optional fields
model = 'large-v3-turbo'      # Whisper model name or path (default: 'large-v3-turbo')
n_threads = 4                 # Number of threads for Whisper (default: 1)
backend = 'faster_whisper'    # Transcription backend: 'whisper_cpp' or 'faster_whisper' (default: 'whisper_cpp')
```

**Note:** Command line arguments take priority over config file settings. For example:
```commandline
uv run python main.py stream --config configs/rthk2.toml --model base --n-threads 8
```
This will use the 'base' model and 8 threads regardless of what's in the config file.

## Queue backlog limiter

Long-running streams can fall behind if Whisper processing slows down. To avoid unbounded memory growth, producers and consumers share a `QueueBacklogLimiter` (defined in `whisper_transcribe_py/audio_transcriber.py`) that keeps track of how many seconds of unprocessed audio are buffered:
- If adding a chunk would push the backlog over `max_seconds`, the chunk is dropped and a warning is printed to stderr. This favors staying live over perfect recall.
- The microphone CLI uses `QUEUE_TIME_LIMIT_SECONDS` (default 60 s, defined in `whisper_transcribe_py/audio_transcriber.py`) to cap the recorder queue.
- Consumers resume normal processing when the backlog shrinks below `QUEUE_RESUME_LIMIT_SECONDS` (default 15 s). This hysteresis prevents rapid oscillation between dropping and accepting chunks.
- Stream configs wrap both the downloader and the speech detector with the same limiter so dropped chunks reference the show name.

To change the caps, edit `QUEUE_TIME_LIMIT_SECONDS` or `QUEUE_RESUME_LIMIT_SECONDS` in `whisper_transcribe_py/audio_transcriber.py`, or create your own limiter and pass it to `process_queue`, `process_mic`, or `stream_url_thread`. Matching limiter instances for producers and consumers ensures dropped time is accounted for correctly.

## Web Interface

The web interface provides a modern, user-friendly way to interact with Whisper Transcribe through your browser. It supports real-time microphone recording and transcription with instant feedback.

### Features

- **Browser-based Recording**: Record audio directly from your browser microphone
- **Real-time Transcription**: See transcripts appear as you speak
- **Session Management**: Each recording session is tracked with a unique ID
- **Transcript History**: View all transcripts stored in the database by show name
- **RESTful API**: Access all functionality programmatically via HTTP endpoints

### API Endpoints

**Session Management**
- `POST /api/transcribe/stream/session` - Create a new streaming session
  - Request body: `{"language": "en", "sample_rate": 16000}`
  - Returns: `{"session_id": "uuid-here"}`

**Audio Streaming**
- `POST /api/transcribe/stream?session_id=<id>&start=<seconds>&sample_rate=16000&language=en` - Send audio chunk
  - Request body: Raw PCM audio bytes (signed 16-bit little-endian)
  - Returns: `{"status": "queued", "session_id": "...", "samples": 1234}`

**Transcript Retrieval**
- `GET /api/transcribe/stream/{session_id}/transcripts?limit=1000` - Get transcripts for a session
  - Returns latest transcripts with timestamps and content
- `GET /api/shows` - List all show names with metadata
- `GET /api/shows/{show_name}/transcripts?offset=0&limit=50` - Get paginated transcripts for a show

**Session Control**
- `DELETE /api/transcribe/stream/{session_id}` - Stop and close a session

**Health Check**
- `GET /api/health` - Check server status

### Environment Variables

**Required for: web, mic, and stream commands**

- `DATABASE_URL` (required): PostgreSQL connection string
  - Example: `postgresql://user:password@localhost:5432/whisper_db`
  - Used by: `web`, `mic`, and `stream` commands
  - Not required for: `file` command (saves to JSON instead)
- `DATABASE_TIMEOUT` (optional): Connection timeout in seconds (default: 10)

You can set these in a `.env` file in the project root:

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost:5432/whisper_db
DATABASE_TIMEOUT=10
```

The application will automatically load these values using python-dotenv.

### Running the Web Server

See the [Web Setup Guide](WEB_SETUP.md) for detailed instructions on setting up and running the web interface.

## Technology Stack

**Backend**
- FastAPI - Modern Python web framework
- Uvicorn - ASGI server with hot reload support
- PostgreSQL - Transcript persistence
- silero-vad - Voice activity detection
- pywhispercpp - Whisper.cpp Python bindings (default backend)
- faster-whisper - CTranslate2-based Whisper implementation (alternative backend)

**Frontend**
- Vite - Fast build tool and dev server
- React 18 - UI framework
- TypeScript - Type safety
- Tailwind CSS - Utility-first styling
- shadcn/ui - High-quality component library
