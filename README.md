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

### Installation

Whisper Transcribe supports optional dependencies for different use cases:

**Full installation (all features):**
```bash
# Install with all features: transcription, microphone, and web server
uv pip install -e '.[transcribe,mic,web]'
```

**CLI-only (transcription + microphone + database):**
```bash
# Install for CLI usage with transcription to database
uv pip install -e '.[transcribe,mic]'
```

**Web server (view and transcribe):**
```bash
# Install web server with transcription capabilities
uv pip install -e '.[web,transcribe,mic]'
```

**Web server (view-only):**
```bash
# Install web server to view existing transcripts only
uv pip install -e '.[web]'
```

**Transcription to JSON (no database, no microphone):**
```bash
# Install transcription for file processing to JSON output
uv pip install -e '.[transcribe]'
```

**Microphone recording only (no transcription, no database):**
```bash
# Install microphone recording without transcription (saves ~2GB)
uv pip install -e '.[mic]'
```

**Lightweight installation (VAD-only):**
```bash
# Install without transcription, microphone, database, or web server
uv pip install -e .
```

#### Optional dependency groups:

- **`[transcribe]`** - Transcription backends:
  - `pywhispercpp` - Whisper.cpp Python bindings
  - `faster-whisper` - CTranslate2-based Whisper implementation

- **`[mic]`** - Microphone recording (desktop only):
  - `sounddevice` - Audio capture from microphone
  - **Note:** Only works on desktop platforms (Windows, Mac, Linux with audio hardware)

- **`[web]`** - Web server:
  - `fastapi` - Modern Python web framework
  - `uvicorn[standard]` - ASGI server

**Note:** PostgreSQL database adapter (`psycopg[binary]`) is included in base dependencies for all installations.

### Database Setup

A PostgreSQL database is required for:
- CLI transcription to database (`mic` and `stream` commands with `--transcribe`)
- Web server (for viewing and storing transcripts)

**Not required for:**
- File transcription to JSON (`file` command outputs to JSON file)
- VAD-only mode with `--no-transcribe` flag (saves audio segments only)

**Setup:**

```bash
# Option 1: Create a .env file in the project root (Recommended)
echo 'DATABASE_URL=postgresql://user:password@localhost/dbname' > .env

# Option 2: Export as environment variable
export DATABASE_URL="postgresql://user:password@localhost/dbname"
```

**Note**: The project uses python-dotenv to automatically load environment variables from a `.env` file in the project root. Database is only required for transcription mode. VAD-only mode (`--no-transcribe`) saves audio files and does not require a database.

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
- The microphone CLI uses `QUEUE_TIME_LIMIT_SECONDS` (default 120 s, defined in `whisper_transcribe_py/audio_transcriber.py`) to cap the recorder queue. This must be at least 2x the maximum speech segment duration (default 60s) to ensure a single segment never exceeds queue capacity.
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

**SQLite Periodic Backup (optional)**

For SQLite storage mode, you can enable automatic periodic backups to remote storage:

- `PERIODIC_UPLOAD_ENABLED` (optional): Enable automatic backups
  - Values: `yes` to enable, any other value to disable
  - Default: disabled
  - When enabled, creates SQLite backups at configured intervals and uploads to remote storage via rclone

- `SQLITE_BACKUP_INTERVAL_SECONDS` (optional): Backup interval in seconds
  - Default: `3600` (1 hour)
  - Minimum: `1` second
  - Example: `60` for 1-minute backups, `300` for 5-minute backups
  - Invalid values fall back to default of 3600 seconds

- `SQLITE_BACKUP_DEST_DIR` (required if backups enabled): Remote backup destination directory
  - Example: `/backups/audio_transcripts`
  - Used with rclone remote named `remote` (configured via rclone)
  - Backups are organized by show name: `{SQLITE_BACKUP_DEST_DIR}/{show_name}/`

**How periodic backups work:**
1. At each interval (default: 1 hour), records the maximum ID from the database
2. Creates a SQLite backup with timestamp: `{show_name}_YYYYMMDD_HHMMSS.sqlite`
3. Uploads the backup to remote storage using `rclone move`
4. If upload succeeds, deletes backed-up records from the source database to save space
5. Backups are stored locally in `./tmp/speech_sqlite_backup/{show_name}/` before upload

**Prerequisites:**
- rclone must be installed and configured
- The remote connection must be accessible (SFTP, S3, etc.)
- Connection test is performed at startup to fail fast if remote is not accessible

**Rclone Configuration:**

The application uses rclone environment variables (not config file) to connect to remote storage. You must configure a remote named `remote` using environment variables:

**For SFTP (example):**
```bash
RCLONE_CONFIG_REMOTE_TYPE=sftp
RCLONE_CONFIG_REMOTE_HOST=example.com
RCLONE_CONFIG_REMOTE_USER=username
RCLONE_CONFIG_REMOTE_PASS=<obscured_password>  # Use 'rclone obscure' to generate
RCLONE_CONFIG_REMOTE_PORT=22
```

**For S3-compatible storage (example):**
```bash
RCLONE_CONFIG_REMOTE_TYPE=s3
RCLONE_CONFIG_REMOTE_PROVIDER=AWS
RCLONE_CONFIG_REMOTE_ACCESS_KEY_ID=your_access_key
RCLONE_CONFIG_REMOTE_SECRET_ACCESS_KEY=your_secret_key
RCLONE_CONFIG_REMOTE_REGION=us-east-1
RCLONE_CONFIG_REMOTE_ENDPOINT=https://s3.amazonaws.com
```

**For other storage providers:**
- See [rclone documentation](https://rclone.org/docs/) for your specific backend
- All environment variables follow the pattern: `RCLONE_CONFIG_REMOTE_<OPTION>=value`
- Remote name must be `REMOTE` (case-insensitive)

**Generate obscured password for SFTP:**
```bash
rclone obscure "your-password"
# Copy the output to RCLONE_CONFIG_REMOTE_PASS
```

You can set these in a `.env` file in the project root:

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost:5432/whisper_db
DATABASE_TIMEOUT=10

# Optional: Enable SQLite periodic backups
PERIODIC_UPLOAD_ENABLED=yes
SQLITE_BACKUP_INTERVAL_SECONDS=3600  # Default: 3600 seconds (1 hour)
SQLITE_BACKUP_DEST_DIR=/backups/audio_transcripts

# Rclone remote configuration (SFTP example)
RCLONE_CONFIG_REMOTE_TYPE=sftp
RCLONE_CONFIG_REMOTE_HOST=backup.example.com
RCLONE_CONFIG_REMOTE_USER=backup_user
RCLONE_CONFIG_REMOTE_PASS=<obscured_password>
RCLONE_CONFIG_REMOTE_PORT=22
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

## Known Limitations

### Error Handling in Callbacks

**TODO:** The current implementation lacks proper error handling for failures in VAD processing and transcription callbacks:

- **Non-livestream mode (file transcription):** Errors in the transcription callback should abort the program immediately with an appropriate error message, since partial results may be inconsistent or incomplete.

- **Livestream mode (mic/stream commands):** Errors in the transcription callback should reset the transcription state and emit a `TranscriptionNotice` to indicate a gap in the transcript. This allows the stream to continue processing new audio while clearly marking the disruption in the transcript timeline.

Currently, exceptions in callbacks may cause silent failures or undefined behavior, such as the queue got stuck accumulating without being processed when a database error occurs during transcription. Implementing proper error boundaries would improve reliability for long-running transcription sessions.

### Session Termination Notice

**TODO:** When quitting livestream mode (mic/stream commands), the system should emit a `TranscriptionNotice` to indicate the end of the transcription session. This would provide a clear marker in the transcript timeline showing when the session was intentionally terminated, distinguishing it from gaps caused by errors or dropped audio.
