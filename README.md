voice activity detection on an audio stream using silero-vad and transcribing the detected voice segments using whispercpp

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
- `--lang`: Language code (e.g., 'en', 'zh', 'yue')
- `--output`: Path for JSON output file

**Optional arguments:**
- `--model`: Whisper model size (default: 'large-v3-turbo')
- `--n-threads`: Number of threads for Whisper (default: 1)

### 2. Transcribe from microphone
Record from microphone and transcribe in real-time.

```commandline
uv run python main.py mic --lang en
```

**Behavior:**
- Records audio from default microphone
- Transcribes speech segments to console output
- Persists transcripts to the `transcripts` table with show name `microphone`
- Press 'q' + Enter to quit

**Required arguments:**
- `--lang`: Language code (e.g., 'en', 'zh', 'yue')

**Optional arguments:**
- `--n-threads`: Number of threads for Whisper (default: 1)

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
- Press 'q' to quit

**Required arguments:**
- `--config`: Path to TOML configuration file

**Optional arguments:**
- `--model`: Whisper model size (overrides config file, default: 'large-v3-turbo')
- `--n-threads`: Number of threads for Whisper (overrides config file, default: 1)

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
transcribe_model_size = 'large-v3-turbo'  # Whisper model size (default: 'large-v3-turbo')
n_threads = 4                             # Number of threads for Whisper (default: 1)
```

**Note:** Command line arguments take priority over config file settings. For example:
```commandline
uv run python main.py stream --config configs/rthk2.toml --model base --n-threads 8
```
This will use the 'base' model and 8 threads regardless of what's in the config file.

## Queue backlog limiter

Long-running streams can fall behind if Whisper processing slows down. To avoid unbounded memory growth, producers and consumers share a `QueueBacklogLimiter` (defined in `whisper_transcribe_py/audio_transcriber.py`) that keeps track of how many seconds of unprocessed audio are buffered:
- If adding a chunk would push the backlog over `max_seconds`, the chunk is dropped and a warning is printed to stderr. This favors staying live over perfect recall.
- The microphone CLI uses `CLI_QUEUE_TIME_LIMIT_SECONDS` (default 60 s) to cap the recorder queue.
- Stream configs wrap both the downloader and the speech detector with the same limiter so dropped chunks reference the show name.

To change the cap, edit `CLI_QUEUE_TIME_LIMIT_SECONDS` in `main.py` or create your own limiter and pass it to `process_queue`, `process_mic`, or `stream_url_thread`. Matching limiter instances for producers and consumers ensures dropped time is accounted for correctly.
