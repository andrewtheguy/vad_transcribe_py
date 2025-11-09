# Web Interface Setup Guide

The web interface has been successfully integrated into your Whisper Transcribe project!

## Prerequisites

**Required Environment Variables**

The web server requires a PostgreSQL database connection:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/whisper_db"
```

Optional:
```bash
export DATABASE_TIMEOUT=10  # Connection timeout in seconds (default: 10)
```

The database schema will be automatically initialized when the web server starts. The server will create a `transcripts` table if it doesn't already exist.

## Quick Start

### Development Mode (Recommended for Development)

**Option 1: Run both frontend and backend together**
```bash
# Terminal 1: Start the backend API server
uv run python main.py web --dev

# Terminal 2: Start the frontend dev server
cd frontend
npm run dev
```

- Backend API: http://localhost:5002
- Frontend UI: http://localhost:5173

**Option 2: Run production build**
```bash
# Build the frontend first (one-time)
cd frontend && npm run build && cd ..

# Start the server (serves both API and built frontend)
uv run python main.py web
```

- Access at: http://localhost:5002

### Production Mode

```bash
# Build the frontend
cd frontend
npm run build
cd ..

# Run the server
uv run python main.py web --host 0.0.0.0 --port 5002
```

## CLI Options

```bash
uv run python main.py web [OPTIONS]

Options:
  --host HOST    Host to bind to (default: 0.0.0.0)
  --port PORT    Port to bind to (default: 5002)
  --dev          Enable development mode with hot reload and CORS
```

## Project Structure

```
whisper_transcribe_py/
├── frontend/                      # Vite + React + TypeScript frontend
│   ├── src/
│   │   ├── components/ui/        # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   └── tabs.tsx
│   │   ├── lib/
│   │   │   └── utils.ts          # Utility functions
│   │   ├── App.tsx               # Main shell UI
│   │   ├── main.tsx              # Entry point
│   │   └── index.css             # Tailwind styles
│   ├── dist/                     # Production build output
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── tsconfig.json
├── whisper_transcribe_py/
│   └── api/
│       ├── __init__.py
│       └── server.py             # FastAPI server
├── scripts/
│   └── build_frontend.sh         # Build helper script
└── main.py                       # CLI with new 'web' command

```

## Current Features

The web interface provides a fully functional real-time transcription system with browser-based microphone recording.

### Implemented Features
- **Browser Microphone Recording**: Record audio directly from the browser
- **Real-time Transcription**: Live transcription using HTTP streaming (PCM chunks over POST)
- **Session Management**: Create, track, and stop transcription sessions
- **Database Persistence**: All transcripts saved to PostgreSQL with timestamps
- **Transcript Retrieval**: Fetch transcripts by session or show name with pagination
- **Show Management**: List all shows and their transcripts from the database

### API Endpoints (Fully Implemented)

**Core Transcription**
- `GET /api/health` - Health check endpoint
- `POST /api/transcribe/stream/session` - Create a new streaming session
  - Request: `{"language": "en", "sample_rate": 16000}`
  - Response: `{"session_id": "uuid"}`
- `POST /api/transcribe/stream` - Send audio chunk for transcription
  - Query params: `session_id`, `start`, `sample_rate`, `language`
  - Body: Raw PCM audio bytes (signed 16-bit little-endian)
  - Response: `{"status": "queued", "session_id": "...", "samples": 1234}`
- `DELETE /api/transcribe/stream/{session_id}` - Stop a streaming session

**Transcript Access**
- `GET /api/transcribe/stream/{session_id}/transcripts` - Get transcripts for active session
- `GET /api/shows` - List all show names with statistics
- `GET /api/shows/{show_name}/transcripts` - Get paginated transcripts for a show

**Not Yet Implemented**
- `POST /api/transcribe/file` - File upload transcription (returns 501)

### Architecture

**Audio Streaming Protocol**: Uses HTTP POST (not WebSocket) for audio chunk submission
- Frontend captures microphone audio using Web Audio API
- Audio is resampled to 16kHz mono PCM
- Chunks are sent as raw binary data via POST requests
- Backend enqueues audio for VAD processing and transcription
- Transcripts are written to PostgreSQL database
- Frontend polls for new transcripts or can use session-based retrieval

## Technologies Used

### Frontend
- **Vite** - Fast build tool and dev server
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - High-quality component library
- **Lucide React** - Icon library

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server with hot reload
- **Static file serving** - Serves built frontend in production

## Build Scripts

### Manual Build
```bash
cd frontend
npm install
npm run build
```

### Using Helper Script
```bash
./scripts/build_frontend.sh
```

## Troubleshooting

### Database Connection Required
The web server **requires** a PostgreSQL database and will not start without it. If you see:
```
RuntimeError: DATABASE_URL environment variable is required for web server
```

Solution:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
```

Make sure PostgreSQL is running and the connection string is correct.

### Frontend not built
If you see "Frontend not built" error:
```bash
cd frontend
npm install
npm run build
```

### Port already in use
The default port is 5002. To change it:
```bash
uv run python main.py web --port 8080
```

### CORS issues in development
Make sure to use `--dev` flag when running the backend in development mode:
```bash
uv run python main.py web --dev
```

This enables CORS for the Vite dev server running on port 5173.

### Cannot run multiple instances
Only one instance can run at a time due to file locking. If you see:
```
Another instance is already running
```

Stop the other instance first, or use different modes (web vs mic vs stream).
