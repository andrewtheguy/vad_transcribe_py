# Web Interface Setup Guide

The web interface has been successfully integrated into your Whisper Transcribe project!

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

The web interface is a **shell/skeleton** focused on microphone recording, ready for future implementation:

### Shell UI Components
- **Microphone Recording**: Centered, focused interface for live microphone recording
- **Results Panel**: Placeholder for real-time transcription display

### API Endpoints (Placeholders)
- `GET /api/health` - Health check (working)
- `POST /api/transcribe/file` - File transcription (not implemented)
- `WebSocket /api/transcribe/stream` - Streaming transcription (not implemented)
- `GET /api/sessions` - Session management (not implemented)

## Next Steps for Implementation

To add microphone recording functionality:

1. **WebSocket Audio Streaming**: Implement WebSocket endpoint at `/api/transcribe/stream` for browser audio streaming
2. **Frontend Audio Capture**: Add browser MediaRecorder API or Web Audio API to capture microphone input
3. **Real-time Transcription Display**: Stream transcription results back to frontend via WebSocket
4. **State Management**: Add React state management for recording status and transcription results
5. **Optional Database**: Connect to existing PostgreSQL database for storing transcription history

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

### Frontend not built
If you see "Frontend not built" error:
```bash
cd frontend
npm install
npm run build
```

### Port already in use
Change the port:
```bash
uv run python main.py web --port 8080
```

### CORS issues in development
Make sure to use `--dev` flag when running the backend in development:
```bash
uv run python main.py web --dev
```
