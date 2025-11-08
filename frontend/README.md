# Whisper Transcribe Frontend

Web interface for the Whisper Transcribe application built with Vite, React, TypeScript, Tailwind CSS, and shadcn/ui.

## Development

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

The frontend will be available at http://localhost:5173

Make sure the backend API is running:
```bash
uv run python main.py web --dev
```

### Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── ui/           # shadcn/ui components
│   ├── lib/              # Utilities
│   ├── App.tsx           # Main application
│   ├── main.tsx          # Entry point
│   └── index.css         # Global styles
├── index.html            # HTML template
├── vite.config.ts        # Vite configuration
├── tailwind.config.js    # Tailwind configuration
└── tsconfig.json         # TypeScript configuration
```

## Technologies

- **Vite** - Build tool and dev server
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first CSS
- **shadcn/ui** - Component library
- **Lucide React** - Icon library
