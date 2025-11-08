import { Link, Outlet, useLocation } from 'react-router-dom'
import { Mic, History } from 'lucide-react'

export default function AppLayout() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <header className="mb-8">
          <div className="text-center mb-6">
            <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
              Whisper Transcribe
            </h1>
            <p className="text-slate-600 dark:text-slate-400">
              AI-powered speech transcription with voice activity detection
            </p>
          </div>

          {/* Navigation */}
          <nav className="flex items-center justify-center gap-2 mt-6">
            <Link
              to="/record"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                location.pathname === '/record' || location.pathname === '/'
                  ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                  : 'bg-white text-slate-700 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
              }`}
            >
              <Mic className="w-4 h-4" />
              Record
            </Link>
            <Link
              to="/history"
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                location.pathname === '/history'
                  ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                  : 'bg-white text-slate-700 hover:bg-slate-100 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
              }`}
            >
              <History className="w-4 h-4" />
              History
            </Link>
          </nav>
        </header>

        {/* Page content */}
        <main>
          <Outlet />
        </main>
      </div>
    </div>
  )
}
