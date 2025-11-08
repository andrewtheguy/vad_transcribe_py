import { Mic, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-2">
            Whisper Transcribe
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            AI-powered speech transcription with voice activity detection
          </p>
        </header>

        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 justify-center">
              <Mic className="w-6 h-6" />
              Microphone Recording
            </CardTitle>
            <CardDescription className="text-center">
              Record audio from your microphone for real-time transcription
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-12">
              <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-slate-100 dark:bg-slate-800 mb-6">
                <Mic className="w-12 h-12 text-slate-600 dark:text-slate-400" />
              </div>
              <p className="text-slate-600 dark:text-slate-400 mb-8 max-w-md mx-auto">
                Click the button below to start recording from your microphone.
                Transcriptions will appear in real-time as you speak.
              </p>
              <Button disabled size="lg" className="px-8">
                <Mic className="w-5 h-5 mr-2" />
                Start Recording (Coming Soon)
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Transcription Results
            </CardTitle>
            <CardDescription>
              Your transcriptions will appear here in real-time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-8 min-h-[200px] text-center flex items-center justify-center">
              <p className="text-slate-500 dark:text-slate-400">
                No transcriptions yet. Start recording to see results.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
