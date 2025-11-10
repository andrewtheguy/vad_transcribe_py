import { useCallback, useEffect, useRef, useState } from 'react'
import { Mic, Square, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import workletUrl from '../workers/audio-processor.ts?worker&url'

const TARGET_SAMPLE_RATE = 16_000
const TRANSCRIPT_FETCH_LIMIT = 1000
const TRANSCRIPT_POLL_INTERVAL_MS = 2000
const AUDIO_PROCESSOR_BUFFER_SIZE = 4096
const TARGET_CHUNK_SAMPLES = AUDIO_PROCESSOR_BUFFER_SIZE * 4
const LANGUAGE_STORAGE_KEY = 'whisper-transcribe-language'
const API_BASE =
  (
    (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
    (import.meta.env.VITE_API_BASE as string | undefined) ??
    ''
  ).replace(/\/$/, '')

type PendingChunk = {
  start: number
  payload: ArrayBuffer
}

type TranscriptRow = {
  id: number
  start_timestamp: string
  end_timestamp: string
  content: string
}

type SessionResponse = {
  session_id: string
}

const apiUrl = (path: string) => `${API_BASE}${path}`
const SUPPORTED_LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'es', label: 'Spanish' },
  { code: 'zh', label: 'Mandarin (zh)' },
  { code: 'yue', label: 'Cantonese (yue)' },
]
const DEFAULT_LANGUAGE = SUPPORTED_LANGUAGES[0]?.code ?? 'en'

const isSupportedLanguage = (code: string | null | undefined) =>
  Boolean(code && SUPPORTED_LANGUAGES.some((lang) => lang.code === code))

const getStoredLanguage = () => {
  if (typeof window === 'undefined') {
    return DEFAULT_LANGUAGE
  }
  const stored = window.localStorage.getItem(LANGUAGE_STORAGE_KEY)
  return isSupportedLanguage(stored) ? (stored as string) : DEFAULT_LANGUAGE
}

/**
 * Format a UTC ISO timestamp to local timezone string.
 * Backend sends timestamps in UTC (ISO 8601 format with +00:00 or Z suffix).
 * This function converts to the user's local timezone for display.
 */
const formatTimestamp = (iso: string) => {
  if (!iso) {
    return ''
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  // Display in local timezone with date and time in 24-hour format
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export default function RecordPage() {
  const [isRecording, setIsRecording] = useState(false)
  const [status, setStatus] = useState('Idle')
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [language, setLanguage] = useState<string>(() => getStoredLanguage())
  const [transcripts, setTranscripts] = useState<TranscriptRow[]>([])
  const [transcriptError, setTranscriptError] = useState<string | null>(null)
  const [transcriptionEnabled, setTranscriptionEnabled] = useState<boolean | null>(null)
  const [alternateApiUrl, setAlternateApiUrl] = useState<string | null>(null)
  const [recordingError, setRecordingError] = useState<string | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const chunkQueueRef = useRef<PendingChunk[]>([])
  const flushingRef = useRef(false)
  const destroyedRef = useRef(false)
  const stopRecordingRef = useRef<() => Promise<void>>(async () => {})
  const transcriptPollerRef = useRef<number | null>(null)
  const transcriptContainerRef = useRef<HTMLDivElement | null>(null)
  const userScrolledUpRef = useRef(false)

  useEffect(() => {
    if (!isSupportedLanguage(language)) {
      setLanguage(DEFAULT_LANGUAGE)
      return
    }
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language)
    }
  }, [language])

  // Check transcription config on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch(apiUrl('/api/transcribe/config'))
        if (response.ok) {
          const data = await response.json()
          setTranscriptionEnabled(data.transcription_enabled ?? true)
          setAlternateApiUrl(data.alternate_api_url ?? null)
        } else {
          // If endpoint doesn't exist, assume transcription is enabled (for backwards compatibility)
          setTranscriptionEnabled(true)
        }
      } catch (error) {
        console.error('Failed to fetch transcription config:', error)
        // Default to enabled on error
        setTranscriptionEnabled(true)
      }
    }
    void fetchConfig()
  }, [])

  const logEvent = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    console.info(`[Whisper Transcribe ${timestamp}] ${message}`)
  }, [])

  const createServerSession = useCallback(async (): Promise<string> => {
    const response = await fetch(apiUrl('/api/transcribe/stream/session'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        language,
        sample_rate: TARGET_SAMPLE_RATE,
      }),
    })
    if (!response.ok) {
      // Try to parse JSON error response first
      let errorMessage = `Server responded with ${response.status}`
      try {
        const errorData = await response.json()
        if (errorData.detail) {
          errorMessage = errorData.detail
        }
      } catch {
        // If JSON parsing fails, try text
        const textError = await response.text()
        if (textError) {
          errorMessage = textError
        }
      }
      throw new Error(errorMessage)
    }
    const data = (await response.json()) as SessionResponse
    if (!data.session_id) {
      throw new Error('Server did not return a session id')
    }
    logEvent(`Session ${data.session_id} prepared on server`)
    return data.session_id
  }, [language, logEvent])

  const cleanupAudioGraph = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.port.close()
      workletNodeRef.current.disconnect()
      workletNodeRef.current = null
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => undefined)
      audioContextRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
    }
  }, [])

  const finalizeSession = useCallback(async () => {
    const sessionId = sessionIdRef.current
    sessionIdRef.current = null
    setActiveSessionId(null)
    if (!sessionId) {
      return
    }
    try {
      await fetch(apiUrl(`/api/transcribe/stream/${sessionId}`), { method: 'DELETE' })
      logEvent(`Session ${sessionId} closed`)
    } catch (_err) {
      logEvent(`Session ${sessionId} closed locally (DELETE failed)`)
    }
  }, [logEvent])

  const fetchTranscripts = useCallback(async () => {
    const sessionId = sessionIdRef.current
    if (!sessionId) {
      return
    }
    try {
      const params = new URLSearchParams({
        limit: TRANSCRIPT_FETCH_LIMIT.toString(),
      })
      const response = await fetch(
        apiUrl(`/api/transcribe/stream/${sessionId}/transcripts?${params.toString()}`),
      )
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `Server responded with ${response.status}`)
      }
      const data = await response.json()
      setTranscripts(data.transcripts ?? [])
      setTranscriptError(null)
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Unknown error while fetching transcripts'
      setTranscriptError(message)
      console.error('[Whisper Transcribe] Transcript fetch failed:', message)
    }
  }, [])

  const clearTranscriptPolling = useCallback(() => {
    if (transcriptPollerRef.current !== null) {
      window.clearInterval(transcriptPollerRef.current)
      transcriptPollerRef.current = null
    }
  }, [])

  const startTranscriptPolling = useCallback(() => {
    clearTranscriptPolling()
    void fetchTranscripts()
    transcriptPollerRef.current = window.setInterval(() => {
      void fetchTranscripts()
    }, TRANSCRIPT_POLL_INTERVAL_MS)
  }, [clearTranscriptPolling, fetchTranscripts])

  const handleFatalError = useCallback(
    async (message: string) => {
      logEvent(message)
      cleanupAudioGraph()
      chunkQueueRef.current = []
      clearTranscriptPolling()
      setIsRecording(false)
      setStatus('Idle')
      setRecordingError(message)
      await finalizeSession()
    },
    [logEvent, cleanupAudioGraph, clearTranscriptPolling, finalizeSession],
  )

  const flushQueue = useCallback(async () => {
    if (flushingRef.current || destroyedRef.current) {
      return
    }
    flushingRef.current = true
    while (chunkQueueRef.current.length > 0) {
      const sessionId = sessionIdRef.current
      if (!sessionId) {
        chunkQueueRef.current = []
        break
      }
      const chunk = chunkQueueRef.current.shift()!
      const params = new URLSearchParams({
        session_id: sessionId,
        start: chunk.start.toString(),
        sample_rate: TARGET_SAMPLE_RATE.toString(),
        language,
      })
      try {
        const response = await fetch(apiUrl(`/api/transcribe/stream?${params.toString()}`), {
          method: 'POST',
          body: chunk.payload,
          headers: {
            'Content-Type': 'application/octet-stream',
          },
        })
        if (!response.ok) {
          const detail = await response.text()
          throw new Error(detail || `Server responded with ${response.status}`)
        }
        logEvent(
          `Chunk @ ${chunk.start.toFixed(2)}s • ${(chunk.payload.byteLength / 2).toLocaleString()} samples`,
        )
      } catch (error) {
        const message =
          error instanceof Error ? error.message : 'Unknown error while streaming audio chunk'
        await handleFatalError(`Streaming halted: ${message}`)
        break
      }
    }
    flushingRef.current = false
  }, [logEvent, handleFatalError, language])

  const enqueueChunk = useCallback(
    (chunk: PendingChunk) => {
      chunkQueueRef.current.push(chunk)
      void flushQueue()
    },
    [flushQueue],
  )

  const startRecording = useCallback(async () => {
    if (isRecording) {
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      logEvent('Media devices API is not available in this browser')
      return
    }
    setStatus('Preparing session…')
    setRecordingError(null) // Clear any previous errors
    let sessionId: string | null = null
    try {
      sessionId = await createServerSession()
      sessionIdRef.current = sessionId
      setActiveSessionId(sessionId)
      setTranscripts([])
      setTranscriptError(null)
      userScrolledUpRef.current = false
      startTranscriptPolling()
      setStatus('Requesting microphone…')

      // Request stereo audio if available
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: { ideal: 2 }, // Prefer stereo if available
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      const AudioContextCtor =
        window.AudioContext ||
        (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!AudioContextCtor) {
        throw new Error('Web Audio API is not supported in this browser')
      }
      const audioContext = new AudioContextCtor()

      // Check for AudioWorklet support
      if (!audioContext.audioWorklet) {
        throw new Error(
          'AudioWorklet is not supported in this browser. ' +
          'Please use Chrome 66+, Firefox 76+, Safari 14.1+, or Edge 79+'
        )
      }

      const source = audioContext.createMediaStreamSource(stream)

      // Load the AudioWorklet processor
      try {
        await audioContext.audioWorklet.addModule(workletUrl)
      } catch (error) {
        console.error('Failed to load audio worklet:', error)
        throw new Error('Failed to initialize audio processing')
      }

      // Create the AudioWorklet node
      const workletNode = new AudioWorkletNode(audioContext, 'audio-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 0,
        channelCount: 2, // Allow stereo input
        channelCountMode: 'max', // Accept max channels from source
        processorOptions: {
          targetSampleRate: TARGET_SAMPLE_RATE,
          targetChunkSamples: TARGET_CHUNK_SAMPLES,
        },
      })

      // Set up message handler for chunks from the worklet
      workletNode.port.onmessage = (event) => {
        const { type, chunk, start, message } = event.data

        if (type === 'chunk') {
          enqueueChunk({ start, payload: chunk })
        } else if (type === 'error') {
          console.error('[AudioWorklet] Error:', message)
        }
      }

      // Store references
      audioContextRef.current = audioContext
      sourceRef.current = source
      workletNodeRef.current = workletNode
      mediaStreamRef.current = stream
      chunkQueueRef.current = []

      // Connect audio graph (no need to connect to destination)
      source.connect(workletNode)

      setIsRecording(true)
      setStatus('Streaming audio to backend')
      if (sessionId) {
        logEvent(`Session ${sessionId} started`)
      }
    } catch (error) {
      await handleFatalError(
        error instanceof Error ? `Failed to start recording: ${error.message}` : 'Failed to start recording',
      )
    }
  }, [logEvent, enqueueChunk, handleFatalError, isRecording, startTranscriptPolling, createServerSession])

  const stopRecording = useCallback(async () => {
    if (!isRecording && !sessionIdRef.current) {
      return
    }
    setStatus('Stopping…')
    cleanupAudioGraph()
    chunkQueueRef.current = []
    clearTranscriptPolling()
    setIsRecording(false)
    await finalizeSession()
    setStatus('Idle')
  }, [cleanupAudioGraph, clearTranscriptPolling, finalizeSession, isRecording])

  useEffect(() => {
    stopRecordingRef.current = stopRecording
  }, [stopRecording])

  useEffect(() => {
    destroyedRef.current = false
    return () => {
      destroyedRef.current = true
      void stopRecordingRef.current()
      clearTranscriptPolling()
    }
  }, [clearTranscriptPolling])

  // Auto-scroll to bottom when new transcripts arrive (unless user scrolled up)
  useEffect(() => {
    const container = transcriptContainerRef.current
    if (!container || transcripts.length === 0) {
      return
    }

    // Only auto-scroll if user hasn't manually scrolled up
    if (!userScrolledUpRef.current) {
      container.scrollTop = container.scrollHeight
    }
  }, [transcripts])

  // Track user scroll behavior
  const handleScroll = useCallback(() => {
    const container = transcriptContainerRef.current
    if (!container) {
      return
    }

    // Check if user is near the bottom (within 50px)
    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50
    userScrolledUpRef.current = !isNearBottom
  }, [])

  // Show loading state while checking config
  if (transcriptionEnabled === null) {
    return (
      <Card>
        <CardContent className="py-12">
          <p className="text-center text-slate-500 dark:text-slate-400">
            Loading...
          </p>
        </CardContent>
      </Card>
    )
  }

  // Show disabled message if transcription is not enabled
  if (transcriptionEnabled === false) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 justify-center">
            <Mic className="w-6 h-6" />
            Microphone Recording
          </CardTitle>
          <CardDescription className="text-center">
            Transcription is currently disabled
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-slate-100 dark:bg-slate-800 mb-6">
              <Mic className="w-12 h-12 text-slate-400 dark:text-slate-600" />
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-4 max-w-md mx-auto">
              This server is running in view-only mode. Transcription functionality is disabled.
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-500 max-w-md mx-auto">
              You can view historical transcripts on the History page, but cannot create new recordings.
              {alternateApiUrl && (
                <>
                  <br />
                  <br />
                  Alternate API URL is configured: <code className="font-mono text-xs">{alternateApiUrl}</code>
                </>
              )}
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
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
          <div className="text-center py-8">
            <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-slate-100 dark:bg-slate-800 mb-6">
              {isRecording ? (
                <Square className="w-12 h-12 text-rose-500" />
              ) : (
                <Mic className="w-12 h-12 text-slate-600 dark:text-slate-400" />
              )}
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-8 max-w-md mx-auto">
              {isRecording
                ? 'Recording in progress. Audio is down-sampled to 16 kHz in the browser and streamed to the backend.'
                : 'Click the button below to start recording from your microphone. Audio will be resampled client-side.'}
            </p>
            <div className="flex flex-col items-center gap-4">
              <Button
                size="lg"
                className="px-8"
                variant={isRecording ? 'destructive' : 'default'}
                onClick={isRecording ? () => void stopRecording() : () => void startRecording()}
              >
                {isRecording ? (
                  <>
                    <Square className="w-5 h-5 mr-2" />
                    Stop Recording
                  </>
                ) : (
                  <>
                    <Mic className="w-5 h-5 mr-2" />
                    Start Recording
                  </>
                )}
              </Button>
              <div className="text-sm text-slate-500 dark:text-slate-400 text-center">
                Status: <span className="font-medium">{status}</span>
                {recordingError && (
                  <div className="mt-2 text-rose-600 dark:text-rose-400 font-medium max-w-md mx-auto">
                    {recordingError}
                  </div>
                )}
                <div className="mt-2 flex items-center gap-2 justify-center">
                  <label className="text-xs uppercase tracking-wide text-slate-400">
                    Language
                  </label>
                  <select
                    value={language}
                    onChange={(event) => setLanguage(event.target.value)}
                    disabled={isRecording}
                    className="rounded-md border border-slate-200 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-800"
                  >
                    {SUPPORTED_LANGUAGES.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.label}
                      </option>
                    ))}
                  </select>
                </div>
                {activeSessionId ? (
                  <div className="mt-1 break-all">
                    Session ID:{' '}
                    <span className="font-mono text-xs text-slate-600 dark:text-slate-300">
                      {activeSessionId}
                    </span>
                  </div>
                ) : null}
              </div>
            </div>
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
            Audio chunks stream to the backend; transcriptions currently print on the server console
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            ref={transcriptContainerRef}
            onScroll={handleScroll}
            className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 min-h-[220px] max-h-[500px] text-left overflow-y-auto"
          >
            {transcriptError ? (
              <p className="text-rose-500 dark:text-rose-400 text-center">
                Failed to load transcripts: {transcriptError}
              </p>
            ) : transcripts.length === 0 ? (
              <p className="text-slate-500 dark:text-slate-400 text-center">
                No transcripts yet. Start recording to stream audio.
              </p>
            ) : (
              <ul className="space-y-1.5 text-sm text-slate-800 dark:text-slate-100">
                {transcripts.map((row) => (
                  <li
                    key={row.id}
                    className="rounded border border-slate-200 bg-white/80 px-2.5 py-1.5 dark:border-slate-700 dark:bg-slate-800/80"
                  >
                    <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      {formatTimestamp(row.start_timestamp)}
                    </span>
                    <span className="ml-2 whitespace-pre-wrap text-slate-700 dark:text-slate-200">{row.content}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </CardContent>
      </Card>
    </>
  )
}
