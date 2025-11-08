import { useCallback, useEffect, useRef, useState } from 'react'
import { Mic, Square, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

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
  timestamp: string
  content: string
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

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const formatTimestamp = (iso: string) => {
  if (!iso) {
    return ''
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  return date.toLocaleString()
}

const downsampleBuffer = (
  input: Float32Array,
  inputSampleRate: number,
  targetSampleRate: number,
): Float32Array => {
  if (targetSampleRate === inputSampleRate) {
    return input
  }

  if (targetSampleRate > inputSampleRate) {
    // Handle the rare case where we must up-sample.
    const ratio = targetSampleRate / inputSampleRate
    const newLength = Math.floor(input.length * ratio)
    const result = new Float32Array(newLength)
    for (let i = 0; i < newLength; i += 1) {
      const index = i / ratio
      const low = Math.floor(index)
      const high = Math.min(Math.ceil(index), input.length - 1)
      const frac = index - low
      result[i] = input[low] * (1 - frac) + input[high] * frac
    }
    return result
  }

  const ratio = inputSampleRate / targetSampleRate
  const newLength = Math.floor(input.length / ratio)
  const result = new Float32Array(newLength)
  let offsetResult = 0
  let offsetBuffer = 0

  while (offsetResult < newLength) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i += 1) {
      accum += input[i]
      count += 1
    }
    result[offsetResult] = count > 0 ? accum / count : 0
    offsetResult += 1
    offsetBuffer = nextOffsetBuffer
  }

  return result
}

const floatTo16BitPCM = (input: Float32Array): ArrayBuffer => {
  const buffer = new ArrayBuffer(input.length * 2)
  const view = new DataView(buffer)
  for (let i = 0; i < input.length; i += 1) {
    const sample = clamp(input[i], -1, 1)
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
  }
  return buffer
}

export default function RecordPage() {
  const [isRecording, setIsRecording] = useState(false)
  const [status, setStatus] = useState('Idle')
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [language, setLanguage] = useState<string>(() => getStoredLanguage())
  const [transcripts, setTranscripts] = useState<TranscriptRow[]>([])
  const [transcriptError, setTranscriptError] = useState<string | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const sentSamplesRef = useRef(0)
  const chunkQueueRef = useRef<PendingChunk[]>([])
  const pendingSamplesRef = useRef<Float32Array | null>(null)
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

  const logEvent = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    console.info(`[Whisper Transcribe ${timestamp}] ${message}`)
  }, [])

  const cleanupAudioGraph = useCallback(() => {
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current.onaudioprocess = null
      processorRef.current = null
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
      sentSamplesRef.current = 0
      clearTranscriptPolling()
      setIsRecording(false)
      setStatus('Idle')
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

  const transmitChunkSamples = useCallback(
    (samples: Float32Array) => {
      if (!samples.length) {
        return
      }
      const chunkStart = sentSamplesRef.current / TARGET_SAMPLE_RATE
      sentSamplesRef.current += samples.length
      const payload = floatTo16BitPCM(samples)
      enqueueChunk({ start: chunkStart, payload })
    },
    [enqueueChunk],
  )

  const flushPendingSamples = useCallback(
    (force = false) => {
      let pending = pendingSamplesRef.current
      if (!pending || pending.length === 0) {
        pendingSamplesRef.current = pending
        return
      }
      while (pending && pending.length >= TARGET_CHUNK_SAMPLES) {
        const chunk = pending.slice(0, TARGET_CHUNK_SAMPLES)
        transmitChunkSamples(chunk)
        pending = pending.length > TARGET_CHUNK_SAMPLES ? pending.slice(TARGET_CHUNK_SAMPLES) : null
      }
      if (force && pending && pending.length) {
        transmitChunkSamples(pending)
        pending = null
      }
      pendingSamplesRef.current = pending
    },
    [transmitChunkSamples],
  )

  const queueSamplesForChunking = useCallback(
    (samples: Float32Array) => {
      if (!samples.length) {
        return
      }
      if (!pendingSamplesRef.current || pendingSamplesRef.current.length === 0) {
        pendingSamplesRef.current = samples
      } else {
        const merged = new Float32Array(pendingSamplesRef.current.length + samples.length)
        merged.set(pendingSamplesRef.current)
        merged.set(samples, pendingSamplesRef.current.length)
        pendingSamplesRef.current = merged
      }
      flushPendingSamples()
    },
    [flushPendingSamples],
  )

  const startRecording = useCallback(async () => {
    if (isRecording) {
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      logEvent('Media devices API is not available in this browser')
      return
    }
    setStatus('Requesting microphone…')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const AudioContextCtor =
        window.AudioContext ||
        (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!AudioContextCtor) {
        throw new Error('Web Audio API is not supported in this browser')
      }
      const audioContext = new AudioContextCtor()
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(AUDIO_PROCESSOR_BUFFER_SIZE, 1, 1)

      audioContextRef.current = audioContext
      sourceRef.current = source
      processorRef.current = processor
      mediaStreamRef.current = stream
      sentSamplesRef.current = 0
      chunkQueueRef.current = []
      pendingSamplesRef.current = null

      const sessionId = crypto.randomUUID()
      sessionIdRef.current = sessionId
      setActiveSessionId(sessionId)

      processor.onaudioprocess = (event) => {
        const raw = event.inputBuffer.getChannelData(0)
        const copy = new Float32Array(raw.length)
        copy.set(raw)
        const downsampled = downsampleBuffer(copy, audioContext.sampleRate, TARGET_SAMPLE_RATE)
        queueSamplesForChunking(downsampled)
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      setIsRecording(true)
      setStatus('Streaming audio to backend')
      logEvent(`Session ${sessionId} started`)
      setTranscripts([])
      setTranscriptError(null)
      userScrolledUpRef.current = false
      startTranscriptPolling()
    } catch (error) {
      await handleFatalError(
        error instanceof Error ? `Failed to start recording: ${error.message}` : 'Failed to start recording',
      )
    }
  }, [logEvent, queueSamplesForChunking, handleFatalError, isRecording, language, startTranscriptPolling])

  const stopRecording = useCallback(async () => {
    if (!isRecording && !sessionIdRef.current) {
      return
    }
    setStatus('Stopping…')
    flushPendingSamples(true)
    cleanupAudioGraph()
    chunkQueueRef.current = []
    sentSamplesRef.current = 0
    clearTranscriptPolling()
    setIsRecording(false)
    await finalizeSession()
    setStatus('Idle')
  }, [cleanupAudioGraph, clearTranscriptPolling, finalizeSession, flushPendingSamples, isRecording])

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
                      {formatTimestamp(row.timestamp)}
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
