import { useCallback, useEffect, useRef, useState } from 'react'
import { History, Loader2 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const API_BASE =
  (
    (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
    (import.meta.env.VITE_API_BASE as string | undefined) ??
    ''
  ).replace(/\/$/, '')

const apiUrl = (path: string) => `${API_BASE}${path}`

type Show = {
  name: string
  transcript_count: number
  latest_timestamp: string | null
  earliest_timestamp: string | null
}

type TranscriptRow = {
  id: number
  timestamp: string
  content: string
}

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

const formatDate = (iso: string | null) => {
  if (!iso) {
    return ''
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  return date.toLocaleDateString()
}

export default function HistoryPage() {
  const [shows, setShows] = useState<Show[]>([])
  const [selectedShow, setSelectedShow] = useState<string | null>(null)
  const [transcripts, setTranscripts] = useState<TranscriptRow[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)

  const transcriptContainerRef = useRef<HTMLDivElement | null>(null)
  const offsetRef = useRef(0)
  const shouldScrollToBottomRef = useRef(false)

  // Fetch shows on mount
  useEffect(() => {
    const fetchShows = async () => {
      setLoading(true)
      setError(null)
      try {
        const response = await fetch(apiUrl('/api/shows'))
        if (!response.ok) {
          throw new Error(`Failed to fetch shows: ${response.statusText}`)
        }
        const data = await response.json()
        setShows(data.shows ?? [])
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load shows'
        setError(message)
        console.error('[History] Failed to fetch shows:', err)
      } finally {
        setLoading(false)
      }
    }

    void fetchShows()
  }, [])

  // Fetch transcripts for selected show
  const fetchTranscripts = useCallback(
    async (showName: string, offset: number, isLoadingMore: boolean) => {
      if (isLoadingMore) {
        setLoadingMore(true)
      } else {
        setLoading(true)
      }
      setError(null)

      try {
        const params = new URLSearchParams({
          offset: offset.toString(),
          limit: '100',
        })
        const response = await fetch(
          apiUrl(`/api/shows/${encodeURIComponent(showName)}/transcripts?${params.toString()}`),
        )
        if (!response.ok) {
          throw new Error(`Failed to fetch transcripts: ${response.statusText}`)
        }
        const data = await response.json()

        setTotal(data.total ?? 0)

        // Backend returns DESC order (latest first), reverse to get ASC (oldest first)
        const newTranscripts = [...(data.transcripts ?? [])].reverse()

        if (isLoadingMore) {
          // When loading more (scrolling up), prepend older transcripts at the top
          const container = transcriptContainerRef.current
          const prevScrollHeight = container?.scrollHeight ?? 0

          setTranscripts((prev) => [...newTranscripts, ...prev])

          // Restore scroll position after prepending
          setTimeout(() => {
            if (container) {
              const newScrollHeight = container.scrollHeight
              container.scrollTop = newScrollHeight - prevScrollHeight + container.scrollTop
            }
          }, 0)
        } else {
          // Initial load - scroll to bottom after render
          setTranscripts(newTranscripts)
          shouldScrollToBottomRef.current = true
        }

        // Check if there are more transcripts to load
        const currentCount = offset + (data.transcripts?.length ?? 0)
        setHasMore(currentCount < data.total)
        offsetRef.current = currentCount
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load transcripts'
        setError(message)
        console.error('[History] Failed to fetch transcripts:', err)
      } finally {
        setLoading(false)
        setLoadingMore(false)
      }
    },
    [],
  )

  // Scroll to bottom after initial load
  useEffect(() => {
    if (shouldScrollToBottomRef.current && transcriptContainerRef.current) {
      transcriptContainerRef.current.scrollTop = transcriptContainerRef.current.scrollHeight
      shouldScrollToBottomRef.current = false
    }
  }, [transcripts])

  // Handle show selection
  const handleSelectShow = useCallback(
    (showName: string) => {
      setSelectedShow(showName)
      offsetRef.current = 0
      setHasMore(true)
      void fetchTranscripts(showName, 0, false)
    },
    [fetchTranscripts],
  )

  // Handle scroll for infinite loading
  const handleScroll = useCallback(() => {
    const container = transcriptContainerRef.current
    if (!container || !selectedShow || loadingMore || !hasMore) {
      return
    }

    // Check if user scrolled near the top (within 100px)
    if (container.scrollTop < 100) {
      void fetchTranscripts(selectedShow, offsetRef.current, true)
    }
  }, [selectedShow, loadingMore, hasMore, fetchTranscripts])

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Left sidebar - Show list */}
      <div className="col-span-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="w-5 h-5" />
              Shows
            </CardTitle>
            <CardDescription>
              Select a show to view its transcript history
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading && shows.length === 0 ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
              </div>
            ) : error && shows.length === 0 ? (
              <p className="text-rose-500 dark:text-rose-400 text-sm text-center py-4">
                {error}
              </p>
            ) : shows.length === 0 ? (
              <p className="text-slate-500 dark:text-slate-400 text-sm text-center py-4">
                No shows found. Start recording to create transcripts.
              </p>
            ) : (
              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {shows.map((show) => (
                  <button
                    key={show.name}
                    onClick={() => handleSelectShow(show.name)}
                    className={`w-full text-left p-3 rounded-lg border transition-colors ${
                      selectedShow === show.name
                        ? 'border-slate-900 bg-slate-100 dark:border-slate-100 dark:bg-slate-800'
                        : 'border-slate-200 bg-white hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:hover:bg-slate-800'
                    }`}
                  >
                    <div className="font-medium text-sm text-slate-900 dark:text-slate-100 mb-1">
                      {show.name}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">
                      {show.transcript_count.toLocaleString()} transcripts
                    </div>
                    {show.latest_timestamp && (
                      <div className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                        Latest: {formatDate(show.latest_timestamp)}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Right side - Transcript viewer */}
      <div className="col-span-8">
        <Card>
          <CardHeader>
            <CardTitle>
              {selectedShow ? `Transcripts: ${selectedShow}` : 'Transcripts'}
            </CardTitle>
            <CardDescription>
              {selectedShow
                ? `Showing ${transcripts.length.toLocaleString()} of ${total.toLocaleString()} transcripts (latest at bottom, scroll up to load older)`
                : 'Select a show from the list to view its transcripts'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!selectedShow ? (
              <div className="flex items-center justify-center py-16 text-slate-500 dark:text-slate-400">
                <div className="text-center">
                  <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>Select a show to view transcripts</p>
                </div>
              </div>
            ) : loading && transcripts.length === 0 ? (
              <div className="flex items-center justify-center py-16">
                <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
              </div>
            ) : error && transcripts.length === 0 ? (
              <p className="text-rose-500 dark:text-rose-400 text-center py-8">
                {error}
              </p>
            ) : (
              <div
                ref={transcriptContainerRef}
                onScroll={handleScroll}
                className="bg-slate-50 dark:bg-slate-900 rounded-lg p-4 min-h-[500px] max-h-[600px] text-left overflow-y-auto"
              >
                {loadingMore && (
                  <div className="flex items-center justify-center py-3 mb-2">
                    <Loader2 className="w-5 h-5 animate-spin text-slate-400 mr-2" />
                    <span className="text-sm text-slate-500">Loading more...</span>
                  </div>
                )}
                {transcripts.length === 0 ? (
                  <p className="text-slate-500 dark:text-slate-400 text-center py-8">
                    No transcripts found for this show.
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
                        <span className="ml-2 whitespace-pre-wrap text-slate-700 dark:text-slate-200">
                          {row.content}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
