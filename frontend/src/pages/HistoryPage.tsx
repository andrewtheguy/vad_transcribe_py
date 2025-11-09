import { useCallback, useEffect, useRef, useState } from 'react'
import { History, Loader2 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const FOLLOW_POLL_INTERVAL_MS = 2000
const FOLLOW_MAX_TRANSCRIPTS = 1000

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
  // Display in local timezone with date and time
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/**
 * Format a UTC ISO timestamp to local timezone date string.
 */
const formatDate = (iso: string | null) => {
  if (!iso) {
    return ''
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  // Display in local timezone with date only
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
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
  const [isFollowing, setIsFollowing] = useState(false)

  const transcriptContainerRef = useRef<HTMLDivElement | null>(null)
  const offsetRef = useRef(0)
  const shouldScrollToBottomRef = useRef(false)
  const followPollerRef = useRef<number | null>(null)
  const latestIdRef = useRef<number | null>(null)

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

          setTranscripts((prev) => {
            // Deduplicate by ID to avoid duplicate keys
            const existingIds = new Set(prev.map(t => t.id))
            const uniqueNew = newTranscripts.filter(t => !existingIds.has(t.id))
            return [...uniqueNew, ...prev]
          })

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

  // Fetch latest transcripts for following mode
  const fetchLatestTranscripts = useCallback(
    async (showName: string) => {
      try {
        const params = new URLSearchParams({
          offset: '0',
          limit: FOLLOW_MAX_TRANSCRIPTS.toString(),
        })
        const response = await fetch(
          apiUrl(`/api/shows/${encodeURIComponent(showName)}/transcripts?${params.toString()}`),
        )
        if (!response.ok) {
          throw new Error(`Failed to fetch transcripts: ${response.statusText}`)
        }
        const data = await response.json()

        // Backend returns DESC order (latest first), reverse to get ASC (oldest first)
        const newTranscripts = [...(data.transcripts ?? [])].reverse()

        // Track latest ID
        if (newTranscripts.length > 0) {
          latestIdRef.current = newTranscripts[newTranscripts.length - 1].id
        }

        setTranscripts(newTranscripts)
        setTotal(data.total ?? 0)
        shouldScrollToBottomRef.current = true
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load transcripts'
        setError(message)
        console.error('[History] Failed to fetch latest transcripts:', err)
      }
    },
    [],
  )

  // Clear follow polling
  const clearFollowPolling = useCallback(() => {
    if (followPollerRef.current !== null) {
      window.clearInterval(followPollerRef.current)
      followPollerRef.current = null
    }
  }, [])

  // Start follow polling
  const startFollowPolling = useCallback(() => {
    if (!selectedShow) return

    clearFollowPolling()
    void fetchLatestTranscripts(selectedShow)
    followPollerRef.current = window.setInterval(() => {
      if (selectedShow) {
        void fetchLatestTranscripts(selectedShow)
      }
    }, FOLLOW_POLL_INTERVAL_MS)
  }, [selectedShow, fetchLatestTranscripts, clearFollowPolling])

  // Scroll to bottom after initial load or when following
  useEffect(() => {
    if (shouldScrollToBottomRef.current && transcriptContainerRef.current) {
      transcriptContainerRef.current.scrollTop = transcriptContainerRef.current.scrollHeight
      shouldScrollToBottomRef.current = false
    }
  }, [transcripts])

  // Handle following toggle
  const handleToggleFollow = useCallback(
    (checked: boolean) => {
      setIsFollowing(checked)

      if (checked) {
        // Switching to following mode
        if (selectedShow) {
          shouldScrollToBottomRef.current = true
          startFollowPolling()
        }
      } else {
        // Switching to non-following mode
        clearFollowPolling()
      }
    },
    [selectedShow, startFollowPolling, clearFollowPolling],
  )

  // Handle show selection
  const handleSelectShow = useCallback(
    (showName: string) => {
      setSelectedShow(showName)
      setIsFollowing(false)
      clearFollowPolling()
      offsetRef.current = 0
      setHasMore(true)
      latestIdRef.current = null
      void fetchTranscripts(showName, 0, false)
    },
    [fetchTranscripts, clearFollowPolling],
  )

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearFollowPolling()
    }
  }, [clearFollowPolling])

  // Handle scroll for infinite loading (only when not following)
  const handleScroll = useCallback(() => {
    const container = transcriptContainerRef.current
    if (!container || !selectedShow || loadingMore || !hasMore || isFollowing) {
      return
    }

    // Check if user scrolled near the top (within 100px)
    if (container.scrollTop < 100) {
      void fetchTranscripts(selectedShow, offsetRef.current, true)
    }
  }, [selectedShow, loadingMore, hasMore, isFollowing, fetchTranscripts])

  return (
    <div className="space-y-6">
      {/* Mobile dropdown - only visible on mobile */}
      <div className="md:hidden">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="w-5 h-5" />
              Select Show
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading && shows.length === 0 ? (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
              </div>
            ) : error && shows.length === 0 ? (
              <p className="text-rose-500 dark:text-rose-400 text-sm text-center py-2">
                {error}
              </p>
            ) : shows.length === 0 ? (
              <p className="text-slate-500 dark:text-slate-400 text-sm text-center py-2">
                No shows found. Start recording to create transcripts.
              </p>
            ) : (
              <select
                value={selectedShow ?? ''}
                onChange={(e) => e.target.value && handleSelectShow(e.target.value)}
                className="w-full p-3 rounded-lg border border-slate-300 bg-white text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 focus:ring-2 focus:ring-slate-500 focus:border-transparent"
              >
                <option value="">Choose a show...</option>
                {shows.map((show) => (
                  <option key={show.name} value={show.name}>
                    {show.name} ({show.transcript_count.toLocaleString()} transcripts)
                  </option>
                ))}
              </select>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Left sidebar - Show list (hidden on mobile) */}
        <div className="hidden md:block md:col-span-4">
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
        <div className="col-span-12 md:col-span-8">
        <Card>
          <CardHeader>
            <div className="flex items-start justify-between">
              <div>
                <CardTitle>
                  {selectedShow ? `Transcripts: ${selectedShow}` : 'Transcripts'}
                </CardTitle>
                <CardDescription>
                  {selectedShow
                    ? isFollowing
                      ? `Following latest transcripts (max ${FOLLOW_MAX_TRANSCRIPTS.toLocaleString()}, auto-updates every ${FOLLOW_POLL_INTERVAL_MS / 1000}s)`
                      : `Showing ${transcripts.length.toLocaleString()} of ${total.toLocaleString()} transcripts (latest at bottom, scroll up to load older)`
                    : 'Select a show from the list to view its transcripts'}
                </CardDescription>
              </div>
              {selectedShow && (
                <div className="flex items-center gap-2">
                  <label
                    htmlFor="follow-checkbox"
                    className="text-sm font-medium text-slate-700 dark:text-slate-300 cursor-pointer"
                  >
                    Follow
                  </label>
                  <input
                    id="follow-checkbox"
                    type="checkbox"
                    checked={isFollowing}
                    onChange={(e) => handleToggleFollow(e.target.checked)}
                    className="w-4 h-4 rounded border-slate-300 text-slate-900 focus:ring-slate-500 dark:border-slate-600 dark:bg-slate-700 cursor-pointer"
                  />
                </div>
              )}
            </div>
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
    </div>
  )
}
