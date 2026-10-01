import { afterEach, describe, expect, it, vi } from 'vitest'
import { API_URL, ApiError, api, subscribeToTraining } from './api'

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

describe('H&M API service', () => {
  afterEach(() => {
    api.invalidateDatasetStats()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('uses H&M customer and article contracts', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ profile_id: 'abc', strategy: 'model', fallback_reason: null, model: { version: '1' }, exclude_seen_articles: true, inference_duration_ms: 3, items: [] }))
    await api.getProfileRecommendations('a/b')
    expect(fetchMock).toHaveBeenCalledWith(`${API_URL}/customers/a%2Fb/recommendations?k=10&exclude_seen_articles=true`, expect.objectContaining({ signal: undefined }))
  })

  it('requests paginated purchase history', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ profile_id: 'abc', total: 0, limit: 8, offset: 16, items: [] }))
    await api.getProfilePurchases('abc', 8, 16)
    expect(fetchMock).toHaveBeenCalledWith(`${API_URL}/customers/abc/history?limit=8&offset=16`, expect.any(Object))
  })

  it('surfaces API detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ detail: 'Model unavailable' }, 503))
    await expect(api.getProfileRecommendations('abc')).rejects.toEqual(new ApiError('Model unavailable', 503))
  })

  it('preserves AbortError for stale request cancellation', async () => {
    const aborted = new DOMException('The operation was aborted.', 'AbortError')
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(aborted)
    await expect(api.getProfile('abc', new AbortController().signal)).rejects.toBe(aborted)
  })

  it('caches dataset stats and supports forced refresh', async () => {
    const payload = { dataset_name: 'H&M', total_purchases: 1, unique_customers: 1, unique_articles: 1, purchased_articles: 1, average_purchases_per_customer: 1, median_purchases_per_customer: 1, average_purchases_per_article: 1, date_range: { start: null, end: null }, sampled: false, purchases_per_customer: { '1': 1 }, top_articles: [], age_band_counts: { '25-34': 1 } }
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => json(payload))
    await api.getDatasetStats({ force: true })
    await api.getDatasetStats()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await api.getDatasetStats({ force: true })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('subscribes to named training events and closes terminal streams', () => {
    class FakeEventSource {
      static instance: FakeEventSource
      listeners: Record<string, Array<(event: Event) => void>> = {}
      close = vi.fn()
      onerror: (() => void) | null = null
      constructor(public url: string) { FakeEventSource.instance = this }
      addEventListener(type: string, listener: EventListenerOrEventListenerObject) {
        const callback = typeof listener === 'function' ? listener : listener.handleEvent.bind(listener)
        this.listeners[type] = [...(this.listeners[type] || []), callback]
      }
      emit(type: string, data: unknown) {
        const event = new MessageEvent(type, { data: JSON.stringify(data) })
        this.listeners[type]?.forEach((listener) => listener(event))
      }
    }
    vi.stubGlobal('EventSource', FakeEventSource)
    const events: unknown[] = []
    subscribeToTraining('run/1', (event) => events.push(event), vi.fn())
    FakeEventSource.instance.emit('completed', { type: 'completed', run: { id: 'run/1' } })
    expect(FakeEventSource.instance.url).toBe(`${API_URL}/training/runs/run%2F1/events`)
    expect(Object.keys(FakeEventSource.instance.listeners)).toEqual(['status_changed', 'epoch_completed', 'network_snapshot', 'completed', 'failed'])
    expect(events).toHaveLength(1)
    expect(FakeEventSource.instance.close).toHaveBeenCalledOnce()
  })
})
