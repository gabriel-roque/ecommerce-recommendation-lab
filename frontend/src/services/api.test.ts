import { afterEach, describe, expect, it, vi } from 'vitest'
import { API_URL, ApiError, api, subscribeToTraining } from './api'

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
const weights = { model: 45, category: 30, popularity: 15, trend: 10 }
const recommendationResponse = { profile_id: 'abc', strategy: 'model', fallback_reason: null, model: { version: '1' }, score_semantics: { name: 'hybrid_ranking_score', description: 'Ranking relativo.', minimum: 0, maximum: 100 }, applied_weights: weights, exclude_seen_articles: true, inference_duration_ms: 3, items: [] }

describe('serviço da API', () => {
  afterEach(() => {
    api.invalidateDatasetStats()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('sends four recommendation weights and top-10 contract', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(recommendationResponse))
    await api.getCustomerRecommendations('a/b', 10, weights)
    expect(fetchMock).toHaveBeenCalledWith(`${API_URL}/customers/a%2Fb/recommendations?k=10&exclude_seen_articles=true&model_weight=45&category_weight=30&popularity_weight=15&trend_weight=10`, expect.objectContaining({ signal: undefined }))
  })

  it('maps customer identity and product purchase history', async () => {
    const customerPayload = { profile_id: 'abc', name: 'Lucas', state: 'SP', region: 'Sudeste', age_band: '25-34', purchase_count: 1, first_purchase_date: '2026-01-01', last_purchase_date: '2026-01-02', personalizable: true }
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json(customerPayload))
    await expect(api.getCustomer('abc')).resolves.toMatchObject({ name: 'Lucas', state: 'SP', region: 'Sudeste', last_purchase_at: '2026-01-02' })
  })

  it('requests paginated purchase history', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ profile_id: 'abc', total: 0, limit: 8, offset: 16, items: [] }))
    await api.getCustomerPurchases('abc', 8, 16)
    expect(fetchMock).toHaveBeenCalledWith(`${API_URL}/customers/abc/history?limit=8&offset=16`, expect.any(Object))
  })

  it('requests a random customer covered by active model', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ profile_id: 'abc', name: 'Lucas', state: 'SP', region: 'Sudeste', age_band: '25-34', purchase_count: 1, personalizable: true }))
    await api.getRandomCustomer()
    expect(fetchMock).toHaveBeenCalledWith(`${API_URL}/customers/random?personalizable=true&model_eligible=true`, expect.any(Object))
  })

  it('surfaces API detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(json({ detail: 'Modelo indisponível' }, 503))
    await expect(api.getCustomerRecommendations('abc', 10, weights)).rejects.toEqual(new ApiError('Modelo indisponível', 503))
  })

  it('preserves AbortError for stale request cancellation', async () => {
    const aborted = new DOMException('The operation was aborted.', 'AbortError')
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(aborted)
    await expect(api.getCustomer('abc', new AbortController().signal)).rejects.toBe(aborted)
  })

  it('caches dataset stats and supports forced refresh', async () => {
    const payload = { dataset_name: 'E-commerce Recommendation Lab', currency: 'BRL', total_purchases: 1, unique_customers: 1, unique_articles: 1, purchased_articles: 1, average_purchases_per_customer: 1, median_purchases_per_customer: 1, average_purchases_per_article: 1, date_range: { start: null, end: null }, sampled: false, purchases_per_customer: { '1': 1 }, top_articles: [], age_band_counts: { '25-34': 1 } }
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
