import type {
  ArticleMetadata,
  DatasetStats,
  Profile,
  PurchaseHistory,
  Recommendation,
  RecommendationResult,
  TrainingConfig,
  TrainingEvent,
  TrainingProfilesResponse,
  TrainingRun,
} from '../types/api'

interface RawDatasetStats {
  dataset_name: string
  total_purchases: number
  unique_customers: number
  unique_articles: number
  purchased_articles: number
  average_purchases_per_customer: number
  median_purchases_per_customer: number
  average_purchases_per_article: number
  date_range: { start: string | null; end: string | null }
  sampled: boolean
  purchases_per_customer: Record<string, number>
  top_articles: Array<{ article_id: string; purchases: number }>
  age_band_counts: Record<string, number>
}

interface RawProfile {
  profile_id: string
  age_band: string | null
  purchase_count: number
  first_purchase_date: string | null
  last_purchase_date: string | null
}

interface RawPurchase {
  article_id: string
  purchased_on: string
  price: number
  sales_channel_id: number
  article: ArticleMetadata
}

interface RawPurchaseHistory {
  profile_id: string
  total: number
  limit: number
  offset: number
  items: RawPurchase[]
}

interface RawRecommendation {
  rank: number
  article_id: string
  score: number
  article: ArticleMetadata
  evidence: Array<{
    kind: string
    description: string
    purchase_count?: number
    customer_count?: number
    related_article_ids?: string[]
  }>
}

interface RawRecommendationResult {
  profile_id: string | null
  strategy: 'model' | 'popularity'
  items: RawRecommendation[]
  inference_duration_ms: number | null
  model: { version: string } | null
  fallback_reason: string | null
}

interface RawTrainingProfilesResponse {
  default: TrainingConfig['data_profile']
  profiles: Array<{
    name: TrainingConfig['data_profile']
    label: string
    target_purchases: number | null
    description: string
  }>
}

function mapDatasetStats(stats: RawDatasetStats): DatasetStats {
  const missing = stats.age_band_counts.unknown ?? 0
  const known = Math.max(0, stats.unique_customers - missing)
  return {
    dataset_name: stats.dataset_name,
    purchase_count: stats.total_purchases,
    profile_count: stats.unique_customers,
    article_count: stats.unique_articles,
    average_purchases_per_profile: stats.average_purchases_per_customer,
    median_purchases_per_profile: stats.median_purchases_per_customer,
    average_purchases_per_article: stats.average_purchases_per_article,
    date_range: stats.date_range,
    sampled: stats.sampled,
    purchases_per_profile: Object.entries(stats.purchases_per_customer).map(([bucket, profiles]) => ({ bucket, profiles })),
    top_articles: stats.top_articles,
    age_coverage: {
      known,
      missing,
      known_ratio: known / Math.max(1, stats.unique_customers),
      groups: Object.fromEntries(Object.entries(stats.age_band_counts).filter(([name]) => name !== 'unknown')),
    },
  }
}

function mapProfile(profile: RawProfile): Profile {
  return {
    profile_id: profile.profile_id,
    age_band: profile.age_band,
    purchase_count: profile.purchase_count,
    first_purchase_at: profile.first_purchase_date,
    last_purchase_at: profile.last_purchase_date,
  }
}

function mapPurchaseHistory(history: RawPurchaseHistory): PurchaseHistory {
  return {
    profile_id: history.profile_id,
    total: history.total,
    limit: history.limit,
    offset: history.offset,
    purchases: history.items.map((purchase) => ({
      article_id: purchase.article_id,
      purchased_at: purchase.purchased_on,
      source_price: purchase.price,
      sales_channel_id: purchase.sales_channel_id,
      article: purchase.article,
    })),
  }
}

function mapRecommendation(item: RawRecommendation): Recommendation {
  return {
    rank: item.rank,
    article_id: item.article_id,
    relevance_score: item.score,
    article: item.article,
    evidence: item.evidence.map((fact) => ({
      reason: fact.description,
      cohort_buyers: fact.kind === 'age_band_popularity' ? fact.customer_count ?? 0 : 0,
      related_history_articles: fact.kind === 'purchase_history_department' ? fact.related_article_ids?.length ?? 0 : 0,
      same_department: fact.kind === 'purchase_history_department',
    })),
  }
}

function mapRecommendationResult(result: RawRecommendationResult): RecommendationResult {
  return {
    profile_id: result.profile_id ?? '',
    strategy: result.strategy === 'model' ? 'hybrid' : 'popular',
    fallback_reason: result.fallback_reason,
    model_version: result.model?.version ?? null,
    exclude_purchased_articles: true,
    inference_duration_ms: result.inference_duration_ms ?? 0,
    recommendations: result.items.map(mapRecommendation),
  }
}

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError('Não foi possível conectar à API. Verifique se o serviço está disponível.', 0)
  }

  if (!response.ok) {
    let message = `A API respondeu com status ${response.status}.`
    try {
      const body = (await response.json()) as { detail?: string; message?: string }
      message = body.detail || body.message || message
    } catch {
      // Keep HTTP fallback when response has no JSON body.
    }
    throw new ApiError(message, response.status)
  }

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

interface RequestOptions {
  signal?: AbortSignal
  force?: boolean
}

type CacheEntry<T> = { value?: T }
const datasetCache: CacheEntry<DatasetStats> = {}

function cached<T>(entry: CacheEntry<T>, load: () => Promise<T>, force = false) {
  if (force) {
    delete entry.value
  }
  if (entry.value !== undefined) return Promise.resolve(entry.value)
  return load().then((value) => {
    entry.value = value
    return value
  })
}

interface TrainingRunList {
  total: number
  items: TrainingRun[]
}

export const api = {
  getDatasetStats: ({ signal, force }: RequestOptions = {}) => cached(datasetCache, () => request<RawDatasetStats>('/dataset/stats', { signal }).then(mapDatasetStats), force),
  invalidateDatasetStats: () => { delete datasetCache.value },
  getTrainingProfiles: (signal?: AbortSignal) => request<RawTrainingProfilesResponse>('/training/profiles', { signal }).then((response): TrainingProfilesResponse => ({
    default_profile: response.default,
    profiles: response.profiles.map((profile) => ({ ...profile, id: profile.name })),
  })),
  createTrainingRun: (config: TrainingConfig, signal?: AbortSignal) => request<TrainingRun>('/training/runs', { method: 'POST', body: JSON.stringify(config), signal }),
  getTrainingRuns: (limit = 20, signal?: AbortSignal) => request<TrainingRunList>(`/training/runs?limit=${limit}`, { signal }),
  getTrainingRun: (id: string, signal?: AbortSignal) => request<TrainingRun>(`/training/runs/${encodeURIComponent(id)}`, { signal }),
  getProfile: (profileId: string, signal?: AbortSignal) => request<RawProfile>(`/customers/${encodeURIComponent(profileId)}`, { signal }).then(mapProfile),
  getProfilePurchases: (profileId: string, limit = 8, offset = 0, signal?: AbortSignal) => request<RawPurchaseHistory>(`/customers/${encodeURIComponent(profileId)}/history?limit=${limit}&offset=${offset}`, { signal }).then(mapPurchaseHistory),
  getRandomProfile: (signal?: AbortSignal) => request<RawProfile>('/customers/random?personalizable=true&model_eligible=true', { signal }).then(mapProfile),
  getPopularRecommendations: (k = 10, signal?: AbortSignal) => request<RawRecommendationResult>(`/recommendations/popular?k=${k}`, { signal }).then(mapRecommendationResult),
  getProfileRecommendations: (profileId: string, k = 10, signal?: AbortSignal) => request<RawRecommendationResult>(`/customers/${encodeURIComponent(profileId)}/recommendations?k=${k}&exclude_seen_articles=true`, { signal }).then(mapRecommendationResult),
}

export function subscribeToTraining(
  id: string,
  onEvent: (event: TrainingEvent) => void,
  onConnectionError: () => void,
): () => void {
  const source = new EventSource(`${API_URL}/training/runs/${encodeURIComponent(id)}/events`)
  const receive = (message: MessageEvent<string>) => {
    try {
      const event = JSON.parse(message.data) as TrainingEvent
      onEvent(event)
      if (event.type === 'completed' || event.type === 'failed') source.close()
    } catch {
      onConnectionError()
    }
  }
  const namedEvents = ['status_changed', 'epoch_completed', 'network_snapshot', 'completed', 'failed']
  namedEvents.forEach((type) => source.addEventListener(type, receive as EventListener))
  source.onerror = onConnectionError
  return () => source.close()
}
