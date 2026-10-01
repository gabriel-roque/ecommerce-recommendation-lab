export type TrainingStatus =
  | 'QUEUED'
  | 'PREPARING_DATA'
  | 'TRAINING'
  | 'EVALUATING'
  | 'SAVING_MODEL'
  | 'COMPLETED'
  | 'FAILED'

export interface AgeCoverage {
  known: number
  missing: number
  known_ratio: number
  groups: Record<string, number>
}

export interface DatasetStats {
  dataset_name: string
  purchase_count: number
  profile_count: number
  article_count: number
  average_purchases_per_profile: number
  median_purchases_per_profile: number
  average_purchases_per_article: number
  date_range: { start: string | null; end: string | null }
  sampled: boolean
  purchases_per_profile: Array<{ bucket: string; profiles: number }>
  top_articles: Array<{ article_id: string; purchases: number }>
  age_coverage: AgeCoverage
}

export interface TrainingConfig {
  data_profile: 'fast' | 'balanced' | 'extensive' | 'full'
  embedding_dim: number
  dense_1: number
  dense_2: number
  batch_size: number
  epochs: number
  negative_ratio: number
  learning_rate: number
}

export interface TrainingProfile {
  id: TrainingConfig['data_profile']
  label: string
  target_purchases: number | null
  description: string
}

export interface TrainingProfilesResponse {
  profiles: TrainingProfile[]
  default_profile: TrainingConfig['data_profile']
}

export interface TrainingMetric {
  epoch: number
  timestamp?: string
  loss: number
  val_loss: number | null
  learning_rate?: number
  epoch_duration_seconds?: number
  examples_per_second?: number
}

export interface NetworkConnection {
  source_layer: 'profile_embedding' | 'article_embedding' | 'dense_64' | 'dense_32'
  source_index: number
  target_layer: 'dense_64' | 'dense_32' | 'output'
  target_index: number
  weight: number
}

export interface NetworkSnapshot {
  epoch: number
  timestamp: string
  probe_profile_id: string | null
  probe_article_id: string | null
  profile_embedding: number[]
  article_embedding: number[]
  dense_64: number[]
  dense_32: number[]
  output_score: number
  connections: NetworkConnection[]
}

export interface HardwareInfo {
  device: string
  ram_gb?: number
  tensorflow_version?: string
  python_version?: string
}

export interface TrainingRun {
  id: string
  status: TrainingStatus
  config: TrainingConfig
  current_epoch: number
  progress: number
  device: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  updated_at: string
  metrics: TrainingMetric[]
  network_snapshot?: NetworkSnapshot | null
  hardware?: HardwareInfo | null
  elapsed_seconds?: number | null
  error_message?: string | null
  model_version?: string | null
  evaluation?: {
    recall_at_10: number
    hit_rate_at_10: number
    ndcg_at_10: number
    candidate_strategy: string
  } | null
}

export type TrainingEvent =
  | { type: 'status_changed'; status: TrainingStatus; timestamp: string; current_epoch: number; progress: number; elapsed_seconds?: number | null; hardware?: HardwareInfo | null; error_message?: string | null }
  | ({ type: 'epoch_completed'; snapshot?: NetworkSnapshot; elapsed_seconds?: number | null } & TrainingMetric)
  | { type: 'network_snapshot'; snapshot: NetworkSnapshot }
  | { type: 'completed'; run: TrainingRun }
  | { type: 'failed'; error_message: string }

export interface Profile {
  profile_id: string
  age_band: string | null
  purchase_count: number
  first_purchase_at: string | null
  last_purchase_at: string | null
}

export interface ArticleMetadata {
  article_id: string
  product_name: string | null
  product_type_name: string | null
  product_group_name: string | null
  colour_group_name: string | null
  department_name: string | null
  section_name: string | null
  garment_group_name: string | null
  detail_desc: string | null
}

export interface Purchase {
  purchased_at: string
  article_id: string
  source_price: number
  sales_channel_id: number | null
  article: ArticleMetadata
}

export interface PurchaseHistory {
  profile_id: string
  total: number
  limit: number
  offset: number
  purchases: Purchase[]
}

export interface RecommendationEvidence {
  reason: string
  cohort_buyers: number
  related_history_articles: number
  same_department: boolean
}

export interface Recommendation {
  rank: number
  article_id: string
  relevance_score: number
  article: ArticleMetadata
  evidence: RecommendationEvidence[]
}

export interface RecommendationResult {
  profile_id: string
  strategy: 'popular' | 'hybrid'
  fallback_reason: string | null
  model_version: string | null
  exclude_purchased_articles: boolean
  inference_duration_ms: number
  recommendations: Recommendation[]
}
