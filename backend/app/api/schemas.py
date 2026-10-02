from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]


class DateRangeResponse(BaseModel):
    start: date | None
    end: date | None


class TopArticleResponse(BaseModel):
    article_id: str
    product_name: str
    purchases: int = Field(ge=0)


class DatasetStatsResponse(BaseModel):
    dataset_name: Literal["E-commerce Recommendation Lab"] = "E-commerce Recommendation Lab"
    locale: Literal["pt-BR"] = "pt-BR"
    currency: Literal["BRL"] = "BRL"
    semantics: Literal["purchase_history"] = "purchase_history"
    total_purchases: int = Field(ge=0)
    unique_customers: int = Field(ge=0)
    customers_with_purchases: int = Field(ge=0)
    unique_articles: int = Field(ge=0)
    purchased_articles: int = Field(ge=0)
    average_purchases_per_customer: float = Field(ge=0)
    median_purchases_per_customer: float = Field(ge=0)
    average_purchases_per_article: float = Field(ge=0)
    date_range: DateRangeResponse
    purchases_per_customer: dict[str, int]
    age_band_counts: dict[str, int]
    top_articles: list[TopArticleResponse]
    sampled: bool


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_profile: Literal["fast", "balanced", "extensive", "full"] = "balanced"
    embedding_dim: int = Field(default=32, ge=1, le=256)
    dense_1: int = Field(default=64, ge=1, le=1024)
    dense_2: int = Field(default=32, ge=1, le=1024)
    batch_size: int = Field(default=4096, ge=1, le=65_536)
    epochs: int = Field(default=3, ge=1, le=100)
    negative_ratio: int = Field(default=2, ge=1, le=20)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    shuffle_buffer: int = Field(default=100_000, ge=1, le=1_000_000)
    evaluation_negatives: int = Field(default=100, ge=10, le=10_000)
    evaluation_batch_size: int = Field(default=8192, ge=1, le=65_536)
    random_seed: int = Field(default=42, ge=0, le=4_294_967_295)


class TrainingProfileResponse(BaseModel):
    name: Literal["fast", "balanced", "extensive", "full"]
    label: str
    description: str
    target_purchases: int | None


class TrainingProfilesResponse(BaseModel):
    default: Literal["balanced"] = "balanced"
    profiles: list[TrainingProfileResponse]


class TrainingMetricResponse(BaseModel):
    epoch: int
    timestamp: datetime
    loss: float
    val_loss: float | None
    learning_rate: float | None = None
    epoch_duration_seconds: float | None = None
    examples_per_second: float | None = None


class NetworkConnectionResponse(BaseModel):
    source_layer: str
    source_index: int
    target_layer: str
    target_index: int
    weight: float


class NetworkSnapshotResponse(BaseModel):
    epoch: int
    timestamp: datetime
    probe_profile_id: str | None
    probe_customer_name: str | None
    probe_article_id: str | None
    profile_embedding: list[float]
    article_embedding: list[float]
    dense_64: list[float]
    dense_32: list[float]
    output_score: float
    connections: list[NetworkConnectionResponse]


class HardwareInfoResponse(BaseModel):
    device: str
    ram_gb: float | None = None
    tensorflow_version: str | None = None
    python_version: str | None = None


class EvaluationResponse(BaseModel):
    recall_at_10: float
    hit_rate_at_10: float
    ndcg_at_10: float
    candidate_strategy: str


class TrainingRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: str
    status: str
    config: dict[str, Any]
    current_epoch: int
    progress: float
    device: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime
    metrics: list[TrainingMetricResponse] = Field(default_factory=list)
    network_snapshot: NetworkSnapshotResponse | None = None
    hardware: HardwareInfoResponse | None = None
    elapsed_seconds: float | None = None
    model_version: str | None = None
    evaluation: EvaluationResponse | None = None


class TrainingRunListResponse(BaseModel):
    total: int
    items: list[TrainingRunResponse]


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: str
    training_run_id: str
    status: str
    is_active: bool
    config: dict[str, Any] | None
    metrics: dict[str, Any] | None
    created_at: datetime
    activated_at: datetime | None


class ArticleResponse(BaseModel):
    article_id: str
    product_name: str
    department_name: str
    category_name: str
    subcategory_name: str
    brand: str
    price_band: str
    detail_desc: str


class CustomerResponse(BaseModel):
    profile_id: str
    name: str
    age_band: str
    state: str
    region: str
    purchase_count: int = Field(ge=0)
    first_purchase_date: date | None
    last_purchase_date: date | None
    personalizable: bool


class CustomerHistoryItem(BaseModel):
    purchased_on: date
    price: float
    sales_channel_id: int
    article_id: str
    article: ArticleResponse


class CustomerHistoryResponse(BaseModel):
    profile_id: str
    total: int = Field(ge=0)
    limit: int
    offset: int
    items: list[CustomerHistoryItem]


class RecommendationEvidence(BaseModel):
    kind: Literal[
        "model_score",
        "purchase_history_department",
        "age_band_popularity",
        "global_popularity",
    ]
    description: str
    purchase_count: int | None = Field(default=None, ge=0)
    customer_count: int | None = Field(default=None, ge=0)
    related_article_ids: list[str] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    rank: PositiveInt
    article_id: str
    score: float = Field(ge=0, le=100)
    grade: Literal["baixo", "medio", "alto", "muito_alto"]
    article: ArticleResponse
    evidence: list[RecommendationEvidence] = Field(min_length=1)
    breakdown: dict[str, "RecommendationComponent"]

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        if not float("-inf") < value < float("inf"):
            raise ValueError("score must be finite")
        return value


class ModelMetadataResponse(BaseModel):
    version: str
    trained_at: datetime
    data_profile: str | None = None


class ScoreSemanticsResponse(BaseModel):
    name: Literal["hybrid_ranking_score", "normalized_purchase_popularity"]
    description: str
    minimum: Literal[0] = 0
    maximum: Literal[100] = 100


class RecommendationComponent(BaseModel):
    raw_score: float = Field(ge=0, le=1)
    weight: float = Field(ge=0, le=100)
    contribution: float = Field(ge=0, le=100)
    label: str
    explanation: str


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    profile_id: str | None = None
    strategy: Literal["model", "popularity"]
    fallback_reason: Literal[
        "no_purchase_history",
        "no_active_model",
        "profile_not_in_model",
        "model_unavailable",
    ] | None = None
    model: ModelMetadataResponse | None = None
    score_semantics: ScoreSemanticsResponse
    applied_weights: dict[str, float]
    exclude_seen_articles: bool
    inference_duration_ms: float = Field(ge=0)
    items: list[RecommendationItem]
