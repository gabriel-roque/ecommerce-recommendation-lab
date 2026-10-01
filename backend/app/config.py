from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_name: str = "H&M Recommendation Lab"
    database_url: str = "postgresql+psycopg://postgres:postgres@postgres:5432/recommendations"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    auto_create_tables: bool = True
    sse_poll_interval_seconds: float = Field(default=0.5, gt=0, le=30)
    require_full_dataset: bool = True
    dataset_manifest_path: str = str(_PROJECT_ROOT / "data/processed/dataset_manifest.json")
    dataset_stats_path: str = str(_PROJECT_ROOT / "data/processed/dataset_stats.json")
    training_samples_dir: str = str(_PROJECT_ROOT / "data/processed/samples")
    popularity_path: str = str(_PROJECT_ROOT / "data/processed/popularity.json")
    artifacts_dir: str = "artifacts"
    recommendation_cache_ttl_seconds: float = Field(default=30, gt=0, le=3600)
    recommendation_cache_max_entries: int = Field(default=512, ge=1, le=100_000)
    recommendation_candidate_limit: int = Field(default=20_000, ge=100, le=1_000_000)
    preload_active_model: bool = True

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://", "sqlite://")):
            raise ValueError("DATABASE_URL must use PostgreSQL or SQLite")
        return value


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    origins_value = os.getenv("CORS_ORIGINS")
    if origins_value:
        try:
            parsed_origins = json.loads(origins_value)
            origins = parsed_origins if isinstance(parsed_origins, list) else origins_value.split(",")
        except json.JSONDecodeError:
            origins = origins_value.split(",")
        origins = [origin.strip() for origin in origins if origin.strip()]
    else:
        origins = ["http://localhost:5173"]

    return Settings(
        app_name=os.getenv("APP_NAME", "H&M Recommendation Lab"),
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@postgres:5432/recommendations",
        ),
        cors_origins=origins,
        auto_create_tables=_as_bool(os.getenv("AUTO_CREATE_TABLES"), True),
        sse_poll_interval_seconds=float(os.getenv("SSE_POLL_INTERVAL_SECONDS", "0.5")),
        require_full_dataset=_as_bool(os.getenv("REQUIRE_FULL_DATASET"), True),
        dataset_manifest_path=os.getenv(
            "DATASET_MANIFEST_PATH",
            str(_PROJECT_ROOT / "data/processed/dataset_manifest.json"),
        ),
        dataset_stats_path=os.getenv(
            "DATASET_STATS_PATH",
            str(_PROJECT_ROOT / "data/processed/dataset_stats.json"),
        ),
        training_samples_dir=os.getenv(
            "TRAINING_SAMPLES_DIR",
            str(_PROJECT_ROOT / "data/processed/samples"),
        ),
        popularity_path=os.getenv(
            "POPULARITY_PATH",
            str(_PROJECT_ROOT / "data/processed/popularity.json"),
        ),
        artifacts_dir=os.getenv("ARTIFACTS_DIR", "artifacts"),
        recommendation_cache_ttl_seconds=float(
            os.getenv("RECOMMENDATION_CACHE_TTL_SECONDS", "30")
        ),
        recommendation_cache_max_entries=int(
            os.getenv("RECOMMENDATION_CACHE_MAX_ENTRIES", "512")
        ),
        recommendation_candidate_limit=int(
            os.getenv("RECOMMENDATION_CANDIDATE_LIMIT", "20000")
        ),
        preload_active_model=_as_bool(os.getenv("PRELOAD_ACTIVE_MODEL"), True),
    )


# Training modules read settings lazily through this shared immutable instance.
settings = get_settings()
