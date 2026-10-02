"""Memory-bounded catalog scoring and cold-start behavior."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from app.models.artifact import load_artifact
from app.models.features import HybridFeatures
from app.models.mappings import DenseMappings

from .popularity import PopularityRecommender


SeenItemsSource = Mapping[int, Iterable[int]] | Callable[[int], Iterable[int]]


@dataclass(frozen=True)
class Recommendation:
    article_key: int
    score: float


@dataclass(frozen=True)
class RecommendationResult:
    customer_key: int
    strategy: str
    recommendations: list[Recommendation]
    inference_duration_seconds: float
    cold_start: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_key": self.customer_key,
            "strategy": self.strategy,
            "cold_start": self.cold_start,
            "reason": self.reason,
            "inference_duration_seconds": self.inference_duration_seconds,
            "recommendations": [
                {"article_key": item.article_key, "score": item.score}
                for item in self.recommendations
            ],
        }


class RecommendationService:
    def __init__(
        self,
        model: Any,
        mappings: DenseMappings,
        features: HybridFeatures,
        popularity: PopularityRecommender,
        seen_items: SeenItemsSource,
        *,
        batch_size: int = 8192,
        candidate_article_keys: Iterable[int] | np.ndarray | None = None,
        candidate_limit: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.mappings = mappings
        self.features = features
        self.popularity = popularity
        self.seen_items = seen_items
        self.batch_size = int(batch_size)
        self.candidate_article_keys = (
            None
            if candidate_article_keys is None
            else np.asarray(candidate_article_keys, dtype=np.int64).reshape(-1)
        )
        self.candidate_limit = candidate_limit

    @classmethod
    def from_artifact(
        cls,
        artifact_directory: str | Path,
        popularity: PopularityRecommender,
        seen_items: SeenItemsSource,
        *,
        batch_size: int = 8192,
    ) -> "RecommendationService":
        artifact = load_artifact(artifact_directory)
        return cls(
            artifact.model,
            artifact.mappings,
            artifact.features,
            popularity,
            seen_items,
            batch_size=batch_size,
        )

    def _seen(self, customer_key: int) -> set[int]:
        values = (
            self.seen_items(customer_key)
            if callable(self.seen_items)
            else self.seen_items.get(customer_key, ())
        )
        return {int(item) for item in values}

    def _predict(self, customer_index: int, article_indices: np.ndarray) -> np.ndarray:
        inputs = self.features.model_inputs(
            np.full(article_indices.size, customer_index, dtype=np.int32),
            article_indices,
        )
        if hasattr(self.model, "predict"):
            values = self.model.predict(inputs, batch_size=self.batch_size, verbose=0)
        else:
            values = self.model(inputs, training=False)
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if scores.size != article_indices.size or np.any(~np.isfinite(scores)):
            raise ValueError("Model returned invalid recommendation scores")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Model recommendation scores are outside [0, 1]")
        return scores

    def recommend(
        self,
        customer_key: int,
        k: int = 10,
        *,
        exclude_seen_items: bool = True,
    ) -> RecommendationResult:
        if not 1 <= k <= self.mappings.num_articles:
            raise ValueError("k must be between 1 and catalog size")
        started = time.perf_counter()
        source_customer_key = int(customer_key)
        seen = self._seen(source_customer_key) if exclude_seen_items else set()
        customer_index = self.mappings.customer_index(source_customer_key)
        if customer_index is None:
            values = self.popularity.recommend(k, exclude_item_ids=seen)
            recommendations = [Recommendation(item, score) for item, score in values]
            return RecommendationResult(
                source_customer_key,
                "popularity",
                recommendations,
                time.perf_counter() - started,
                True,
                "customer_not_in_artifact",
            )

        if self.candidate_article_keys is None:
            candidates = np.arange(self.mappings.num_articles, dtype=np.int64)
        else:
            candidates = self.mappings.article_indices(self.candidate_article_keys)
            candidates = candidates[candidates >= 0]
            if self.candidate_limit is not None:
                candidates = candidates[: self.candidate_limit]
            if candidates.size < k:
                candidates = np.arange(self.mappings.num_articles, dtype=np.int64)

        best: list[tuple[float, int, int]] = []
        for start in range(0, candidates.size, self.batch_size):
            item_indices = candidates[start : start + self.batch_size]
            original_ids = np.asarray(self.mappings.article_keys[item_indices])
            if seen:
                keep = ~np.isin(original_ids, np.fromiter(seen, dtype=np.int64))
                item_indices = item_indices[keep]
                original_ids = original_ids[keep]
            if item_indices.size == 0:
                continue
            scores = self._predict(customer_index, item_indices)
            for item, score in zip(original_ids, scores, strict=True):
                item_value, score_value = int(item), float(score)
                entry = (score_value, -item_value, item_value)
                if len(best) < k:
                    heapq.heappush(best, entry)
                elif entry > best[0]:
                    heapq.heapreplace(best, entry)
        ordered = sorted(best, reverse=True)
        recommendations = [Recommendation(item, score) for score, _, item in ordered]
        if not recommendations:
            values = self.popularity.recommend(k, exclude_item_ids=seen)
            return RecommendationResult(
                source_customer_key,
                "popularity",
                [Recommendation(item, score) for item, score in values],
                time.perf_counter() - started,
                False,
                "no_eligible_hybrid_candidates",
            )
        return RecommendationResult(
            source_customer_key,
            "neural",
            recommendations,
            time.perf_counter() - started,
            False,
            "personalized_ecommerce_hybrid",
        )


_artifact_cache: dict[str, Any] = {}


def recommend_items(
    *,
    db: Any,
    model_version: Any,
    customer_key: int,
    k: int,
    exclude_item_ids: set[int],
) -> list[tuple[int, float]]:
    """Adapter expected by API layer; active model metadata comes from database."""
    del db
    artifact_path = str(model_version.artifact_path)
    artifact = _artifact_cache.get(artifact_path)
    if artifact is None:
        artifact = load_artifact(artifact_path)
        _artifact_cache.clear()
        _artifact_cache[artifact_path] = artifact
    if artifact.mappings.customer_index(customer_key) is None:
        raise KeyError(customer_key)
    fallback = PopularityRecommender(
        np.asarray([artifact.mappings.article_keys[0]], dtype=np.int64),
        np.asarray([1.0]),
    )
    service = RecommendationService(
        artifact.model,
        artifact.mappings,
        artifact.features,
        fallback,
        {customer_key: exclude_item_ids},
    )
    result = service.recommend(customer_key, k, exclude_seen_items=True)
    return [(item.article_key, item.score) for item in result.recommendations]
