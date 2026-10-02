from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import ModelVersion
from app.recommendation.popularity_cache import get_popularity


class RecommendationUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendRecommendationResult:
    strategy: str
    items: Sequence[dict[str, Any] | tuple[int, float]]
    reason: str | None = None


class RecommendationBackend(Protocol):
    def recommend(
        self,
        *,
        db: Session,
        model_version: ModelVersion,
        customer_key: int,
        k: int,
        exclude_item_ids: set[int],
    ) -> BackendRecommendationResult:
        ...


class ApplicationRecommendationBackend:
    """Thin adapter keeping TensorFlow loading outside API modules."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Any] = {}
        self._recommendations: OrderedDict[
            tuple[str, int, int, frozenset[int]], tuple[float, BackendRecommendationResult]
        ] = OrderedDict()
        self._cache_lock = Lock()

    def _load_dependencies(self, db: Session, model_version: ModelVersion) -> tuple[Any, Any, Any]:
        try:
            from app.models.artifact import load_artifact
            from app.recommendation.popularity import PopularityRecommender
            from app.recommendation.service import RecommendationService
        except (ImportError, ModuleNotFoundError) as exc:
            raise RecommendationUnavailable("Recommendation module is unavailable") from exc

        with self._cache_lock:
            if model_version.id not in self._artifacts:
                self._artifacts.clear()
                self._recommendations.clear()
            artifact = self._artifacts.get(model_version.id)
            try:
                if artifact is None:
                    artifact = load_artifact(model_version.artifact_path)
                    self._artifacts[model_version.id] = artifact
                popularity = get_popularity(db)
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                raise RecommendationUnavailable("Active model artifact is unavailable") from exc
        return RecommendationService, artifact, popularity

    def recommend(
        self,
        *,
        db: Session,
        model_version: ModelVersion,
        customer_key: int,
        k: int,
        exclude_item_ids: set[int],
    ) -> BackendRecommendationResult:
        settings = get_settings()
        key = (model_version.id, customer_key, k, frozenset(exclude_item_ids))
        now = time.monotonic()
        with self._cache_lock:
            cached = self._recommendations.get(key)
            if cached is not None and now - cached[0] <= settings.recommendation_cache_ttl_seconds:
                self._recommendations.move_to_end(key)
                return cached[1]
            if cached is not None:
                del self._recommendations[key]
        service_class, artifact, popularity = self._load_dependencies(db, model_version)
        service = service_class(
            artifact.model,
            artifact.mappings,
            artifact.features,
            popularity,
            lambda requested_customer_key: (
                exclude_item_ids if requested_customer_key == customer_key else set()
            ),
            candidate_article_keys=artifact.mappings.article_keys,
            candidate_limit=None,
        )
        result = service.recommend(
            customer_key,
            max(
                1,
                artifact.mappings.num_articles
                - sum(
                    artifact.mappings.article_index(article_key) is not None
                    for article_key in exclude_item_ids
                ),
            ),
            exclude_seen_items=bool(exclude_item_ids),
        )
        response = BackendRecommendationResult(
            strategy=result.strategy,
            items=[(item.article_key, item.score) for item in result.recommendations],
            reason=result.reason,
        )
        with self._cache_lock:
            self._recommendations[key] = (now, response)
            self._recommendations.move_to_end(key)
            while len(self._recommendations) > settings.recommendation_cache_max_entries:
                self._recommendations.popitem(last=False)
        return response


_backend: RecommendationBackend = ApplicationRecommendationBackend()


def get_recommendation_backend() -> RecommendationBackend:
    return _backend


def warm_active_model() -> None:
    """Load active artifact and popularity cache without delaying API startup."""
    if not isinstance(_backend, ApplicationRecommendationBackend):
        return
    from sqlalchemy import select

    from app.database.models import ModelVersion
    from app.database.session import get_session

    try:
        with get_session() as db:
            model = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)).limit(1))
            if model is not None:
                _backend._load_dependencies(db, model)
    except Exception:
        # Recommendation endpoint reports a precise error if warming fails.
        return
