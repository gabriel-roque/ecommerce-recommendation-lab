"""Reproducible sampled-candidate ranking evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from app.models.features import HybridFeatures

from .sampling import InteractionIndex


def recall_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def hit_rate_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    return float(bool(relevant.intersection(ranked[:k])))


def ndcg_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(rank + 2.0)
        for rank, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(rank + 2.0) for rank in range(min(k, len(relevant))))
    return float(dcg / ideal) if ideal else 0.0


@dataclass(frozen=True)
class SampledRankingResult:
    recall_at_k: float
    hit_rate_at_k: float
    ndcg_at_k: float
    k: int
    evaluated_positives: int
    skipped_positives: int
    negative_candidates: int
    candidate_strategy: str = "one positive plus reproducibly sampled unseen negatives"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _predict(
    model: Any,
    customers: np.ndarray,
    articles: np.ndarray,
    features: HybridFeatures,
    batch_size: int,
) -> np.ndarray:
    inputs = features.model_inputs(customers, articles)
    if hasattr(model, "predict"):
        values = model.predict(inputs, batch_size=batch_size, verbose=0)
    else:
        values = model(inputs, training=False)
    scores = np.asarray(values).reshape(-1)
    if scores.size != articles.size or np.any(~np.isfinite(scores)):
        raise ValueError("Model returned invalid ranking predictions")
    return scores


def evaluate_sampled_ranking(
    model: Any,
    user_indices: Iterable[int] | np.ndarray,
    positive_item_indices: Iterable[int] | np.ndarray,
    interactions: InteractionIndex,
    features: HybridFeatures,
    *,
    k: int = 10,
    negative_candidates: int = 100,
    batch_size: int = 8192,
    seed: int = 42,
) -> SampledRankingResult:
    if k <= 0 or batch_size <= 0:
        raise ValueError("k and batch_size must be positive")
    if negative_candidates < k:
        raise ValueError("negative_candidates must be at least k")
    users = np.asarray(
        user_indices if isinstance(user_indices, np.ndarray) else list(user_indices),
        dtype=np.int64,
    ).reshape(-1)
    positives = np.asarray(
        positive_item_indices
        if isinstance(positive_item_indices, np.ndarray)
        else list(positive_item_indices),
        dtype=np.int64,
    ).reshape(-1)
    if users.size != positives.size:
        raise ValueError("user_indices and positive_item_indices must have equal length")

    rng = np.random.default_rng(seed)
    recalls: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    skipped = 0
    pending_users: list[np.ndarray] = []
    pending_candidates: list[np.ndarray] = []
    pending_positives: list[int] = []
    pending_count = 0

    def score_pending() -> None:
        nonlocal pending_count
        if not pending_candidates:
            return
        flat_users = np.concatenate(pending_users)
        flat_candidates = np.concatenate(pending_candidates)
        flat_scores = _predict(model, flat_users, flat_candidates, features, batch_size)
        offset = 0
        for candidates, positive_value in zip(
            pending_candidates, pending_positives, strict=True
        ):
            scores = flat_scores[offset : offset + candidates.size]
            offset += candidates.size
            ranked_order = np.lexsort((candidates, -scores))
            ranked = candidates[ranked_order].tolist()
            relevant = {positive_value}
            recalls.append(recall_at_k(relevant, ranked, k))
            hits.append(hit_rate_at_k(relevant, ranked, k))
            ndcgs.append(ndcg_at_k(relevant, ranked, k))
        pending_users.clear()
        pending_candidates.clear()
        pending_positives.clear()
        pending_count = 0

    for user, positive in zip(users, positives, strict=True):
        user_value, positive_value = int(user), int(positive)
        if not 0 <= positive_value < interactions.num_items:
            skipped += 1
            continue
        seen = interactions.seen(user_value)
        positive_is_seen = interactions.contains(user_value, positive_value)
        available_count = interactions.num_items - len(seen) - int(not positive_is_seen)
        count = min(negative_candidates, available_count)
        sampled: set[int] = set()
        attempts = 0
        while len(sampled) < count and attempts < max(64, count * 8):
            candidate = int(rng.integers(interactions.num_items))
            if candidate != positive_value and not interactions.contains(user_value, candidate):
                sampled.add(candidate)
            attempts += 1
        if len(sampled) < count:
            available = np.fromiter(
                (
                    item for item in range(interactions.num_items)
                    if item != positive_value and item not in seen and item not in sampled
                ),
                dtype=np.int64,
            )
            needed = count - len(sampled)
            if needed:
                sampled.update(
                    int(item)
                    for item in rng.choice(available, size=needed, replace=False)
                )
        negatives = np.fromiter(sorted(sampled), dtype=np.int64, count=count)
        candidates = np.concatenate((np.asarray([positive_value]), negatives))
        pending_users.append(np.full(candidates.size, user_value, dtype=np.int64))
        pending_candidates.append(candidates)
        pending_positives.append(positive_value)
        pending_count += candidates.size
        if pending_count >= batch_size:
            score_pending()
    score_pending()
    if not recalls:
        raise ValueError("No valid positive events were available for ranking evaluation")
    return SampledRankingResult(
        recall_at_k=float(np.mean(recalls)),
        hit_rate_at_k=float(np.mean(hits)),
        ndcg_at_k=float(np.mean(ndcgs)),
        k=k,
        evaluated_positives=len(recalls),
        skipped_positives=skipped,
        negative_candidates=negative_candidates,
    )
