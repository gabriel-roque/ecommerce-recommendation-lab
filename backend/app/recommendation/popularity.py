"""Purchase-count popularity baseline and cold-start fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _column(purchases: Any, name: str) -> Iterable[Any]:
    try:
        return purchases[name]
    except (KeyError, TypeError, IndexError):
        return [purchase[name] for purchase in purchases]


@dataclass(frozen=True)
class PopularityRecommender:
    article_keys: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        items = np.asarray(self.article_keys, dtype=np.int64).reshape(-1)
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        if items.size != scores.size or np.any(~np.isfinite(scores)):
            raise ValueError("Popularity item IDs and scores must be finite and aligned")
        if items.size != np.unique(items).size:
            raise ValueError("Popularity item IDs must be unique")
        object.__setattr__(self, "article_keys", items)
        object.__setattr__(self, "scores", scores)

    @classmethod
    def fit(
        cls,
        purchases: Any,
    ) -> "PopularityRecommender":
        totals: dict[int, float] = {}
        for article in _column(purchases, "article_key"):
            totals[int(article)] = totals.get(int(article), 0.0) + 1.0
        if not totals:
            raise ValueError("Cannot fit popularity model with no purchases")
        ordered = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
        item_ids = np.fromiter((item for item, _ in ordered), dtype=np.int64)
        raw_scores = np.fromiter((score for _, score in ordered), dtype=np.float64)
        scores = raw_scores / raw_scores[0]
        return cls(item_ids, scores)

    def recommend(
        self, k: int, *, exclude_item_ids: Iterable[int] = ()
    ) -> list[tuple[int, float]]:
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        excluded = {int(item) for item in exclude_item_ids}
        result: list[tuple[int, float]] = []
        for item, score in zip(self.article_keys, self.scores, strict=True):
            if int(item) not in excluded:
                result.append((int(item), float(score)))
                if len(result) == k:
                    break
        return result

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, article_keys=self.article_keys, scores=self.scores)

    @classmethod
    def load(cls, path: str | Path) -> "PopularityRecommender":
        with np.load(path, allow_pickle=False) as data:
            return cls(data["article_keys"], data["scores"])
