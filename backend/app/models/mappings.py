"""Compact mappings between H&M database keys and embedding indices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _integer_array(values: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = values
    else:
        array = np.fromiter(values, dtype=np.int64)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.dtype.kind not in "iu":
        try:
            converted = array.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer IDs") from exc
        if not np.all(array == converted):
            raise ValueError(f"{name} must contain integer IDs")
        array = converted
    return np.asarray(array, dtype=np.int64)


@dataclass(frozen=True)
class DenseMappings:
    """Sorted H&M keys; array offsets are stable dense indices."""

    customer_keys: np.ndarray
    article_keys: np.ndarray

    def __post_init__(self) -> None:
        customers = _integer_array(self.customer_keys, "customer_keys")
        articles = _integer_array(self.article_keys, "article_keys")
        if len(customers) and np.any(customers[1:] <= customers[:-1]):
            raise ValueError("customer_keys must be unique and sorted")
        if len(articles) and np.any(articles[1:] <= articles[:-1]):
            raise ValueError("article_keys must be unique and sorted")
        object.__setattr__(self, "customer_keys", customers)
        object.__setattr__(self, "article_keys", articles)

    @classmethod
    def build(
        cls,
        customer_keys: Iterable[int] | np.ndarray,
        article_keys: Iterable[int] | np.ndarray,
    ) -> "DenseMappings":
        customers = np.unique(_integer_array(customer_keys, "customer_keys"))
        articles = np.unique(_integer_array(article_keys, "article_keys"))
        if customers.size == 0 or articles.size == 0:
            raise ValueError("Mappings require at least one customer and one article")
        return cls(customers, articles)

    @property
    def num_customers(self) -> int:
        return int(self.customer_keys.size)

    @property
    def num_articles(self) -> int:
        return int(self.article_keys.size)

    @staticmethod
    def _indices(source: np.ndarray, values: Iterable[int] | np.ndarray) -> np.ndarray:
        requested = _integer_array(values, "ids")
        positions = np.searchsorted(source, requested)
        valid = positions < source.size
        if source.size:
            valid &= source[np.minimum(positions, source.size - 1)] == requested
        return np.where(valid, positions, -1).astype(np.int64, copy=False)

    def customer_indices(self, customer_keys: Iterable[int] | np.ndarray) -> np.ndarray:
        return self._indices(self.customer_keys, customer_keys)

    def article_indices(self, article_keys: Iterable[int] | np.ndarray) -> np.ndarray:
        return self._indices(self.article_keys, article_keys)

    def customer_index(self, customer_key: int) -> int | None:
        value = int(self.customer_indices(np.asarray([customer_key], dtype=np.int64))[0])
        return None if value < 0 else value

    def article_index(self, article_key: int) -> int | None:
        value = int(self.article_indices(np.asarray([article_key], dtype=np.int64))[0])
        return None if value < 0 else value

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "customer_keys.npy", self.customer_keys, allow_pickle=False)
        np.save(path / "article_keys.npy", self.article_keys, allow_pickle=False)

    @classmethod
    def load(cls, directory: str | Path, mmap_mode: str | None = "r") -> "DenseMappings":
        path = Path(directory)
        customers = np.load(path / "customer_keys.npy", allow_pickle=False, mmap_mode=mmap_mode)
        articles = np.load(path / "article_keys.npy", allow_pickle=False, mmap_mode=mmap_mode)
        return cls(customers, articles)
