"""Dense categorical customer and product features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .mappings import DenseMappings


CUSTOMER_FEATURES = (
    "age_band",
)
ARTICLE_FEATURES = (
    "department_name",
    "category_name",
    "subcategory_name",
    "brand",
    "price_band",
)
FEATURE_NAMES = CUSTOMER_FEATURES + ARTICLE_FEATURES


def _value(row: Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def _encoded(
    rows: Iterable[Any], key_name: str, feature_names: tuple[str, ...]
) -> tuple[dict[int, dict[str, str]], dict[str, list[str]]]:
    values: dict[int, dict[str, str]] = {}
    categories = {name: set() for name in feature_names}
    for row in rows:
        key = int(_value(row, key_name))
        encoded_row: dict[str, str] = {}
        for name in feature_names:
            raw = _value(row, name)
            value = "" if raw is None else str(raw).strip()
            encoded_row[name] = value
            if value:
                categories[name].add(value)
        values[key] = encoded_row
    return values, {name: sorted(items) for name, items in categories.items()}


@dataclass(frozen=True)
class HybridFeatures:
    """Feature indices aligned with ``DenseMappings`` arrays; zero means unknown."""

    customer_values: dict[str, np.ndarray]
    article_values: dict[str, np.ndarray]
    vocabularies: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if set(self.customer_values) != set(CUSTOMER_FEATURES):
            raise ValueError("Customer feature set does not match hybrid schema")
        if set(self.article_values) != set(ARTICLE_FEATURES):
            raise ValueError("Article feature set does not match hybrid schema")
        if set(self.vocabularies) != set(FEATURE_NAMES):
            raise ValueError("Feature vocabularies do not match hybrid schema")
        customer_values: dict[str, np.ndarray] = {}
        article_values: dict[str, np.ndarray] = {}
        for name, values in {**self.customer_values, **self.article_values}.items():
            array = np.asarray(values, dtype=np.int32).reshape(-1)
            if np.any(array < 0) or np.any(array > len(self.vocabularies[name])):
                raise ValueError(f"Feature indices outside {name} vocabulary")
            if name in CUSTOMER_FEATURES:
                customer_values[name] = array
            else:
                article_values[name] = array
        object.__setattr__(self, "customer_values", customer_values)
        object.__setattr__(self, "article_values", article_values)

    def validate_mappings(self, mappings: DenseMappings) -> None:
        if any(values.size != mappings.num_customers for values in self.customer_values.values()):
            raise ValueError("Customer features do not align with customer mappings")
        if any(values.size != mappings.num_articles for values in self.article_values.values()):
            raise ValueError("Article features do not align with article mappings")

    @classmethod
    def build(
        cls,
        mappings: DenseMappings,
        customers: Iterable[Any],
        articles: Iterable[Any],
    ) -> "HybridFeatures":
        customer_rows, customer_vocabulary = _encoded(
            customers, "customer_key", CUSTOMER_FEATURES
        )
        article_rows, article_vocabulary = _encoded(
            articles, "article_key", ARTICLE_FEATURES
        )
        vocabularies = {**customer_vocabulary, **article_vocabulary}
        lookups = {
            name: {value: index + 1 for index, value in enumerate(values)}
            for name, values in vocabularies.items()
        }
        customer_values = {
            name: np.asarray(
                [lookups[name].get(customer_rows.get(int(key), {}).get(name, ""), 0) for key in mappings.customer_keys],
                dtype=np.int32,
            )
            for name in CUSTOMER_FEATURES
        }
        article_values = {
            name: np.asarray(
                [lookups[name].get(article_rows.get(int(key), {}).get(name, ""), 0) for key in mappings.article_keys],
                dtype=np.int32,
            )
            for name in ARTICLE_FEATURES
        }
        result = cls(
            customer_values,
            article_values,
            {name: tuple(values) for name, values in vocabularies.items()},
        )
        result.validate_mappings(mappings)
        return result

    @property
    def cardinalities(self) -> dict[str, int]:
        return {name: len(values) + 1 for name, values in self.vocabularies.items()}

    def model_inputs(
        self, customer_indices: np.ndarray, article_indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        customers = np.asarray(customer_indices, dtype=np.int32).reshape(-1)
        articles = np.asarray(article_indices, dtype=np.int32).reshape(-1)
        if customers.size != articles.size:
            raise ValueError("Customer and article feature indices must have equal length")
        return {
            "customer_index": customers,
            "article_index": articles,
            **{name: values[customers] for name, values in self.customer_values.items()},
            **{name: values[articles] for name, values in self.article_values.items()},
        }

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        np.savez_compressed(
            path / "features.npz", **self.customer_values, **self.article_values
        )
        (path / "feature_vocabulary.json").write_text(
            json.dumps(self.vocabularies, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "HybridFeatures":
        path = Path(directory)
        vocabularies = json.loads((path / "feature_vocabulary.json").read_text(encoding="utf-8"))
        with np.load(path / "features.npz", allow_pickle=False) as values:
            customers = {name: np.asarray(values[name]) for name in CUSTOMER_FEATURES}
            articles = {name: np.asarray(values[name]) for name in ARTICLE_FEATURES}
        return cls(
            customers,
            articles,
            {name: tuple(items) for name, items in vocabularies.items()},
        )
