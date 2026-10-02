"""On-the-fly negative sampling without false negatives."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from app.models.ncf import require_tensorflow
from app.models.features import HybridFeatures

from .config import DATA_PROFILES, DataProfile


SAMPLE_CACHE_VERSION = 3
SAMPLE_STRATEGY = "deterministic_customer_purchase_history_v2"


@dataclass(frozen=True)
class SampledPurchases:
    values: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.values["timestamp"])

    def __getitem__(self, column: str) -> np.ndarray:
        return self.values[column]

    def take(self, indices: np.ndarray) -> "SampledPurchases":
        return SampledPurchases({name: values[indices] for name, values in self.values.items()})


def source_fingerprint(source_manifest: Mapping[str, Any]) -> str:
    encoded = json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _purchase_manifest(
    purchases: Any,
    *,
    profile: DataProfile,
    seed: int,
    source_manifest: Mapping[str, Any],
    sampled: bool,
) -> dict[str, Any]:
    timestamps = np.asarray(purchases["timestamp"], dtype=np.int64)
    customers = np.asarray(purchases["customer_key"], dtype=np.int64)
    articles = np.asarray(purchases["article_key"], dtype=np.int64)

    def iso(value: int | None) -> str | None:
        return (
            datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
            if value is not None
            else None
        )

    minimum = int(timestamps.min()) if timestamps.size else None
    maximum = int(timestamps.max()) if timestamps.size else None
    return {
        "cache_version": SAMPLE_CACHE_VERSION,
        "profile": profile,
        "dataset": "ecommerce_sintetico_br",
        "semantics": "purchase_only",
        "target_purchases": DATA_PROFILES[profile]["target_purchases"],
        "actual_purchases": int(timestamps.size),
        "actual_customers": int(np.unique(customers).size),
        "actual_articles": int(np.unique(articles).size),
        "min_timestamp": minimum,
        "max_timestamp": maximum,
        "date_range": {"start": iso(minimum), "end": iso(maximum)},
        "source_fingerprint": source_fingerprint(source_manifest),
        "strategy": "full" if profile == "full" else SAMPLE_STRATEGY,
        "seed": int(seed),
        "sampled": sampled,
    }


def sample_customer_histories(
    purchases: Any,
    *,
    profile: DataProfile,
    seed: int,
    source_manifest: Mapping[str, Any],
) -> tuple[SampledPurchases, dict[str, Any]]:
    """Select complete customer purchase histories with deterministic activity strata."""
    if profile not in DATA_PROFILES:
        raise ValueError(f"Unknown data profile: {profile}")
    values = {
        name: np.asarray(purchases[name])
        for name in ("timestamp", "customer_key", "article_key", "price", "sales_channel_id")
    }
    sizes = {array.size for array in values.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("Training purchases must be non-empty aligned columns")
    all_purchases = SampledPurchases(values)
    target = DATA_PROFILES[profile]["target_purchases"]
    if profile == "full" or target is None:
        return all_purchases, _purchase_manifest(
            all_purchases,
            profile=profile,
            seed=seed,
            source_manifest=source_manifest,
            sampled=False,
        )

    customers, inverse, counts = np.unique(
        values["customer_key"].astype(np.int64, copy=False), return_inverse=True, return_counts=True
    )
    activity = np.searchsorted(np.asarray([2, 5, 10, 50]), counts, side="left")
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = {}
    for bucket in range(5):
        members = np.flatnonzero(activity == bucket)
        if members.size:
            groups[bucket] = rng.permutation(members).tolist()

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_events = 0

    def add(index: int) -> None:
        nonlocal selected_events
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
            selected_events += int(counts[index])

    active = sorted(groups)
    while selected_events < target and active:
        remaining: list[int] = []
        for key in active:
            members = groups[key]
            while members and members[-1] in selected_set:
                members.pop()
            if members and selected_events < target:
                add(members.pop())
            if members:
                remaining.append(key)
        active = remaining

    selected_customer_keys = customers[np.asarray(selected, dtype=np.int64)]
    mask = np.isin(values["customer_key"], selected_customer_keys)
    sampled_purchases = SampledPurchases({name: array[mask] for name, array in values.items()})
    manifest = _purchase_manifest(
        sampled_purchases,
        profile=profile,
        seed=seed,
        source_manifest=source_manifest,
        sampled=len(selected) < customers.size,
    )
    return sampled_purchases, manifest


def sample_cache_paths(
    cache_directory: str | Path,
    *,
    profile: DataProfile,
    seed: int,
    source_manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    fingerprint = source_fingerprint(source_manifest)[:16]
    stem = f"sample-v{SAMPLE_CACHE_VERSION}-{profile}-seed{seed}-{fingerprint}"
    root = Path(cache_directory)
    return root / f"{stem}.npz", root / f"{stem}.manifest.json"


def load_cached_sample(
    cache_directory: str | Path,
    *,
    profile: DataProfile,
    seed: int,
    source_manifest: Mapping[str, Any],
) -> tuple[SampledPurchases, dict[str, Any]] | None:
    data_path, manifest_path = sample_cache_paths(
        cache_directory,
        profile=profile,
        seed=seed,
        source_manifest=source_manifest,
    )
    if not data_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(data_path, allow_pickle=False) as data:
            purchases = SampledPurchases(
                {
                    name: np.asarray(data[name])
                    for name in ("timestamp", "customer_key", "article_key", "price", "sales_channel_id")
                }
            )
        expected = _purchase_manifest(
            purchases,
            profile=profile,
            seed=seed,
            source_manifest=source_manifest,
            sampled=bool(manifest.get("sampled")),
        )
        if manifest != expected:
            return None
        return purchases, manifest
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def write_cached_sample(
    cache_directory: str | Path,
    purchases: SampledPurchases,
    manifest: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    profile = str(manifest["profile"])
    if profile not in DATA_PROFILES:
        raise ValueError("Sample manifest has invalid profile")
    data_path, manifest_path = sample_cache_paths(
        cache_directory,
        profile=profile,  # type: ignore[arg-type]
        seed=int(manifest["seed"]),
        source_manifest=source_manifest,
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_data = data_path.with_name(f".{data_path.name}.{token}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        with temporary_data.open("wb") as stream:
            np.savez_compressed(stream, **purchases.values)
        temporary_manifest.write_text(
            json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_data, data_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_data.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return data_path, manifest_path


@dataclass(frozen=True)
class InteractionIndex:
    """Known pairs in compact CSR form, grouped by dense user index."""

    offsets: np.ndarray
    items: np.ndarray
    num_items: int

    @classmethod
    def from_pairs(
        cls,
        user_indices: Iterable[int] | np.ndarray,
        item_indices: Iterable[int] | np.ndarray,
        num_items: int,
    ) -> "InteractionIndex":
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        user_array = np.asarray(list(user_indices) if not isinstance(user_indices, np.ndarray) else user_indices)
        item_array = np.asarray(list(item_indices) if not isinstance(item_indices, np.ndarray) else item_indices)
        if user_array.size != item_array.size:
            raise ValueError("user_indices and item_indices must have equal length")
        users = user_array.astype(np.int64, copy=False).reshape(-1)
        items = item_array.astype(np.int64, copy=False).reshape(-1)
        if users.size == 0:
            return cls(np.asarray([0], dtype=np.int64), items, num_items)
        if np.any(users < 0) or np.any((items < 0) | (items >= num_items)):
            raise ValueError("Interaction indices are outside mapping bounds")
        order = np.lexsort((items, users))
        users, items = users[order], items[order]
        unique = np.ones(users.size, dtype=bool)
        unique[1:] = (users[1:] != users[:-1]) | (items[1:] != items[:-1])
        users, items = users[unique], items[unique]
        counts = np.bincount(users, minlength=int(users[-1]) + 1)
        offsets = np.empty(counts.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(offsets, items, num_items)

    def contains(self, user_index: int, item_index: int) -> bool:
        seen = self.seen(user_index)
        position = int(np.searchsorted(seen, item_index))
        return position < seen.size and int(seen[position]) == item_index

    def seen(self, user_index: int) -> np.ndarray:
        user = int(user_index)
        if user < 0 or user + 1 >= self.offsets.size:
            return np.empty(0, dtype=np.int64)
        return self.items[self.offsets[user] : self.offsets[user + 1]]


class NegativeSampler:
    def __init__(self, interactions: InteractionIndex, negative_ratio: int = 2, seed: int = 42):
        if negative_ratio < 0:
            raise ValueError("negative_ratio cannot be negative")
        self.interactions = interactions
        self.negative_ratio = int(negative_ratio)
        self.seed = int(seed)
        self._complements: dict[int, np.ndarray] = {}
        counts = np.diff(interactions.offsets)
        known_users = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
        self._known_keys = known_users * interactions.num_items + interactions.items

    def _contains_many(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users.astype(np.int64, copy=False) * self.interactions.num_items + items
        positions = np.searchsorted(self._known_keys, keys)
        valid = positions < self._known_keys.size
        result = np.zeros(keys.size, dtype=bool)
        result[valid] = self._known_keys[positions[valid]] == keys[valid]
        return result

    def _negative(self, user: int, rng: np.random.Generator) -> int:
        seen = self.interactions.seen(user)
        if len(seen) >= self.interactions.num_items:
            raise ValueError(f"Customer {user} has purchased every catalog article")
        for _ in range(32):
            candidate = int(rng.integers(self.interactions.num_items))
            if not self.interactions.contains(user, candidate):
                return candidate
        candidates = self._complements.get(user)
        if candidates is None:
            mask = np.ones(self.interactions.num_items, dtype=bool)
            if seen.size:
                mask[seen] = False
            candidates = np.flatnonzero(mask)
            self._complements[user] = candidates
        return int(candidates[int(rng.integers(candidates.size))])

    def iter_examples(
        self,
        user_indices: Iterable[int] | np.ndarray,
        item_indices: Iterable[int] | np.ndarray,
        positive_weights: Iterable[float] | np.ndarray | None = None,
        *,
        epoch: int = 0,
    ) -> Iterator[tuple[dict[str, np.int32], np.float32, np.float32]]:
        users = np.asarray(
            user_indices if isinstance(user_indices, np.ndarray) else list(user_indices),
            dtype=np.int64,
        ).reshape(-1)
        items = np.asarray(
            item_indices if isinstance(item_indices, np.ndarray) else list(item_indices),
            dtype=np.int64,
        ).reshape(-1)
        if users.size != items.size:
            raise ValueError("user_indices and item_indices must have equal length")
        weights = (
            np.ones(users.size, dtype=np.float32)
            if positive_weights is None
            else np.asarray(
                positive_weights
                if isinstance(positive_weights, np.ndarray)
                else list(positive_weights),
                dtype=np.float32,
            ).reshape(-1)
        )
        if weights.size != users.size:
            raise ValueError("positive_weights must match positive events")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("positive_weights must be finite and positive")
        rng = np.random.default_rng(self.seed + int(epoch))
        for user, item, weight in zip(users, items, weights, strict=True):
            user_value, item_value = int(user), int(item)
            if not self.interactions.contains(user_value, item_value):
                raise ValueError("Positive event is missing from interaction index")
            yield (
                {"customer_index": np.int32(user_value), "article_index": np.int32(item_value)},
                np.float32(1.0),
                np.float32(weight),
            )
            for _ in range(self.negative_ratio):
                negative = self._negative(user_value, rng)
                yield (
                    {"customer_index": np.int32(user_value), "article_index": np.int32(negative)},
                    np.float32(0.0),
                    np.float32(1.0),
                )

    def iter_batches(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        positive_weights: np.ndarray | None = None,
        features: HybridFeatures | None = None,
        *,
        epoch: int = 0,
        positive_batch_size: int = 16_384,
    ) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """Generate bounded blocks instead of crossing into TensorFlow per example."""
        if positive_batch_size <= 0:
            raise ValueError("positive_batch_size must be positive")
        users = np.asarray(user_indices, dtype=np.int64).reshape(-1)
        items = np.asarray(item_indices, dtype=np.int64).reshape(-1)
        weights = (
            np.ones(users.size, dtype=np.float32)
            if positive_weights is None
            else np.asarray(positive_weights, dtype=np.float32).reshape(-1)
        )
        if users.size != items.size or weights.size != users.size:
            raise ValueError("users, items, and weights must have equal length")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("positive_weights must be finite and positive")

        rng = np.random.default_rng(self.seed + int(epoch))
        stride = 1 + self.negative_ratio
        for start in range(0, users.size, positive_batch_size):
            stop = min(start + positive_batch_size, users.size)
            source_users = users[start:stop]
            source_items = items[start:stop]
            source_weights = weights[start:stop]
            size = source_users.size * stride
            batch_users = np.repeat(source_users, stride).astype(np.int32, copy=False)
            batch_items = np.empty(size, dtype=np.int32)
            labels = np.zeros(size, dtype=np.float32)
            sample_weights = np.ones(size, dtype=np.float32)
            positive_positions = np.arange(0, size, stride)
            batch_items[positive_positions] = source_items.astype(np.int32, copy=False)
            labels[positive_positions] = 1.0
            sample_weights[positive_positions] = source_weights

            if np.any(~self._contains_many(source_users, source_items)):
                raise ValueError("Positive event is missing from interaction index")
            if self.negative_ratio:
                negative_positions = np.flatnonzero(labels == 0)
                negative_users = batch_users[negative_positions].astype(np.int64, copy=False)
                candidates = rng.integers(
                    self.interactions.num_items,
                    size=negative_positions.size,
                    dtype=np.int64,
                )
                collisions = self._contains_many(negative_users, candidates)
                while np.any(collisions):
                    candidates[collisions] = rng.integers(
                        self.interactions.num_items,
                        size=int(np.count_nonzero(collisions)),
                        dtype=np.int64,
                    )
                    collisions = self._contains_many(negative_users, candidates)
                batch_items[negative_positions] = candidates.astype(np.int32, copy=False)

            model_inputs = (
                features.model_inputs(batch_users, batch_items)
                if features is not None
                else {"customer_index": batch_users, "article_index": batch_items}
            )
            yield model_inputs, labels, sample_weights


def build_tf_dataset(
    user_indices: Iterable[int] | np.ndarray,
    item_indices: Iterable[int] | np.ndarray,
    interactions: InteractionIndex,
    *,
    features: HybridFeatures | None = None,
    positive_weights: Iterable[float] | np.ndarray | None = None,
    negative_ratio: int = 2,
    batch_size: int = 4096,
    shuffle: bool = True,
    shuffle_buffer: int = 100_000,
    seed: int = 42,
    resample_each_iteration: bool = True,
) -> Any:
    """Build a finite tf.data pipeline with optional per-iteration resampling."""
    tensorflow = require_tensorflow()
    users = np.asarray(
        user_indices if isinstance(user_indices, np.ndarray) else list(user_indices),
        dtype=np.int64,
    ).reshape(-1)
    items = np.asarray(
        item_indices if isinstance(item_indices, np.ndarray) else list(item_indices),
        dtype=np.int64,
    ).reshape(-1)
    weights = (
        None
        if positive_weights is None
        else np.asarray(
            positive_weights
            if isinstance(positive_weights, np.ndarray)
            else list(positive_weights),
            dtype=np.float32,
        )
    )
    sampler = NegativeSampler(interactions, negative_ratio, seed)
    counter = 0
    lock = Lock()

    def generator() -> Iterator[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        nonlocal counter
        epoch = 0
        if resample_each_iteration:
            with lock:
                epoch = counter
                counter += 1
        return sampler.iter_batches(users, items, weights, features, epoch=epoch)

    feature_names = (
        ("customer_index", "article_index", *features.customer_values, *features.article_values)
        if features is not None
        else ("customer_index", "article_index")
    )
    signature = (
        {
            name: tensorflow.TensorSpec(shape=(None,), dtype=tensorflow.int32)
            for name in feature_names
        },
        tensorflow.TensorSpec(shape=(None,), dtype=tensorflow.float32),
        tensorflow.TensorSpec(shape=(None,), dtype=tensorflow.float32),
    )
    dataset = tensorflow.data.Dataset.from_generator(
        generator, output_signature=signature
    ).unbatch()
    example_count = users.size * (1 + negative_ratio)
    if shuffle and example_count:
        dataset = dataset.shuffle(
            min(int(example_count), int(shuffle_buffer)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    batch_count = (int(example_count) + int(batch_size) - 1) // int(batch_size)
    dataset = dataset.batch(batch_size).apply(
        tensorflow.data.experimental.assert_cardinality(batch_count)
    )
    return dataset.prefetch(tensorflow.data.AUTOTUNE)
