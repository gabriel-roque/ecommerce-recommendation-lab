from collections import Counter
import importlib.util

import numpy as np
import pytest

from app.training.config import TrainingConfig
from app.training.sampling import (
    InteractionIndex,
    NegativeSampler,
    build_tf_dataset,
    load_cached_sample,
    sample_customer_histories,
    write_cached_sample,
)


def test_negative_sampling_keeps_every_positive_and_never_uses_seen_item():
    users = np.asarray([0, 0, 1, 1])
    items = np.asarray([0, 1, 1, 2])
    interactions = InteractionIndex.from_pairs(users, items, num_items=5)
    sampler = NegativeSampler(interactions, negative_ratio=2, seed=7)

    examples = list(sampler.iter_examples(users, items, [1, 3, 1, 5]))
    positives = [(int(row[0]["customer_index"]), int(row[0]["article_index"])) for row in examples if row[1] == 1]
    negatives = [(int(row[0]["customer_index"]), int(row[0]["article_index"])) for row in examples if row[1] == 0]

    assert Counter(positives) == Counter(zip(users.tolist(), items.tolist(), strict=True))
    assert len(negatives) == len(positives) * 2
    assert all(not interactions.contains(user, item) for user, item in negatives)


def test_negative_sampling_is_generated_again_for_each_epoch():
    interactions = InteractionIndex.from_pairs([0], [0], num_items=100)
    sampler = NegativeSampler(interactions, negative_ratio=8, seed=11)
    first = [int(row[0]["article_index"]) for row in sampler.iter_examples([0], [0], epoch=0)]
    second = [int(row[0]["article_index"]) for row in sampler.iter_examples([0], [0], epoch=1)]
    assert first[0] == second[0] == 0
    assert first[1:] != second[1:]


def test_batched_sampling_keeps_positives_and_rejects_known_negatives():
    users = np.asarray([0, 0, 1], dtype=np.int64)
    items = np.asarray([1, 3, 2], dtype=np.int64)
    interactions = InteractionIndex.from_pairs(users, items, num_items=6)
    sampler = NegativeSampler(interactions, negative_ratio=2, seed=9)

    batches = list(sampler.iter_batches(users, items, positive_batch_size=2))
    labels = np.concatenate([batch[1] for batch in batches])
    assert labels.tolist().count(1.0) == len(users)
    for features, batch_labels, _ in batches:
        for user, item, label in zip(
            features["customer_index"], features["article_index"], batch_labels, strict=True
        ):
            if label == 0:
                assert not interactions.contains(int(user), int(item))


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed")
def test_validation_negative_sampling_is_stable_between_iterations():
    interactions = InteractionIndex.from_pairs([0], [0], num_items=100)
    dataset = build_tf_dataset(
        [0],
        [0],
        interactions,
        negative_ratio=8,
        batch_size=9,
        shuffle=False,
        seed=11,
        resample_each_iteration=False,
    )

    first = list(dataset.as_numpy_iterator())
    second = list(dataset.as_numpy_iterator())

    assert int(dataset.cardinality().numpy()) == 1
    np.testing.assert_array_equal(first[0][0]["article_index"], second[0][0]["article_index"])


def test_training_config_requires_enough_evaluation_negatives_for_top_10():
    with pytest.raises(ValueError, match="at least 10"):
        TrainingConfig(evaluation_negatives=9)


def test_training_config_profile_defaults_and_validation():
    config = TrainingConfig()
    assert config.data_profile == "balanced"
    assert config.epochs == 3
    assert config.evaluation_negatives == 100
    with pytest.raises(ValueError, match="data_profile"):
        TrainingConfig(data_profile="tiny")  # type: ignore[arg-type]


def _profile_events():
    customers = np.repeat(np.arange(100, dtype=np.int64), 300)
    return {
        "timestamp": np.arange(customers.size, dtype=np.int64),
        "customer_key": customers,
        "article_key": np.arange(customers.size, dtype=np.int64) % 500,
        "price": np.full(customers.size, 0.1, dtype=np.float32),
        "sales_channel_id": np.full(customers.size, 2, dtype=np.int16),
    }


def test_profile_sampling_is_deterministic_and_keeps_whole_user_histories():
    source = {"dataset": "ecommerce_sintetico_br", "semantics": "purchase_only", "purchases": 30_000, "sampled": False}
    first, first_manifest = sample_customer_histories(
        _profile_events(), profile="fast", seed=7, source_manifest=source
    )
    second, second_manifest = sample_customer_histories(
        _profile_events(), profile="fast", seed=7, source_manifest=source
    )

    for name in first.values:
        np.testing.assert_array_equal(first[name], second[name])
    assert first_manifest == second_manifest
    source_counts = Counter(_profile_events()["customer_key"])
    sampled_counts = Counter(first["customer_key"])
    assert all(sampled_counts[user] == source_counts[user] for user in sampled_counts)
    assert first_manifest["actual_purchases"] == len(first)
    assert first_manifest["source_fingerprint"]
    assert first_manifest["strategy"] == "deterministic_customer_purchase_history_v2"


def test_profile_sample_cache_round_trip(tmp_path):
    source = {"dataset": "ecommerce_sintetico_br", "semantics": "purchase_only", "purchases": 30_000, "sampled": False}
    events, manifest = sample_customer_histories(
        _profile_events(), profile="fast", seed=11, source_manifest=source
    )
    write_cached_sample(tmp_path, events, manifest, source_manifest=source)

    cached = load_cached_sample(
        tmp_path, profile="fast", seed=11, source_manifest=source
    )

    assert cached is not None
    cached_events, cached_manifest = cached
    np.testing.assert_array_equal(cached_events["customer_key"], events["customer_key"])
    assert cached_manifest == manifest
