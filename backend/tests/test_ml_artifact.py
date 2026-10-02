import importlib.util

import pytest

from app.models.artifact import save_artifact, validate_artifact
from app.models.features import ARTICLE_FEATURES, CUSTOMER_FEATURES, HybridFeatures
from app.models.mappings import DenseMappings
from app.models.ncf import build_hybrid_model


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed")
def test_complete_artifact_reloads_and_runs_inference(tmp_path):
    mappings = DenseMappings.build([10, 20], [100, 200, 300])
    customers = [
        {"customer_key": key, **{name: f"{name}-{key}" for name in CUSTOMER_FEATURES}}
        for key in mappings.customer_keys
    ]
    articles = [
        {"article_key": key, **{name: f"{name}-{key}" for name in ARTICLE_FEATURES}}
        for key in mappings.article_keys
    ]
    features = HybridFeatures.build(mappings, customers, articles)
    model = build_hybrid_model(
        2, 3, features.cardinalities, embedding_dim=2, dense_1=4, dense_2=2
    )
    destination = tmp_path / "v1"

    save_artifact(
        destination,
        model,
        mappings,
        features,
        {"embedding_dim": 2},
        {"recall_at_10": 0.5},
        {"dataset": "ecommerce_sintetico_br", "semantics": "purchase_only", "purchases": 4, "sampled": False, "source_fingerprint": "abc"},
    )
    assert {path.name for path in destination.iterdir()} == {
        "model.keras",
        "customer_keys.npy",
        "article_keys.npy",
        "features.npz",
        "feature_vocabulary.json",
        "artifact_manifest.json",
        "config.json",
        "metrics.json",
        "dataset_manifest.json",
    }
    loaded = validate_artifact(destination)
    assert loaded.mappings.customer_keys.tolist() == [10, 20]


def test_load_rejects_artifact_without_ecommerce_format(tmp_path):
    from app.models.artifact import load_artifact

    with pytest.raises(ValueError, match="ecommerce hybrid"):
        load_artifact(tmp_path)
