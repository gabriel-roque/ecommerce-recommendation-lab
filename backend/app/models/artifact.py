"""Complete, versioned model artifact persistence."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .features import HybridFeatures
from .mappings import DenseMappings
from .ncf import require_tensorflow

ARTIFACT_FORMAT = "ecommerce-hybrid"
ARTIFACT_VERSION = 2
DATASET_ID = "ecommerce_sintetico_br"


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class LoadedArtifact:
    model: Any
    mappings: DenseMappings
    features: HybridFeatures
    config: dict[str, Any]
    metrics: dict[str, Any]
    dataset_manifest: dict[str, Any]
    directory: Path
    training_manifest: dict[str, Any] | None = None


def save_artifact(
    directory: str | Path,
    model: Any,
    mappings: DenseMappings,
    features: HybridFeatures,
    config: Mapping[str, Any] | Any,
    metrics: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    training_manifest: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write all required files, exposing destination only after success."""
    if dataset_manifest.get("dataset") != DATASET_ID:
        raise ValueError("Only ecommerce_sintetico_br artifacts can be saved")
    if dataset_manifest.get("semantics") != "purchase_only":
        raise ValueError("Artifact requires purchase-only dataset semantics")
    if not dataset_manifest.get("source_fingerprint"):
        raise ValueError("Artifact requires dataset source_fingerprint")
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        model.save(temporary / "model.keras")
        mappings.save(temporary)
        features.save(temporary)
        _write_json(
            temporary / "artifact_manifest.json",
            {"format": ARTIFACT_FORMAT, "version": ARTIFACT_VERSION, "dataset": DATASET_ID},
        )
        _write_json(temporary / "config.json", config)
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "dataset_manifest.json", dataset_manifest)
        if training_manifest is not None:
            _write_json(temporary / "training_manifest.json", training_manifest)
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Artifact already exists: {destination}")
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_artifact(directory: str | Path, *, compile_model: bool = False) -> LoadedArtifact:
    path = Path(directory)
    try:
        with (path / "artifact_manifest.json").open(encoding="utf-8") as stream:
            artifact_manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact is not an ecommerce hybrid artifact") from exc
    expected = {"format": ARTIFACT_FORMAT, "version": ARTIFACT_VERSION, "dataset": DATASET_ID}
    if artifact_manifest != expected:
        raise ValueError("Unsupported ecommerce artifact format")
    tensorflow = require_tensorflow()
    model = tensorflow.keras.models.load_model(path / "model.keras", compile=compile_model)
    mappings = DenseMappings.load(path)
    features = HybridFeatures.load(path)
    features.validate_mappings(mappings)
    with (path / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (path / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    with (path / "dataset_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    training_manifest_path = path / "training_manifest.json"
    training_manifest = None
    if training_manifest_path.is_file():
        with training_manifest_path.open(encoding="utf-8") as stream:
            training_manifest = json.load(stream)
    if (
        manifest.get("dataset") != DATASET_ID
        or manifest.get("semantics") != "purchase_only"
        or not manifest.get("source_fingerprint")
    ):
        raise ValueError("Artifact dataset manifest is invalid")
    return LoadedArtifact(
        model, mappings, features, config, metrics, manifest, path, training_manifest
    )


def validate_artifact(directory: str | Path) -> LoadedArtifact:
    """Reload saved model and require one finite affinity score."""
    loaded = load_artifact(directory)
    from app.training.callbacks import assert_model_weights_finite

    assert_model_weights_finite(loaded.model)
    inputs = loaded.features.model_inputs(
        np.asarray([0], dtype=np.int32), np.asarray([0], dtype=np.int32)
    )
    scores = np.asarray(loaded.model.predict(inputs, verbose=0)).reshape(-1)
    if scores.size != 1 or not np.isfinite(scores[0]):
        raise ValueError("Reloaded artifact produced an invalid score")
    if not 0.0 <= float(scores[0]) <= 1.0:
        raise ValueError("Reloaded artifact score is outside [0, 1]")
    return loaded
