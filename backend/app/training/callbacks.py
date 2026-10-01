"""Keras callback for real metrics, activations, and weight snapshots."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

try:
    import tensorflow as tf

    _CallbackBase = tf.keras.callbacks.Callback
except ModuleNotFoundError:
    tf = None  # type: ignore[assignment]

    class _CallbackBase:  # type: ignore[no-redef]
        def set_model(self, model: Any) -> None:
            self.model = model


RecordSink = Callable[[dict[str, Any]], None]
_FINITE_CHECK_CHUNK_SIZE = 1_000_000


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else None


def assert_model_weights_finite(model: Any) -> None:
    """Validate all model weights at artifact boundaries, not after every epoch."""
    for weight in getattr(model, "weights", ()):
        if tf is not None and hasattr(weight, "dtype"):
            values = tf.reshape(weight, (-1,))
            size = int(tf.size(values).numpy())
            finite = all(
                bool(
                    tf.reduce_all(
                        tf.math.is_finite(values[offset : offset + _FINITE_CHECK_CHUNK_SIZE])
                    ).numpy()
                )
                for offset in range(0, size, _FINITE_CHECK_CHUNK_SIZE)
            )
        else:
            values = np.asarray(weight.numpy() if hasattr(weight, "numpy") else weight)
            flattened = values.reshape(-1)
            finite = all(
                bool(np.all(np.isfinite(flattened[offset : offset + _FINITE_CHECK_CHUNK_SIZE])))
                for offset in range(0, flattened.size, _FINITE_CHECK_CHUNK_SIZE)
            )
        if not finite:
            raise FloatingPointError(f"Non-finite model weight detected: {weight.name}")


class TrainingVisualizationCallback(_CallbackBase):
    """Publishes measured values only; no progress or activation is synthesized."""

    def __init__(
        self,
        *,
        run_id: int | str | None = None,
        phase: str = "BENCHMARK",
        examples_per_epoch: int,
        metric_sink: RecordSink | None = None,
        snapshot_sink: RecordSink | None = None,
        probe_customer_index: int | None = None,
        probe_article_index: int | None = None,
        probe_customer_key: int | None = None,
        probe_article_key: int | None = None,
        probe_features: dict[str, np.ndarray] | None = None,
        activation_sample_size: int = 8,
        weight_rows: int = 8,
        weight_columns: int = 8,
    ) -> None:
        super().__init__()
        if examples_per_epoch <= 0:
            raise ValueError("examples_per_epoch must be positive")
        self.run_id = run_id
        self.phase = phase
        self.examples_per_epoch = int(examples_per_epoch)
        self.metric_sink = metric_sink
        self.snapshot_sink = snapshot_sink
        self.probe_customer_index = probe_customer_index
        self.probe_article_index = probe_article_index
        self.probe_customer_key = probe_customer_key
        self.probe_article_key = probe_article_key
        self.probe_features = probe_features
        self.activation_sample_size = int(activation_sample_size)
        self.weight_rows = int(weight_rows)
        self.weight_columns = int(weight_columns)
        self.metrics: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self._epoch_started = 0.0
        self._probe_model: Any = None

    def set_model(self, model: Any) -> None:
        super().set_model(model)
        if (
            tf is not None
            and self.probe_customer_index is not None
            and self.probe_article_index is not None
        ):
            layers = [
                model.get_layer(name)
                for name in (
                    "customer_embedding",
                    "article_embedding",
                    "customer_tower",
                    "article_tower",
                    "dense_64",
                    "dense_32",
                    "score",
                )
            ]
            self._probe_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[layer.output for layer in layers],
                name="training_visualization_probe",
            )

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._epoch_started = time.perf_counter()

    def _learning_rate(self) -> float | None:
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return None
        learning_rate = optimizer.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate(optimizer.iterations)
        return _number(learning_rate)

    def _assert_finite(self, logs: dict[str, Any]) -> None:
        for name in ("loss", "val_loss"):
            value = _number(logs.get(name))
            if value is not None and not np.isfinite(value):
                raise FloatingPointError(f"Non-finite {name} detected")
        assert_model_weights_finite(self.model)

    @staticmethod
    def _assert_finite_logs(logs: dict[str, Any]) -> None:
        for name in ("loss", "val_loss"):
            value = _number(logs.get(name))
            if value is not None and not np.isfinite(value):
                raise FloatingPointError(f"Non-finite {name} detected")

    def _snapshot(self, epoch: int) -> dict[str, Any] | None:
        if self._probe_model is None:
            return None
        if self.probe_features is None:
            return None
        outputs = self._probe_model.predict(self.probe_features, verbose=0)
        names = (
            "customer_embedding",
            "article_embedding",
            "customer_tower",
            "article_tower",
            "dense_64",
            "dense_32",
            "score",
        )
        activations: dict[str, list[float]] = {}
        for name, output in zip(names, outputs, strict=True):
            values = np.asarray(output).reshape(-1)
            if np.any(~np.isfinite(values)):
                raise FloatingPointError(f"Non-finite {name} activation detected")
            activations[name] = values[: self.activation_sample_size].astype(float).tolist()

        connections: dict[str, list[list[float]]] = {}
        for name in ("dense_64", "dense_32", "score"):
            kernel = np.asarray(self.model.get_layer(name).get_weights()[0])
            rows = kernel.shape[0] if name == "dense_64" else self.weight_rows
            sampled = kernel[:rows, : self.weight_columns]
            connections[name] = sampled.astype(float).tolist()
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "epoch": epoch + 1,
            "probe_customer_index": self.probe_customer_index,
            "probe_article_index": self.probe_article_index,
            "probe_customer_key": self.probe_customer_key,
            "probe_article_key": self.probe_article_key,
            "activations": activations,
            "weights": connections,
            "prediction": activations["score"][0],
        }

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        values = logs or {}
        self._assert_finite_logs(values)
        duration = time.perf_counter() - self._epoch_started
        record = {
            "run_id": self.run_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch + 1,
            "loss": _number(values.get("loss")),
            "validation_loss": _number(values.get("val_loss")),
            "val_loss": _number(values.get("val_loss")),
            "auc": _number(values.get("auc")),
            "validation_auc": _number(values.get("val_auc")),
            "learning_rate": self._learning_rate(),
            "epoch_duration_seconds": duration,
            "examples_per_second": self.examples_per_epoch / duration if duration else None,
        }
        self.metrics.append(record)
        if self.metric_sink is not None:
            self.metric_sink(record)
        snapshot = self._snapshot(epoch)
        if snapshot is not None:
            self.snapshots.append(snapshot)
            if self.snapshot_sink is not None:
                self.snapshot_sink(snapshot)
