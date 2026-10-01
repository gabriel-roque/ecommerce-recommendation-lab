import numpy as np

from app.training.callbacks import TrainingVisualizationCallback


class Value:
    def __init__(self, value, name="value"):
        self.value = value
        self.name = name

    def numpy(self):
        return np.asarray(self.value)


class FakeModel:
    optimizer = type("Optimizer", (), {"learning_rate": Value(0.001)})()
    weights = [Value([1.0, 2.0], "kernel")]


def test_callback_records_measured_metrics_and_throughput(monkeypatch):
    records = []
    callback = TrainingVisualizationCallback(examples_per_epoch=300, metric_sink=records.append)
    callback.set_model(FakeModel())
    times = iter([10.0, 12.0])
    monkeypatch.setattr("app.training.callbacks.time.perf_counter", lambda: next(times))

    callback.on_epoch_begin(0)
    callback.on_epoch_end(0, {"loss": 0.4, "val_loss": 0.5, "auc": 0.7})

    assert records[0]["loss"] == 0.4
    assert records[0]["validation_loss"] == 0.5
    assert records[0]["epoch_duration_seconds"] == 2
    assert records[0]["examples_per_second"] == 150


def test_callback_rejects_non_finite_loss():
    callback = TrainingVisualizationCallback(examples_per_epoch=1)
    callback.set_model(FakeModel())
    callback.on_epoch_begin(0)
    try:
        callback.on_epoch_end(0, {"loss": float("nan")})
    except FloatingPointError:
        pass
    else:
        raise AssertionError("NaN loss was accepted")


def test_weight_finite_check_uses_bounded_chunks(monkeypatch):
    callback = TrainingVisualizationCallback(examples_per_epoch=1)
    model = FakeModel()
    model.weights = [Value([1.0, 2.0, 3.0, 4.0, 5.0], "kernel")]
    callback.set_model(model)
    checked_sizes = []
    isfinite = np.isfinite

    def record_size(values):
        checked_sizes.append(np.asarray(values).size)
        return isfinite(values)

    monkeypatch.setattr("app.training.callbacks._FINITE_CHECK_CHUNK_SIZE", 2)
    monkeypatch.setattr("app.training.callbacks.np.isfinite", record_size)

    callback._assert_finite({"loss": 0.1})

    assert max(checked_sizes) <= 2
    assert checked_sizes.count(2) == 2


class ProbeModel:
    def predict(self, features, verbose=0):
        del features, verbose
        return [
            np.asarray([[0.1, 0.2]]),
            np.asarray([[0.3, 0.4]]),
            np.asarray([[0.5]]),
            np.asarray([[0.6]]),
            np.asarray([[0.55]]),
            np.asarray([[0.65]]),
            np.asarray([[0.7]]),
        ]


class Layer:
    def __init__(self, kernel):
        self.kernel = kernel

    def get_weights(self):
        return [self.kernel]


class SnapshotModel:
    def __init__(self):
        self.layers = {
            "dense_64": Layer(np.arange(24).reshape(6, 4)),
            "dense_32": Layer(np.arange(12).reshape(4, 3)),
            "score": Layer(np.arange(3).reshape(3, 1)),
        }

    def get_layer(self, name):
        return self.layers[name]


def test_dense_64_snapshot_includes_full_hybrid_input_rows():
    callback = TrainingVisualizationCallback(
        examples_per_epoch=1,
        probe_customer_index=0,
        probe_article_index=0,
        probe_features={"customer_index": np.asarray([0]), "article_index": np.asarray([0])},
        weight_rows=2,
        weight_columns=2,
    )
    callback.model = SnapshotModel()
    callback._probe_model = ProbeModel()

    snapshot = callback._snapshot(0)

    assert snapshot is not None
    assert snapshot["weights"]["dense_64"] == [
        [0.0, 1.0],
        [4.0, 5.0],
        [8.0, 9.0],
        [12.0, 13.0],
        [16.0, 17.0],
        [20.0, 21.0],
    ]
