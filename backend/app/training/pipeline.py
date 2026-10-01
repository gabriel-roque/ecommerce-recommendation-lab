"""Temporal benchmark, artifact, and reload training workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from app.models.artifact import LoadedArtifact, save_artifact, validate_artifact
from app.models.features import HybridFeatures
from app.models.mappings import DenseMappings
from app.models.ncf import build_hybrid_model, require_tensorflow

from .callbacks import TrainingVisualizationCallback, assert_model_weights_finite
from .config import TrainingConfig
from .hardware import hardware_info
from .metrics import SampledRankingResult, evaluate_sampled_ranking
from .sampling import InteractionIndex, build_tf_dataset
from .split import TemporalSplit, temporal_split


RecordSink = Callable[[dict[str, Any]], None]
StatusSink = Callable[[str], None]


@dataclass
class TrainingObserver:
    status_sink: StatusSink | None = None
    metric_sink: RecordSink | None = None
    snapshot_sink: RecordSink | None = None

    def status(self, value: str) -> None:
        if self.status_sink is not None:
            self.status_sink(value)


@dataclass
class TrainingOutcome:
    artifact: LoadedArtifact
    ranking: SampledRankingResult
    benchmark_history: dict[str, list[float]]
    benchmark_metrics: list[dict[str, Any]]
    final_metrics: list[dict[str, Any]]
    durations: dict[str, float] = field(default_factory=dict)


def _column(purchases: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    try:
        values = purchases[name]
    except (KeyError, TypeError, IndexError):
        values = [purchase[name] for purchase in purchases]
    array = np.asarray(values)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _size(purchases: Any) -> int:
    return len(purchases)


class TrainingPipeline:
    def __init__(
        self,
        artifact_root: str | Path,
        config: TrainingConfig | None = None,
        observer: TrainingObserver | None = None,
        model_builder: Callable[..., Any] = build_hybrid_model,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self.config = config or TrainingConfig.from_settings()
        self.observer = observer or TrainingObserver()
        self.model_builder = model_builder

    def _mapped(self, purchases: Any, mappings: DenseMappings) -> tuple[np.ndarray, np.ndarray]:
        customers = mappings.customer_indices(_column(purchases, "customer_key", np.int64))
        articles = mappings.article_indices(_column(purchases, "article_key", np.int64))
        if np.any(customers < 0) or np.any(articles < 0):
            raise ValueError("Purchases contain keys absent from dense mappings")
        return customers, articles

    def _model(self, mappings: DenseMappings, features: HybridFeatures) -> Any:
        return self.model_builder(
            mappings.num_customers,
            mappings.num_articles,
            features.cardinalities,
            embedding_dim=self.config.embedding_dim,
            dense_1=self.config.dense_1,
            dense_2=self.config.dense_2,
            learning_rate=self.config.learning_rate,
        )

    def _dataset(
        self,
        purchases: Any,
        mappings: DenseMappings,
        features: HybridFeatures,
        interactions: InteractionIndex,
        *,
        shuffle: bool,
        seed_offset: int = 0,
        resample_each_iteration: bool = True,
    ) -> Any:
        customers, articles = self._mapped(purchases, mappings)
        return build_tf_dataset(
            customers,
            articles,
            interactions,
            features=features,
            negative_ratio=self.config.negative_ratio,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            shuffle_buffer=self.config.shuffle_buffer,
            seed=self.config.random_seed + seed_offset,
            resample_each_iteration=resample_each_iteration,
        )

    def _callback(
        self,
        purchases: Any,
        features: HybridFeatures,
        probe: tuple[int, int, int, int],
        run_id: int | str | None,
        *,
        phase: str,
        snapshots: bool = True,
    ) -> TrainingVisualizationCallback:
        return TrainingVisualizationCallback(
            run_id=run_id,
            phase=phase,
            examples_per_epoch=_size(purchases) * (1 + self.config.negative_ratio),
            metric_sink=self.observer.metric_sink,
            snapshot_sink=self.observer.snapshot_sink if snapshots else None,
            probe_customer_index=probe[0],
            probe_article_index=probe[1],
            probe_customer_key=probe[2],
            probe_article_key=probe[3],
            probe_features=features.model_inputs(
                np.asarray([probe[0]], dtype=np.int32),
                np.asarray([probe[1]], dtype=np.int32),
            ),
        )

    def run(
        self,
        purchases: Any,
        *,
        customers: Any,
        articles: Any,
        model_version: str,
        dataset_manifest: Mapping[str, Any],
        training_manifest: Mapping[str, Any] | None = None,
        run_id: int | str | None = None,
    ) -> TrainingOutcome:
        try:
            return self._run(
                purchases,
                customers=customers,
                articles=articles,
                model_version=model_version,
                dataset_manifest=dataset_manifest,
                training_manifest=training_manifest,
                run_id=run_id,
            )
        except BaseException as error:
            fail = getattr(self.observer, "fail", None)
            if callable(fail):
                fail(error)
            else:
                self.observer.status("FAILED")
            raise

    def _run(
        self,
        purchases: Any,
        *,
        customers: Any,
        articles: Any,
        model_version: str,
        dataset_manifest: Mapping[str, Any],
        training_manifest: Mapping[str, Any] | None = None,
        run_id: int | str | None = None,
    ) -> TrainingOutcome:
        tensorflow = require_tensorflow()
        tensorflow.keras.utils.set_random_seed(self.config.random_seed)
        if _size(purchases) == 0:
            raise ValueError("Cannot train with no purchases")
        started = time.perf_counter()
        self.observer.status("PREPARING_DATA")
        split_started = time.perf_counter()
        split: TemporalSplit = temporal_split(purchases)
        mappings = DenseMappings.build(
            _column(customers, "customer_key", np.int64),
            _column(articles, "article_key", np.int64),
        )
        features = HybridFeatures.build(mappings, customers, articles)
        train_users, train_items = self._mapped(split.train, mappings)
        validation_users, validation_items = self._mapped(split.validation, mappings)
        train_interactions = InteractionIndex.from_pairs(
            train_users, train_items, mappings.num_articles
        )
        validation_interactions = InteractionIndex.from_pairs(
            np.concatenate((train_users, validation_users)),
            np.concatenate((train_items, validation_items)),
            mappings.num_articles,
        )
        if self.config.probe_customer_key is not None:
            probe_customer_index = mappings.customer_index(self.config.probe_customer_key)
            probe_article_index = mappings.article_index(self.config.probe_article_key)  # type: ignore[arg-type]
            if probe_customer_index is None or probe_article_index is None:
                raise ValueError("Configured probe IDs are absent from mappings")
            probe = (
                probe_customer_index,
                probe_article_index,
                self.config.probe_customer_key,
                self.config.probe_article_key,  # type: ignore[arg-type]
            )
        else:
            probe = (
                int(train_users[0]),
                int(train_items[0]),
                int(_column(split.train, "customer_key", np.int64)[0]),
                int(_column(split.train, "article_key", np.int64)[0]),
            )
        data_duration = time.perf_counter() - split_started

        self.observer.status("TRAINING")
        benchmark_started = time.perf_counter()
        benchmark_model = self._model(mappings, features)
        benchmark_callback = self._callback(
            split.train, features, probe, run_id, phase="BENCHMARK"
        )
        callbacks: list[Any] = [benchmark_callback]
        callback_module = getattr(tensorflow.keras, "callbacks", None)
        if callback_module is not None:
            callbacks.extend(
                [
                    callback_module.EarlyStopping(
                        monitor="val_loss", patience=2, restore_best_weights=True
                    ),
                    callback_module.TerminateOnNaN(),
                ]
            )
        history = benchmark_model.fit(
            self._dataset(
                split.train, mappings, features, train_interactions, shuffle=True
            ),
            validation_data=self._dataset(
                split.validation,
                mappings,
                features,
                validation_interactions,
                shuffle=False,
                seed_offset=1,
                resample_each_iteration=False,
            ),
            epochs=self.config.epochs,
            callbacks=callbacks,
            shuffle=False,
            verbose=0,
        )
        benchmark_history = {
            key: [float(value) for value in values] for key, values in history.history.items()
        }
        benchmark_metrics = list(benchmark_callback.metrics)
        benchmark_duration = time.perf_counter() - benchmark_started

        self.observer.status("EVALUATING")
        evaluation_started = time.perf_counter()
        test_users, test_items = self._mapped(split.test, mappings)
        ranking = evaluate_sampled_ranking(
            benchmark_model,
            test_users,
            test_items,
            validation_interactions,
            features,
            k=10,
            negative_candidates=self.config.evaluation_negatives,
            batch_size=self.config.evaluation_batch_size,
            seed=self.config.random_seed,
        )
        evaluation_duration = time.perf_counter() - evaluation_started
        final_metrics: list[dict[str, Any]] = []

        durations = {
            "data_preparation_seconds": data_duration,
            "benchmark_training_seconds": benchmark_duration,
            "evaluation_seconds": evaluation_duration,
        }
        metrics = {
            "benchmark_history": benchmark_history,
            "ranking": ranking.to_dict(),
            "benchmark_epochs": benchmark_metrics,
            "final_epochs": final_metrics,
            "durations": durations,
            "hardware": hardware_info(),
            "split_purchase_counts": {
                "train": _size(split.train),
                "validation": _size(split.validation),
                "test": _size(split.test),
            },
            "data_scope": dict(training_manifest or {}),
        }

        self.observer.status("SAVING_MODEL")
        artifact_path = self.artifact_root / model_version
        assert_model_weights_finite(benchmark_model)
        save_artifact(
            artifact_path,
            benchmark_model,
            mappings,
            features,
            self.config,
            metrics,
            dataset_manifest,
            training_manifest=training_manifest,
        )
        del benchmark_model, history, benchmark_callback
        tensorflow.keras.backend.clear_session()
        loaded = validate_artifact(artifact_path)
        durations["total_seconds"] = time.perf_counter() - started
        return TrainingOutcome(
            artifact=loaded,
            ranking=ranking,
            benchmark_history=benchmark_history,
            benchmark_metrics=benchmark_metrics,
            final_metrics=final_metrics,
            durations=durations,
        )
