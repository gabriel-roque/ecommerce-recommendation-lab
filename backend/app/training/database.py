"""Database sinks for training state, measured epochs, and real snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.database.models import (
    ACTIVE_RUN_STATUSES,
    TrainingMetric,
    TrainingNetworkSnapshot,
    TrainingRun,
)
from app.database.session import get_session

from .pipeline import TrainingObserver


class DatabaseTrainingObserver(TrainingObserver):
    def __init__(self, run_id: str, epochs: int) -> None:
        self.run_id = run_id
        self.epochs = epochs
        super().__init__(
            status_sink=self._save_status,
            metric_sink=self._save_metric,
            snapshot_sink=self._save_snapshot,
        )

    def _save_status(self, status: str) -> None:
        if status in {"COMPLETED", "FAILED"}:
            raise ValueError("Terminal status must be persisted by worker finalization")
        with get_session() as session:
            run = session.get(TrainingRun, self.run_id)
            if run is None:
                raise LookupError(f"Training run not found: {self.run_id}")
            if run.status not in ACTIVE_RUN_STATUSES:
                raise RuntimeError(f"Cannot update training run in status {run.status}")
            run.status = status
            now = datetime.now(timezone.utc)
            if status == "PREPARING_DATA" and run.started_at is None:
                run.started_at = now
            run.heartbeat_at = now
            run.updated_at = now
            session.commit()

    def fail(self, error: BaseException) -> None:
        with get_session() as session:
            run = session.get(TrainingRun, self.run_id)
            if run is None or run.status not in ACTIVE_RUN_STATUSES:
                return
            run.status = "FAILED"
            run.error_message = str(error)
            now = datetime.now(timezone.utc)
            run.completed_at = now
            run.heartbeat_at = now
            run.updated_at = now
            session.commit()

    def _save_metric(self, record: dict[str, Any]) -> None:
        phase = str(record.get("phase", "BENCHMARK"))
        metric = TrainingMetric(
            training_run_id=self.run_id,
            phase=phase,
            epoch=int(record["epoch"]),
            loss=float(record["loss"]),
            validation_loss=record.get("validation_loss"),
            auc=record.get("auc"),
            learning_rate=record.get("learning_rate"),
            epoch_duration_seconds=record.get("epoch_duration_seconds"),
            examples_per_second=record.get("examples_per_second"),
        )
        with get_session() as session:
            run = session.get(TrainingRun, self.run_id)
            if run is None:
                raise LookupError(f"Training run not found: {self.run_id}")
            if run.status not in ACTIVE_RUN_STATUSES:
                raise RuntimeError(f"Cannot update training run in status {run.status}")
            session.add(metric)
            run.current_epoch = int(record["epoch"])
            run.heartbeat_at = datetime.now(timezone.utc)
            if phase == "BENCHMARK":
                run.progress = min(0.8, 0.8 * int(record["epoch"]) / self.epochs)
            else:
                run.progress = min(0.99, 0.8 + 0.19 * int(record["epoch"]) / self.epochs)
            session.commit()

    def _save_snapshot(self, record: dict[str, Any]) -> None:
        snapshot = TrainingNetworkSnapshot(
            training_run_id=self.run_id,
            epoch=int(record["epoch"]),
            probe_customer_key=record.get("probe_customer_key"),
            probe_article_key=record.get("probe_article_key"),
            activations=record["activations"],
            weights=record["weights"],
            prediction=record.get("prediction"),
        )
        with get_session() as session:
            run = session.get(TrainingRun, self.run_id)
            if run is None:
                raise LookupError(f"Training run not found: {self.run_id}")
            if run.status not in ACTIVE_RUN_STATUSES:
                raise RuntimeError(f"Cannot update training run in status {run.status}")
            session.add(snapshot)
            session.commit()
