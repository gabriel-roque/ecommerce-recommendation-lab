from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np

import pytest
from sqlalchemy.exc import IntegrityError

from app.database.models import ModelVersion, TrainingRun
from app.config import Settings
from app.training import worker
from app.training.config import TrainingConfig
from app.training.database import DatabaseTrainingObserver
from app.training.sampling import sample_customer_histories, write_cached_sample


def test_fail_abandoned_run_after_heartbeat_timeout(db):
    old_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=5)
    run = TrainingRun(
        status="TRAINING",
        config={},
        worker_id="dead-worker",
        started_at=old_heartbeat,
        heartbeat_at=old_heartbeat,
    )
    db.add(run)
    db.commit()

    assert worker._fail_abandoned_runs(60) == 1

    db.expire_all()
    assert run.status == "FAILED"
    assert run.completed_at is not None
    assert run.error_message == "Worker heartbeat timed out after 60 seconds"


def test_heartbeat_updates_run_during_work(db, monkeypatch):
    old_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=1)
    run = TrainingRun(
        status="TRAINING",
        config={},
        worker_id="worker-1",
        started_at=old_heartbeat,
        heartbeat_at=old_heartbeat,
    )
    db.add(run)
    db.commit()
    touched = threading.Event()
    touch_heartbeat = worker._touch_heartbeat

    def tracked_touch(run_id: str, worker_id: str) -> None:
        touch_heartbeat(run_id, worker_id)
        touched.set()

    monkeypatch.setattr(worker, "_touch_heartbeat", tracked_touch)

    with worker._heartbeat(run.id, "worker-1", 0.01):
        assert touched.wait(1)

    db.expire_all()
    assert run.heartbeat_at is not None
    observed = run.heartbeat_at.replace(tzinfo=timezone.utc) if run.heartbeat_at.tzinfo is None else run.heartbeat_at
    assert observed > old_heartbeat


def test_finalize_activates_model_and_completes_run_in_one_transaction(db):
    previous_run = TrainingRun(status="COMPLETED", config={}, progress=1)
    run = TrainingRun(status="SAVING_MODEL", config={})
    db.add_all([previous_run, run])
    db.flush()
    previous_model = ModelVersion(
        version="previous",
        training_run_id=previous_run.id,
        artifact_path="artifacts/previous",
        status="READY",
        is_active=True,
    )
    db.add(previous_model)
    db.commit()

    worker._finalize_run(run.id, "next", Path("artifacts/next"), TrainingConfig(), {})

    db.expire_all()
    assert run.status == "COMPLETED"
    assert run.progress == 1
    assert run.completed_at is not None
    assert previous_model.status == "ARCHIVED"
    assert previous_model.is_active is False
    active = db.query(ModelVersion).filter_by(is_active=True).one()
    assert active.version == "next"
    assert active.training_run_id == run.id


def test_finalize_rolls_back_run_and_model_switch_together(db):
    previous_run = TrainingRun(status="COMPLETED", config={}, progress=1)
    run = TrainingRun(status="SAVING_MODEL", config={})
    db.add_all([previous_run, run])
    db.flush()
    previous_model = ModelVersion(
        version="duplicate",
        training_run_id=previous_run.id,
        artifact_path="artifacts/previous",
        status="READY",
        is_active=True,
    )
    db.add(previous_model)
    db.commit()

    with pytest.raises(IntegrityError):
        worker._finalize_run(
            run.id,
            "duplicate",
            Path("artifacts/duplicate"),
            TrainingConfig(),
            {},
        )

    db.expire_all()
    assert run.status == "SAVING_MODEL"
    assert previous_model.status == "READY"
    assert previous_model.is_active is True


def test_database_observer_cannot_complete_run(db):
    run = TrainingRun(status="SAVING_MODEL", config={})
    db.add(run)
    db.commit()

    with pytest.raises(ValueError, match="Terminal status"):
        DatabaseTrainingObserver(run.id, 1).status("COMPLETED")

    db.expire_all()
    assert run.status == "SAVING_MODEL"


def test_invalid_claimed_config_marks_run_failed(db):
    run = TrainingRun(status="QUEUED", config={"epochs": 0})
    db.add(run)
    db.commit()

    with pytest.raises(ValueError, match="epochs must be positive"):
        worker.run_once(
            worker_id="worker-1",
            heartbeat_interval_seconds=1,
            heartbeat_timeout_seconds=10,
        )

    db.expire_all()
    assert run.status == "FAILED"
    assert run.completed_at is not None


def test_training_purchases_loads_cached_profile_without_full_database_load(tmp_path, monkeypatch):
    source_manifest = {
        "dataset": "ecommerce_sintetico_br",
        "source_fingerprint": "abc",
        "semantics": "purchase_only",
        "purchases": 3,
        "sampled": False,
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    source_events = {
        "timestamp": np.asarray([1, 2, 3]),
        "customer_key": np.asarray([10, 10, 20]),
        "article_key": np.asarray([100, 101, 102]),
        "price": np.asarray([0.1, 0.2, 0.3]),
        "sales_channel_id": np.asarray([1, 2, 2]),
    }
    sampled, training_manifest = sample_customer_histories(
        source_events,
        profile="fast",
        seed=42,
        source_manifest=source_manifest,
    )
    samples_dir = tmp_path / "samples"
    write_cached_sample(
        samples_dir,
        sampled,
        training_manifest,
        source_manifest=source_manifest,
    )
    monkeypatch.setattr(
        worker,
        "get_settings",
        lambda: Settings(
            dataset_manifest_path=str(manifest_path), training_samples_dir=str(samples_dir)
        ),
    )
    monkeypatch.setattr(
        worker,
        "_load_purchases",
        lambda: (_ for _ in ()).throw(AssertionError("full database was loaded")),
    )

    purchases, loaded_source, loaded_training = worker._training_purchases(
        TrainingConfig(data_profile="fast")
    )

    assert len(purchases) == 3
    assert loaded_source == source_manifest
    assert loaded_training == training_manifest


def test_wait_for_database_retries_schema_creation(monkeypatch):
    attempts = 0
    sleeps: list[float] = []

    def create_schema() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(worker, "create_schema", create_schema)
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)

    worker._wait_for_database(0.25)

    assert attempts == 2
    assert sleeps == [0.25]


def test_main_keeps_polling_after_run_error(monkeypatch):
    calls = 0

    def run_once() -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("training failed")
        raise KeyboardInterrupt

    monkeypatch.setattr(worker, "_wait_for_database", lambda _: None)
    monkeypatch.setattr(worker, "_env_seconds", lambda *_: 0.01)
    monkeypatch.setattr(worker, "run_once", run_once)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)

    with pytest.raises(KeyboardInterrupt):
        worker.main()

    assert calls == 2
