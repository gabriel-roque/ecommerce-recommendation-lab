import json
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from app.api import routes
from app.config import Settings
from app.database.models import (
    ModelVersion,
    TrainingMetric,
    TrainingNetworkSnapshot,
    TrainingRun,
)


def test_full_dataset_is_required_by_default():
    assert Settings().require_full_dataset is True


def test_create_run_and_reject_concurrent_run(client):
    first = client.post("/training/runs", json={"epochs": 2})
    second = client.post("/training/runs", json={})

    assert first.status_code == 201
    assert first.json()["status"] == "QUEUED"
    assert first.json()["config"]["epochs"] == 2
    assert second.status_code == 409
    assert second.json()["detail"] == "Training already running"


def test_database_constraint_rejects_concurrent_runs(db):
    db.add_all(
        [
            TrainingRun(status="QUEUED", config={}),
            TrainingRun(status="TRAINING", config={}),
        ]
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_terminal_run_allows_new_run(client, db):
    db.add(
        TrainingRun(
            status="COMPLETED",
            config={},
            progress=1,
            completed_at=datetime.now(timezone.utc),
        )
    )
    db.commit()

    response = client.post("/training/runs", json={})

    assert response.status_code == 201


def test_training_config_validation(client):
    assert client.post("/training/runs", json={"epochs": 0}).status_code == 422
    assert client.post("/training/runs", json={"learning_rate": 2}).status_code == 422
    assert client.post("/training/runs", json={"evaluation_negatives": 9}).status_code == 422
    assert client.post("/training/runs", json={"batch_size": 65_537}).status_code == 422
    assert client.post("/training/runs", json={"embedding_dim": 257}).status_code == 422
    assert client.post("/training/runs", json={"probe_customer_key": 1}).status_code == 422
    assert client.post("/training/runs", json={"data_profile": "tiny"}).status_code == 422


def test_training_profiles_endpoint(client):
    response = client.get("/training/profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default"] == "balanced"
    assert {item["name"]: item["target_purchases"] for item in payload["profiles"]} == {
        "fast": 25_000,
        "balanced": 100_000,
        "extensive": 250_000,
        "full": None,
    }


def test_full_dataset_requirement_checks_manifest_and_database(client, tmp_path, monkeypatch):
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "hm",
                "semantics": "purchase_only",
                "catalog_customers": 2,
                "catalog_articles": 2,
                "purchases": 1,
                "sampled": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: Settings(require_full_dataset=True, dataset_manifest_path=str(manifest)),
    )

    response = client.post("/training/runs", json={})

    assert response.status_code == 409
    assert response.json()["detail"].startswith("Full validated dataset required:")


def test_full_dataset_runs_lightweight_preflight(client, db, tmp_path, monkeypatch):
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "hm",
                "semantics": "purchase_only",
                "catalog_customers": 2,
                "catalog_articles": 2,
                "purchases": 3,
                "sampled": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: Settings(require_full_dataset=True, dataset_manifest_path=str(manifest)),
    )
    monkeypatch.setattr(
        routes,
        "_training_dataset_counts",
        lambda db: (3, 2, 2),
    )
    calls = []
    monkeypatch.setattr(routes, "_training_preflight", lambda *args, **kwargs: calls.append(kwargs))

    response = client.post("/training/runs", json={})

    assert response.status_code == 201
    assert calls == [{"customers": 2, "articles": 2, "artifacts_dir": "artifacts"}]
    assert client.post("/training/runs", json={"evaluation_negatives": 9}).status_code == 422


def test_sse_only_returns_persisted_state_and_metrics(client, db):
    run = TrainingRun(status="COMPLETED", config={"epochs": 1}, progress=1, current_epoch=1)
    db.add(run)
    db.flush()
    db.add(
        TrainingMetric(
            training_run_id=run.id,
            phase="BENCHMARK",
            epoch=1,
            loss=0.4,
            validation_loss=0.5,
        )
    )
    db.commit()

    response = client.get(f"/training/runs/{run.id}/events")

    assert response.status_code == 200
    assert "event: completed" in response.text
    assert '"status":"COMPLETED"' in response.text
    assert "event: epoch_completed" in response.text
    assert '"loss":0.4' in response.text
    assert "heartbeat" not in response.text
    assert "epoch_completed" in response.text


def test_training_run_exposes_persisted_metrics_snapshot_hardware_and_evaluation(client, db):
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    completed_at = datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc)
    run = TrainingRun(
        status="COMPLETED",
        config={"epochs": 1, "embedding_dim": 2},
        progress=1,
        current_epoch=1,
        device="CPU",
        tensorflow_version="2.18.0",
        python_version="3.10.0",
        available_ram_bytes=2 * 1024**3,
        started_at=started_at,
        completed_at=completed_at,
    )
    db.add(run)
    db.flush()
    db.add_all(
        [
            TrainingMetric(
                training_run_id=run.id,
                phase="BENCHMARK",
                epoch=1,
                loss=0.4,
                validation_loss=0.5,
                examples_per_second=100,
            ),
            TrainingNetworkSnapshot(
                training_run_id=run.id,
                epoch=1,
                probe_customer_key=10,
                probe_article_key=20,
                activations={
                    "customer_embedding": [0.1, 0.2],
                    "article_embedding": [0.3, 0.4],
                    "customer_tower": [0.1, 0.2],
                    "article_tower": [0.3, 0.4],
                    "dense_64": [0.5],
                    "dense_32": [0.6],
                    "score": [0.7],
                },
                weights={"dense_64": [[0.1], [0.2], [0.3], [0.4]], "dense_32": [[0.5]], "score": [[0.6]]},
                prediction=0.7,
            ),
            ModelVersion(
                version="v1",
                training_run_id=run.id,
                artifact_path="artifacts/v1",
                status="READY",
                is_active=True,
                metrics={
                    "ranking": {
                        "recall_at_k": 0.8,
                        "hit_rate_at_k": 0.9,
                        "ndcg_at_k": 0.7,
                        "candidate_strategy": "sampled",
                    }
                },
            ),
        ]
    )
    db.commit()

    response = client.get(f"/training/runs/{run.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"] == [
        {
            "epoch": 1,
            "timestamp": payload["metrics"][0]["timestamp"],
            "loss": 0.4,
            "val_loss": 0.5,
            "learning_rate": None,
            "epoch_duration_seconds": None,
            "examples_per_second": 100.0,
        }
    ]
    assert payload["network_snapshot"]["probe_profile_id"] is None
    assert payload["network_snapshot"]["probe_article_id"] is None
    assert "probe_customer_key" not in payload["network_snapshot"]
    assert payload["network_snapshot"]["output_score"] == 0.7
    assert payload["network_snapshot"]["connections"]
    assert payload["hardware"] == {
        "device": "CPU",
        "ram_gb": 2.0,
        "tensorflow_version": "2.18.0",
        "python_version": "3.10.0",
    }
    assert payload["elapsed_seconds"] == 120.0
    assert payload["model_version"] == "v1"
    assert payload["evaluation"]["recall_at_10"] == 0.8
