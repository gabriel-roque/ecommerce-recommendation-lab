from __future__ import annotations

import sys

import pytest
from sqlalchemy import func, select

from app import cli
from app.database.models import TrainingRun


def test_verify_data_writes_validation(monkeypatch, tmp_path) -> None:
    manifest = {"catalog_customers": 3, "catalog_articles": 2, "purchases": 4}
    calls = []

    def write_artifacts(dataset_dir):
        calls.append(dataset_dir)
        return {"manifest": manifest, "stats": {}}

    monkeypatch.setattr(cli, "DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(cli, "write_dataset_artifacts", write_artifacts)
    result = cli.verify_data()
    assert calls == [tmp_path / "dataset"]
    assert result["catalog_customers"] == 3
    assert (tmp_path / "processed/dataset_validation.json").is_file()


def test_automatic_training_enqueue_is_idempotent(db) -> None:
    assert cli._enqueue_auto_training("fingerprint") is True
    assert cli._enqueue_auto_training("fingerprint") is False
    assert db.scalar(select(func.count(TrainingRun.id))) == 1
    run = db.scalar(select(TrainingRun))
    assert run is not None
    assert run.config["data_profile"] == "full"
    assert run.config["epochs"] == 4
    assert run.config["source_fingerprint"] == "fingerprint"


def test_bootstrap_is_idempotent_for_same_fingerprint(monkeypatch, tmp_path) -> None:
    manifest = {"dataset": "ecommerce_sintetico_br", "source_fingerprint": "fingerprint"}
    matches = iter((False, True))
    replacements = []
    queued_fingerprints = []
    monkeypatch.setattr(cli, "write_dataset_artifacts", lambda _: {"manifest": manifest})
    monkeypatch.setattr(cli, "create_schema", lambda: None)
    monkeypatch.setattr(cli, "_database_matches", lambda _: next(matches))
    monkeypatch.setattr(cli, "_replace_dataset", replacements.append)
    monkeypatch.setattr(
        cli,
        "_enqueue_auto_training",
        lambda fingerprint: not queued_fingerprints.append(fingerprint) and len(queued_fingerprints) == 1,
    )
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path)

    first = cli.bootstrap()
    second = cli.bootstrap()

    assert first["imported"] is True
    assert second["imported"] is False
    assert replacements == ["fingerprint"]
    assert queued_fingerprints == ["fingerprint", "fingerprint"]


def test_cli_has_no_legacy_train_command(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ecommerce-lab", "train"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
