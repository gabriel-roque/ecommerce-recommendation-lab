from __future__ import annotations

import sys

import pytest

from app import cli


def test_verify_data_passes_full_mode_and_writes_validation(monkeypatch, tmp_path) -> None:
    manifest = {
        "catalog_customers": 3,
        "catalog_articles": 2,
        "purchases": 4,
    }
    calls: list[bool] = []

    def write_artifacts(raw_dir, processed_dir, *, full: bool):
        calls.append(full)
        return {"manifest": manifest, "stats": {}}

    monkeypatch.setattr(cli, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cli, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(cli, "write_hm_artifacts", write_artifacts)

    result = cli.verify_data(True)

    assert calls == [True]
    assert result["catalog_customers"] == 3
    assert (tmp_path / "processed" / "dataset_validation.json").is_file()


def test_cli_has_no_legacy_train_command(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hm-lab", "train"])

    with pytest.raises(SystemExit) as error:
        cli.main()

    assert error.value.code == 2
