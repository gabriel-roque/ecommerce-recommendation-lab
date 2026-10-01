from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.validation import ReportSection, ValidationReportError, generate_validation_report


def _manifest() -> dict[str, object]:
    return {
        "files": {"customers.csv": 5, "articles.csv": 4, "transactions_train.csv": 10},
        "source_files": {
            "customers.csv": {"rows": 5, "bytes": 100, "sha256": "a" * 64},
            "articles.csv": {"rows": 4, "bytes": 200, "sha256": "b" * 64},
            "transactions_train.csv": {"rows": 10, "bytes": 300, "sha256": "c" * 64},
        },
        "source_fingerprint": "d" * 64,
        "official_counts_validated": False,
        "catalog_customers": 5,
        "catalog_articles": 4,
        "interacting_customers": 3,
        "interacting_articles": 2,
        "purchases": 10,
        "date_range": {"start": "2018-09-20", "end": "2020-09-22"},
        "semantics": "purchase_only",
    }


def test_report_uses_hm_observed_results(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    stats = tmp_path / "dataset_stats.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    stats.write_text(json.dumps({"age_coverage_ratio": 0.8}), encoding="utf-8")

    output = generate_validation_report(
        tmp_path / "report.md",
        manifest_path=manifest,
        stats_path=stats,
        sections=(ReportSection("Tests", {"backend_passed": 8}, "pytest"),),
    )
    report = output.read_text(encoding="utf-8")

    assert "**Catalog Customers:** 5" in report
    assert "**Interacting Customers:** 3" in report
    assert "**Purchases:** 10" in report
    assert "**Age Coverage Ratio:** 0.8" in report


def test_report_rejects_missing_or_non_observed_results(tmp_path: Path) -> None:
    with pytest.raises(ValidationReportError):
        ReportSection("Model", {"loss": None}, "training run 1")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"purchases": 10}), encoding="utf-8")
    with pytest.raises(ValidationReportError, match="lacks measured fields"):
        generate_validation_report(tmp_path / "report.md", manifest_path=manifest)
