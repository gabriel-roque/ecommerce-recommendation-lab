from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text

from app.database.models import Article, Customer, Purchase
from app.database.session import create_schema, get_engine, get_session
from app.dataset import PostgreSQLCopySink, import_hm, write_hm_artifacts

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_data(full: bool) -> dict[str, Any]:
    started = time.perf_counter()
    result = write_hm_artifacts(RAW_DIR, PROCESSED_DIR, full=full)
    manifest = result["manifest"]
    validation = {**manifest, "validation_seconds": time.perf_counter() - started}
    _write_json(PROCESSED_DIR / "dataset_validation.json", validation)
    print(
        f"Validated H&M catalog with {manifest['catalog_customers']:,} customers, "
        f"{manifest['catalog_articles']:,} articles, and {manifest['purchases']:,} purchases"
    )
    return validation


def import_data(full: bool) -> dict[str, Any]:
    verify_data(full)
    create_schema()
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("H&M bulk import requires PostgreSQL COPY")

    raw_connection = engine.raw_connection()
    started = time.perf_counter()
    try:
        cursor = raw_connection.cursor()
        try:
            cursor.execute("TRUNCATE purchases, customers, articles RESTART IDENTITY")
        finally:
            cursor.close()
        result = import_hm(RAW_DIR, PostgreSQLCopySink(raw_connection), full=full)
    finally:
        raw_connection.close()

    observed = {
        "source_rows": result["source_rows"],
        "table_rows": result["table_rows"],
        "import_seconds": time.perf_counter() - started,
    }
    _write_json(PROCESSED_DIR / "import_result.json", observed)
    print(json.dumps(observed, indent=2))
    return observed


def optimize_database() -> None:
    create_schema()
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Database optimization requires PostgreSQL")
    statements = (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_purchases_customer_date ON purchases (customer_key, t_dat, id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_purchases_customer_article ON purchases (customer_key, article_key)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_purchases_article_date ON purchases (article_key, t_dat)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_training_runs_created_at ON training_runs (created_at)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_training_runs_status_created_at ON training_runs (status, created_at)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_training_metrics_run_phase_id ON training_metrics (training_run_id, phase, id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_training_snapshots_run_id ON training_network_snapshots (training_run_id, id)",
    )
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        for statement in statements:
            connection.execute(text(statement))
        for table in (
            "customers",
            "articles",
            "purchases",
            "training_runs",
            "training_metrics",
            "training_network_snapshots",
        ):
            connection.execute(text(f"ANALYZE {table}"))
    print("H&M database indexes and planner statistics are ready")


def _run_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "backend/tests/test_dataset_hm.py",
        "backend/tests/test_validation.py",
        "backend/tests/test_cli.py",
        "-q",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True)
    results = {"backend_data_layer": {"passed": completed.returncode == 0, "exit_code": completed.returncode}}
    _write_json(PROCESSED_DIR / "test_results.json", results)
    return results


def validate_full(*, external_tests_passed: bool = False) -> None:
    manifest = verify_data(True)
    test_results = (
        {"backend_data_layer": {"passed": True, "exit_code": 0}}
        if external_tests_passed
        else _run_tests()
    )
    _write_json(PROCESSED_DIR / "test_results.json", test_results)

    try:
        create_schema()
        with get_session() as session:
            database_result = {
                "available": True,
                "customers": int(session.scalar(select(func.count(Customer.customer_key))) or 0),
                "articles": int(session.scalar(select(func.count(Article.article_key))) or 0),
                "purchases": int(session.scalar(select(func.count(Purchase.id))) or 0),
                "interacting_customers": int(
                    session.scalar(select(func.count(func.distinct(Purchase.customer_key)))) or 0
                ),
                "interacting_articles": int(
                    session.scalar(select(func.count(func.distinct(Purchase.article_key)))) or 0
                ),
            }
        database_result["matches_manifest"] = (
            database_result["customers"] == manifest["catalog_customers"]
            and database_result["articles"] == manifest["catalog_articles"]
            and database_result["purchases"] == manifest["purchases"]
            and database_result["interacting_customers"] == manifest["interacting_customers"]
            and database_result["interacting_articles"] == manifest["interacting_articles"]
        )
    except Exception as error:
        database_result = {"available": False, "matches_manifest": False, "error": str(error)}
    _write_json(PROCESSED_DIR / "database_validation.json", database_result)

    import_path = PROCESSED_DIR / "import_result.json"
    import_validation: dict[str, Any] = {"recorded": False, "matches_manifest": False}
    if import_path.is_file():
        import_result = json.loads(import_path.read_text(encoding="utf-8"))
        table_rows = import_result.get("table_rows", {})
        import_validation = {
            "recorded": True,
            "customers": table_rows.get("customers"),
            "articles": table_rows.get("articles"),
            "purchases": table_rows.get("purchases"),
            "import_seconds": import_result.get("import_seconds"),
            "matches_manifest": (
                table_rows.get("customers") == manifest["catalog_customers"]
                and table_rows.get("articles") == manifest["catalog_articles"]
                and table_rows.get("purchases") == manifest["purchases"]
            ),
        }
    _write_json(PROCESSED_DIR / "import_validation.json", import_validation)

    failures = []
    if not all(result["passed"] for result in test_results.values()):
        failures.append("data-layer tests failed")
    if not database_result["matches_manifest"]:
        failures.append("canonical database tables do not match manifest")
    if not import_validation["matches_manifest"]:
        failures.append("canonical import result does not match manifest")
    if failures:
        raise SystemExit(f"Validation failed: {'; '.join(failures)}")
    print("Full H&M validation passed; observed results are in data/processed/*.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="H&M recommendation lab data commands")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify-data", "import-data"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--full", action="store_true")
    subparsers.add_parser("optimize-db")
    validate_parser = subparsers.add_parser("validate-full")
    validate_parser.add_argument("--external-tests-passed", action="store_true")
    arguments = parser.parse_args()

    if arguments.command == "verify-data":
        verify_data(arguments.full)
    elif arguments.command == "import-data":
        import_data(arguments.full)
    elif arguments.command == "optimize-db":
        optimize_database()
    elif arguments.command == "validate-full":
        validate_full(external_tests_passed=arguments.external_tests_passed)


if __name__ == "__main__":
    main()
