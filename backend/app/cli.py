from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.database.models import Article, Customer, DatasetState, ModelVersion, Purchase, TrainingRun
from app.database.session import create_schema, get_engine, get_session
from app.dataset import (
    DATASET_COUNTS,
    DATASET_ID,
    PostgreSQLCopySink,
    generate_dataset,
    import_synthetic,
    write_dataset_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / DATASET_ID
PROCESSED_DIR = ROOT / "data" / "processed"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_data() -> dict[str, Any]:
    result = generate_dataset(DATASET_DIR)
    print("Dataset sintético gerado: 2.000 clientes, 500 produtos e 47.500 compras")
    return result


def verify_data() -> dict[str, Any]:
    started = time.perf_counter()
    result = write_dataset_artifacts(DATASET_DIR)
    manifest = result["manifest"]
    validation = {**manifest, "validation_seconds": time.perf_counter() - started}
    _write_json(PROCESSED_DIR / "dataset_validation.json", validation)
    print(
        f"Dataset validado com {manifest['catalog_customers']:,} clientes, "
        f"{manifest['catalog_articles']:,} produtos e {manifest['purchases']:,} compras"
    )
    return validation


def _database_matches(fingerprint: str) -> bool:
    with get_session() as session:
        state = session.get(DatasetState, 1)
        if (
            state is None
            or state.dataset != DATASET_ID
            or state.source_fingerprint != fingerprint
        ):
            return False
        counts = (
            int(session.scalar(select(func.count(Customer.customer_key))) or 0),
            int(session.scalar(select(func.count(Article.article_key))) or 0),
            int(session.scalar(select(func.count(Purchase.id))) or 0),
        )
    return counts == (
        DATASET_COUNTS["customers.csv"],
        DATASET_COUNTS["products.csv"],
        DATASET_COUNTS["purchases.csv"],
    )


def _enqueue_auto_training(fingerprint: str) -> bool:
    with get_session() as session:
        runs = list(session.scalars(select(TrainingRun)))
        if any(
            isinstance(run.config, dict)
            and run.config.get("source_fingerprint") == fingerprint
            for run in runs
        ):
            return False
        for model in session.scalars(select(ModelVersion)):
            manifest_path = Path(model.artifact_path) / "dataset_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("source_fingerprint") == fingerprint:
                return False
        session.add(
            TrainingRun(
                status="QUEUED",
                config={
                    "data_profile": "full",
                    "embedding_dim": 32,
                    "dense_1": 64,
                    "dense_2": 32,
                    "batch_size": 2048,
                    "epochs": 4,
                    "negative_ratio": 2,
                    "learning_rate": 0.001,
                    "shuffle_buffer": 100_000,
                    "evaluation_negatives": 100,
                    "evaluation_batch_size": 8192,
                    "random_seed": 42,
                    "source_fingerprint": fingerprint,
                    "automatic": True,
                },
            )
        )
        try:
            session.commit()
        except IntegrityError:
            # Unique active-run constraint resolves concurrent bootstrap attempts.
            session.rollback()
            return False
    return True


def _replace_dataset(fingerprint: str) -> None:
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Bootstrap requer PostgreSQL COPY")
    raw_connection = engine.raw_connection()
    try:
        cursor = raw_connection.cursor()
        try:
            cursor.execute(
                "TRUNCATE model_versions, training_network_snapshots, training_metrics, "
                "training_runs, dataset_state, purchases, customers, articles "
                "RESTART IDENTITY CASCADE"
            )
        finally:
            cursor.close()
        import_synthetic(DATASET_DIR, PostgreSQLCopySink(raw_connection))
    finally:
        raw_connection.close()
    with get_session() as session:
        session.add(
            DatasetState(
                id=1,
                dataset=DATASET_ID,
                source_fingerprint=fingerprint,
            )
        )
        session.commit()


def bootstrap() -> dict[str, Any]:
    """Import changed data once and enqueue one full automatic training run."""
    result = write_dataset_artifacts(DATASET_DIR)
    manifest = result["manifest"]
    fingerprint = str(manifest["source_fingerprint"])
    create_schema()
    imported = False
    if not _database_matches(fingerprint):
        _replace_dataset(fingerprint)
        imported = True
    queued = _enqueue_auto_training(fingerprint)
    outcome = {
        "dataset": DATASET_ID,
        "source_fingerprint": fingerprint,
        "imported": imported,
        "automatic_training_queued": queued,
    }
    _write_json(PROCESSED_DIR / "bootstrap_result.json", outcome)
    print(json.dumps(outcome, ensure_ascii=False))
    return outcome


def optimize_database() -> None:
    create_schema()
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        raise RuntimeError("Otimização requer PostgreSQL")
    statements = (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_purchases_customer_date ON purchases (customer_key, t_dat, id)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_purchases_customer_article ON purchases (customer_key, article_key)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_purchases_article_date ON purchases (article_key, t_dat)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_training_runs_created_at ON training_runs (created_at)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_training_runs_status_created_at ON training_runs (status, created_at)",
    )
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        for statement in statements:
            connection.execute(text(statement))
        for table in ("customers", "articles", "purchases", "training_runs", "model_versions"):
            connection.execute(text(f"ANALYZE {table}"))
    print("Índices e estatísticas do banco atualizados")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comandos de dados do E-commerce Recommendation Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("generate-data")
    subparsers.add_parser("verify-data")
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("optimize-db")
    arguments = parser.parse_args()
    if arguments.command == "generate-data":
        generate_data()
    elif arguments.command == "verify-data":
        verify_data()
    elif arguments.command == "bootstrap":
        bootstrap()
    elif arguments.command == "optimize-db":
        optimize_database()


if __name__ == "__main__":
    main()
