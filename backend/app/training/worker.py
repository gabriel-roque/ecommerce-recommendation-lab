"""Database-backed trainer process entry point."""

from __future__ import annotations

import logging
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sqlalchemy import func, select, update

from app.database.models import ACTIVE_RUN_STATUSES, Article, Customer, ModelVersion, Purchase, TrainingRun
from app.database.session import create_schema, get_session
from app.config import get_settings
from app.models.features import ARTICLE_FEATURES, CUSTOMER_FEATURES

from .config import DATA_PROFILES, TrainingConfig
from .database import DatabaseTrainingObserver
from .hardware import hardware_info
from .pipeline import TrainingPipeline
from .sampling import (
    SampledPurchases,
    load_cached_sample,
    sample_customer_histories,
    write_cached_sample,
)


logger = logging.getLogger(__name__)
RUNNING_STATUSES = tuple(status for status in ACTIVE_RUN_STATUSES if status != "QUEUED")


PurchaseColumns = SampledPurchases


def _claim_run(worker_id: str) -> tuple[str, dict[str, Any]] | None:
    with get_session() as session:
        run = session.scalar(
            select(TrainingRun)
            .where(TrainingRun.status == "QUEUED")
            .order_by(TrainingRun.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if run is None:
            return None
        run.status = "PREPARING_DATA"
        run.worker_id = worker_id
        run.started_at = datetime.now(timezone.utc)
        run.heartbeat_at = run.started_at
        session.commit()
        return run.id, dict(run.config)


def _load_purchases() -> PurchaseColumns:
    timestamp_column = Purchase.t_dat
    with get_session() as session:
        count = int(session.scalar(select(func.count(Purchase.id))) or 0)
        if count == 0:
            raise ValueError("O banco não contém compras")
        columns = {
            "timestamp": np.empty(count, dtype=np.int64),
            "customer_key": np.empty(count, dtype=np.int64),
            "article_key": np.empty(count, dtype=np.int64),
            "price": np.empty(count, dtype=np.float32),
            "sales_channel_id": np.empty(count, dtype=np.int16),
        }
        rows = session.execute(
            select(
                timestamp_column.label("timestamp"),
                Purchase.customer_key,
                Purchase.article_key,
                Purchase.price,
                Purchase.sales_channel_id,
            )
            .execution_options(yield_per=100_000)
        )
        actual = 0
        for actual, row in enumerate(rows, start=1):
            index = actual - 1
            timestamp = row.timestamp
            if isinstance(timestamp, datetime):
                timestamp = int(timestamp.replace(tzinfo=timestamp.tzinfo or timezone.utc).timestamp() * 1000)
            elif isinstance(timestamp, date):
                timestamp = int(
                    datetime.combine(timestamp, datetime.min.time(), tzinfo=timezone.utc).timestamp()
                    * 1000
                )
            columns["timestamp"][index] = timestamp
            columns["customer_key"][index] = row.customer_key
            columns["article_key"][index] = row.article_key
            columns["price"][index] = row.price
            columns["sales_channel_id"][index] = row.sales_channel_id or 0
        if actual != count:
            raise RuntimeError(f"Purchase count changed during load: expected {count}, read {actual}")
        return PurchaseColumns(columns)


def _load_sampled_purchase_pool(config: TrainingConfig, catalog_customers: int) -> PurchaseColumns:
    target = DATA_PROFILES[config.data_profile]["target_purchases"]
    if target is None:
        return _load_purchases()
    rng = np.random.default_rng(config.random_seed)
    customer_keys = rng.permutation(np.arange(1, catalog_customers + 1, dtype=np.int64))
    values: dict[str, list[Any]] = {
        "timestamp": [],
        "customer_key": [],
        "article_key": [],
        "price": [],
        "sales_channel_id": [],
    }
    with get_session() as session:
        for start in range(0, customer_keys.size, 1_000):
            batch = customer_keys[start : start + 1_000].tolist()
            rows = session.execute(
                select(
                    Purchase.t_dat,
                    Purchase.customer_key,
                    Purchase.article_key,
                    Purchase.price,
                    Purchase.sales_channel_id,
                ).where(Purchase.customer_key.in_(batch))
            )
            for purchased_on, customer_key, article_key, price, sales_channel_id in rows:
                values["timestamp"].append(
                    int(
                        datetime.combine(
                            purchased_on,
                            datetime.min.time(),
                            tzinfo=timezone.utc,
                        ).timestamp()
                        * 1000
                    )
                )
                values["customer_key"].append(customer_key)
                values["article_key"].append(article_key)
                values["price"].append(price)
                values["sales_channel_id"].append(sales_channel_id or 0)
            if len(values["timestamp"]) >= target * 2:
                break
    if not values["timestamp"]:
        raise ValueError("O banco não contém compras para os clientes amostrados")
    return PurchaseColumns(
        {
            "timestamp": np.asarray(values["timestamp"], dtype=np.int64),
            "customer_key": np.asarray(values["customer_key"], dtype=np.int64),
            "article_key": np.asarray(values["article_key"], dtype=np.int64),
            "price": np.asarray(values["price"], dtype=np.float32),
            "sales_channel_id": np.asarray(values["sales_channel_id"], dtype=np.int16),
        }
    )


def _load_feature_rows(
    model: Any,
    key_name: str,
    names: tuple[str, ...],
    keys: np.ndarray,
) -> list[dict[str, Any]]:
    columns = [getattr(model, key_name)]
    for name in names:
        columns.append(getattr(model, name).label(name))
    unique_keys = np.unique(np.asarray(keys, dtype=np.int64))
    rows: list[dict[str, Any]] = []
    with get_session() as session:
        for start in range(0, unique_keys.size, 10_000):
            batch = unique_keys[start : start + 10_000].tolist()
            rows.extend(
                dict(row._mapping)
                for row in session.execute(
                    select(*columns).where(getattr(model, key_name).in_(batch))
                )
            )
    if len(rows) != unique_keys.size:
        raise ValueError(f"Algumas compras têm valores {key_name} sem metadados")
    return rows


def _manifest(*, validate_database: bool = True) -> dict[str, Any]:
    manifest_path = Path(get_settings().dataset_manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        not isinstance(manifest, dict)
        or manifest.get("dataset") != "ecommerce_sintetico_br"
        or manifest.get("semantics") != "purchase_only"
    ):
        raise ValueError("Manifesto deve identificar ecommerce_sintetico_br com compras")
    if validate_database:
        with get_session() as session:
            purchase_count = int(session.scalar(select(func.count(Purchase.id))) or 0)
            customer_count = int(session.scalar(select(func.count(Customer.customer_key))) or 0)
            article_count = int(session.scalar(select(func.count(Article.article_key))) or 0)
        database_counts = {
            "purchases": purchase_count,
            "catalog_customers": customer_count,
            "catalog_articles": article_count,
        }
        mismatches = [name for name, value in database_counts.items() if manifest.get(name) != value]
        if mismatches:
            raise ValueError(f"Database does not match dataset manifest: {', '.join(mismatches)}")
    return manifest


def _training_purchases(
    config: TrainingConfig,
) -> tuple[PurchaseColumns, dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    source_manifest = _manifest(validate_database=False)
    if config.data_profile != "full":
        cached = load_cached_sample(
            settings.training_samples_dir,
            profile=config.data_profile,
            seed=config.random_seed,
            source_manifest=source_manifest,
        )
        if cached is not None:
            return cached[0], source_manifest, cached[1]

    source_manifest = _manifest(validate_database=True)
    purchase_pool = (
        _load_purchases()
        if config.data_profile == "full"
        else _load_sampled_purchase_pool(config, int(source_manifest["catalog_customers"]))
    )
    purchases, training_manifest = sample_customer_histories(
        purchase_pool,
        profile=config.data_profile,
        seed=config.random_seed,
        source_manifest=source_manifest,
    )
    if config.data_profile != "full":
        write_cached_sample(
            settings.training_samples_dir,
            purchases,
            training_manifest,
            source_manifest=source_manifest,
        )
    return purchases, source_manifest, training_manifest


def _finalize_run(
    run_id: str,
    version: str,
    artifact_path: Path,
    config: TrainingConfig,
    metrics: dict[str, Any],
    training_manifest: dict[str, Any] | None = None,
) -> None:
    with get_session() as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise LookupError(f"Training run not found: {run_id}")
        if run.status not in RUNNING_STATUSES:
            raise RuntimeError(f"Cannot finalize training run in status {run.status}")
        for current in session.scalars(select(ModelVersion).where(ModelVersion.is_active.is_(True))):
            current.is_active = False
            current.status = "ARCHIVED"
        session.flush()
        now = datetime.now(timezone.utc)
        session.add(
            ModelVersion(
                version=version,
                training_run_id=run_id,
                artifact_path=str(artifact_path),
                status="READY",
                is_active=True,
                config={
                    **config.to_dict(),
                    **({"data_manifest": training_manifest} if training_manifest else {}),
                },
                metrics=metrics,
                activated_at=now,
            )
        )
        run.status = "COMPLETED"
        run.progress = 1.0
        run.completed_at = now
        run.heartbeat_at = now
        run.updated_at = now
        session.commit()


def _fail_abandoned_runs(timeout_seconds: float) -> int:
    if timeout_seconds <= 0:
        raise ValueError("Heartbeat timeout must be positive")
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=timeout_seconds)
    last_activity = func.coalesce(
        TrainingRun.heartbeat_at,
        TrainingRun.started_at,
        TrainingRun.updated_at,
        TrainingRun.created_at,
    )
    with get_session() as session:
        result = session.execute(
            update(TrainingRun)
            .where(
                TrainingRun.status.in_(RUNNING_STATUSES),
                last_activity < cutoff,
            )
            .values(
                status="FAILED",
                error_message=f"Worker heartbeat timed out after {timeout_seconds:g} seconds",
                completed_at=now,
                updated_at=now,
            )
        )
        session.commit()
        return int(result.rowcount or 0)


def _touch_heartbeat(run_id: str, worker_id: str) -> None:
    now = datetime.now(timezone.utc)
    with get_session() as session:
        session.execute(
            update(TrainingRun)
            .where(
                TrainingRun.id == run_id,
                TrainingRun.worker_id == worker_id,
                TrainingRun.status.in_(RUNNING_STATUSES),
            )
            .values(heartbeat_at=now, updated_at=now)
        )
        session.commit()


@contextmanager
def _heartbeat(run_id: str, worker_id: str, interval_seconds: float) -> Iterator[None]:
    if interval_seconds <= 0:
        raise ValueError("Heartbeat interval must be positive")
    stopped = threading.Event()

    def update_heartbeat() -> None:
        while not stopped.wait(interval_seconds):
            try:
                _touch_heartbeat(run_id, worker_id)
            except Exception:
                logger.exception("Failed to update heartbeat for training run %s", run_id)

    thread = threading.Thread(target=update_heartbeat, name=f"heartbeat-{run_id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _env_seconds(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def run_once(
    *,
    artifact_root: str | Path | None = None,
    worker_id: str | None = None,
    heartbeat_interval_seconds: float | None = None,
    heartbeat_timeout_seconds: float | None = None,
) -> bool:
    heartbeat_interval = (
        heartbeat_interval_seconds
        if heartbeat_interval_seconds is not None
        else _env_seconds("TRAINER_HEARTBEAT_SECONDS", 10)
    )
    heartbeat_timeout = (
        heartbeat_timeout_seconds
        if heartbeat_timeout_seconds is not None
        else _env_seconds("TRAINER_HEARTBEAT_TIMEOUT_SECONDS", 120)
    )
    if heartbeat_interval <= 0 or heartbeat_timeout <= 0:
        raise ValueError("Heartbeat interval and timeout must be positive")
    if heartbeat_interval >= heartbeat_timeout:
        raise ValueError("Heartbeat interval must be shorter than heartbeat timeout")
    _fail_abandoned_runs(heartbeat_timeout)
    identity = worker_id or f"{socket.gethostname()}:{os.getpid()}"
    claimed = _claim_run(identity)
    if claimed is None:
        return False
    run_id, values = claimed
    observer: DatabaseTrainingObserver | None = None
    try:
        config = TrainingConfig.from_mapping(values)
        observer = DatabaseTrainingObserver(run_id, config.epochs)
        version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{run_id[:8]}"
        root = Path(artifact_root or os.getenv("ARTIFACTS_DIR", "artifacts"))
        with _heartbeat(run_id, identity, heartbeat_interval):
            device = hardware_info()
            with get_session() as session:
                run = session.get(TrainingRun, run_id)
                if run is not None:
                    run.device = device["device"]
                    run.tensorflow_version = device["tensorflow_version"]
                    run.python_version = device["python_version"]
                    run.available_ram_bytes = device["available_ram_bytes"]
                    session.commit()
            purchases, source_manifest, training_manifest = _training_purchases(config)
            customers = _load_feature_rows(
                Customer,
                "customer_key",
                CUSTOMER_FEATURES,
                purchases["customer_key"],
            )
            articles = _load_feature_rows(
                Article,
                "article_key",
                ARTICLE_FEATURES,
                purchases["article_key"],
            )
            with get_session() as session:
                run = session.get(TrainingRun, run_id)
                if run is not None:
                    run.config = {
                        **config.to_dict(),
                        "data_manifest": training_manifest,
                        "source_fingerprint": source_manifest["source_fingerprint"],
                        "automatic": bool(values.get("automatic", False)),
                    }
                    session.commit()
            outcome = TrainingPipeline(root, config, observer).run(
                purchases,
                customers=customers,
                articles=articles,
                model_version=version,
                dataset_manifest=source_manifest,
                training_manifest=training_manifest,
                run_id=run_id,
            )
            metrics = {
                "ranking": outcome.ranking.to_dict(),
                "benchmark_history": outcome.benchmark_history,
                "durations": outcome.durations,
                "data_scope": training_manifest,
            }
            _finalize_run(
                run_id,
                version,
                outcome.artifact.directory,
                config,
                metrics,
                training_manifest,
            )
    except BaseException as error:
        (observer or DatabaseTrainingObserver(run_id, 1)).fail(error)
        raise
    return True


def _wait_for_database(retry_seconds: float) -> None:
    while True:
        try:
            create_schema()
            return
        except Exception:
            logger.exception("Database unavailable; retrying in %s seconds", retry_seconds)
            time.sleep(retry_seconds)


def main() -> None:
    poll_seconds = _env_seconds("TRAINER_POLL_SECONDS", 2)
    database_retry_seconds = _env_seconds("TRAINER_DATABASE_RETRY_SECONDS", 2)
    _wait_for_database(database_retry_seconds)
    while True:
        try:
            worked = run_once()
        except Exception:
            logger.exception("Training run failed")
            worked = False
        if not worked:
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
