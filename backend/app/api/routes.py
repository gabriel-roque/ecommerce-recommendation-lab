from __future__ import annotations

import asyncio
import json
import math
import secrets
import shutil
import time
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import desc, func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.recommendation_backend import (
    RecommendationBackend,
    RecommendationUnavailable,
    get_recommendation_backend,
)
from app.api.schemas import (
    ArticleResponse,
    CustomerHistoryItem,
    CustomerHistoryResponse,
    CustomerResponse,
    DatasetStatsResponse,
    EvaluationResponse,
    HardwareInfoResponse,
    HealthResponse,
    ModelMetadataResponse,
    ModelVersionResponse,
    NetworkConnectionResponse,
    NetworkSnapshotResponse,
    RecommendationItem,
    RecommendationResponse,
    RecommendationEvidence,
    ScoreSemanticsResponse,
    TrainingConfig,
    TrainingMetricResponse,
    TrainingProfileResponse,
    TrainingProfilesResponse,
    TrainingRunListResponse,
    TrainingRunResponse,
)
from app.config import get_settings
from app.database.models import (
    ACTIVE_RUN_STATUSES,
    Article,
    Customer,
    ModelVersion,
    Purchase,
    TrainingMetric,
    TrainingNetworkSnapshot,
    TrainingRun,
)
from app.database.session import get_db, get_session
from app.training.config import DATA_PROFILES

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]
ProfileId = Annotated[str, Path(min_length=1, max_length=64)]
PUBLIC_TRAINING_CONFIG_FIELDS = frozenset(TrainingConfig.model_fields)


def _age_band_column() -> Any:
    return Customer.age_band


def _public_training_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in config.items() if key in PUBLIC_TRAINING_CONFIG_FIELDS
    }


def _load_dataset_manifest(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with FilePath(path).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Dataset manifest unavailable: {exc}"

    required_counts = ("catalog_customers", "catalog_articles", "purchases")
    if not isinstance(manifest, dict):
        return None, "Dataset manifest must be a JSON object"
    if type(manifest.get("sampled")) is not bool:
        return None, "Dataset manifest sampled flag is invalid"
    if manifest.get("dataset") != "hm" or manifest.get("semantics") != "purchase_only":
        return None, "Dataset manifest is not H&M"
    for name in required_counts:
        value = manifest.get(name)
        if type(value) is not int or value < 0:
            return None, f"Dataset manifest {name} count is invalid"
    return manifest, None


def _dataset_validation(
    *,
    manifest_path: str,
    purchases: int,
    customers: int,
    articles: int,
) -> tuple[bool, str | None]:
    manifest, error = _load_dataset_manifest(manifest_path)
    if manifest is None:
        return False, error
    if manifest["sampled"]:
        return False, "Dataset manifest identifies a sampled dataset"

    actual = {"purchases": purchases, "catalog_customers": customers, "catalog_articles": articles}
    expected = {
        "purchases": manifest["purchases"],
        "catalog_customers": manifest["catalog_customers"],
        "catalog_articles": manifest["catalog_articles"],
    }
    mismatches = [name for name, value in expected.items() if actual.get(name, 0) != value]
    if mismatches:
        return False, f"Database does not match dataset manifest: {', '.join(mismatches)}"
    return True, None


def _training_dataset_counts(db: Session) -> tuple[int, int, int]:
    purchases = int(db.scalar(select(func.count(Purchase.id))) or 0)
    customers = int(db.scalar(select(func.count(Customer.customer_key))) or 0)
    articles = int(db.scalar(select(func.count(Article.article_key))) or 0)
    return purchases, customers, articles


def _training_preflight(
    db: Session,
    config: TrainingConfig,
    *,
    customers: int,
    articles: int,
    artifacts_dir: str,
) -> None:
    failures: list[str] = []
    if customers < 1 or articles < 1:
        failures.append("dataset cannot produce customer and article mappings")
    if max(customers, articles) > 2_147_483_647:
        failures.append("mapping indices exceed TensorFlow int32 capacity")
    artifact_path = FilePath(artifacts_dir).expanduser()
    disk_path = artifact_path
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    try:
        free_bytes = shutil.disk_usage(disk_path).free
        parameters = (
            (customers + articles) * config.embedding_dim
            + 2 * config.embedding_dim * config.dense_1
            + config.dense_1 * config.dense_2
            + config.dense_1
            + 2 * config.dense_2
            + 1
        )
        if parameters > 100_000_000:
            failures.append(
                f"model shape requires {parameters} parameters; limit is 100000000"
            )
        required_bytes = max(64 * 1024**2, parameters * 12 + (customers + articles) * 16)
        if free_bytes < required_bytes:
            failures.append(
                f"artifact storage needs {required_bytes} bytes; {free_bytes} bytes available"
            )
    except OSError as exc:
        failures.append(f"artifact storage is unavailable: {exc}")

    if failures:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Training preflight failed: {'; '.join(failures)}",
        )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(db: DbSession) -> HealthResponse | JSONResponse:
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "database": "unavailable"},
        )
    return HealthResponse(status="ok", database="ok")


@router.get("/dataset/stats", response_model=DatasetStatsResponse, tags=["dataset"])
def dataset_stats(db: DbSession) -> DatasetStatsResponse:
    settings = get_settings()
    manifest, _ = _load_dataset_manifest(settings.dataset_manifest_path)
    try:
        cached_stats = json.loads(FilePath(settings.dataset_stats_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached_stats = None
    if (
        settings.require_full_dataset
        and manifest
        and manifest["sampled"] is False
        and isinstance(cached_stats, dict)
        and cached_stats.get("source_fingerprint") == manifest.get("source_fingerprint")
    ):
        total_purchases = manifest["purchases"]
        total_customers = manifest["catalog_customers"]
        total_articles = manifest["catalog_articles"]
        return DatasetStatsResponse(
            total_purchases=total_purchases,
            unique_customers=total_customers,
            customers_with_purchases=manifest["interacting_customers"],
            unique_articles=total_articles,
            purchased_articles=manifest["interacting_articles"],
            average_purchases_per_customer=total_purchases / total_customers,
            median_purchases_per_customer=cached_stats["median_purchases_per_customer"],
            average_purchases_per_article=total_purchases / total_articles,
            date_range=manifest["date_range"],
            purchases_per_customer=cached_stats["purchases_per_customer"],
            age_band_counts=cached_stats["age_band_counts"],
            top_articles=cached_stats["top_articles"][:10],
            sampled=False,
        )
    total_customers = int(db.scalar(select(func.count(Customer.customer_key))) or 0)
    total_articles = int(db.scalar(select(func.count(Article.article_key))) or 0)
    counts = db.execute(
        select(
            func.count(Purchase.id),
            func.count(func.distinct(Purchase.customer_key)),
            func.count(func.distinct(Purchase.article_key)),
            func.min(Purchase.t_dat),
            func.max(Purchase.t_dat),
        )
    ).one()
    total_purchases = int(counts[0] or 0)
    customers_with_purchases = int(counts[1] or 0)
    purchased_articles = int(counts[2] or 0)
    purchase_counts = sorted(
        int(value)
        for value in db.scalars(
            select(func.count(Purchase.id)).group_by(Purchase.customer_key)
        )
    )
    purchase_counts = [0] * max(0, total_customers - len(purchase_counts)) + purchase_counts
    middle = len(purchase_counts) // 2
    if not purchase_counts:
        median = 0.0
    elif len(purchase_counts) % 2:
        median = float(purchase_counts[middle])
    else:
        median = (purchase_counts[middle - 1] + purchase_counts[middle]) / 2
    histogram = {
        "0": sum(value == 0 for value in purchase_counts),
        "1": sum(value == 1 for value in purchase_counts),
        "2-5": sum(2 <= value <= 5 for value in purchase_counts),
        "6-10": sum(6 <= value <= 10 for value in purchase_counts),
        "11-50": sum(11 <= value <= 50 for value in purchase_counts),
        "51-100": sum(51 <= value <= 100 for value in purchase_counts),
        "101+": sum(value >= 101 for value in purchase_counts),
    }
    age_column = _age_band_column()
    age_band_counts = {
        str(age_band or "unknown"): int(count)
        for age_band, count in db.execute(
            select(age_column, func.count(Customer.customer_key)).group_by(age_column)
        )
    }
    top_articles = [
        {"article_id": str(article_id), "purchases": int(purchases)}
        for article_id, purchases in db.execute(
            select(Article.article_id, func.count(Purchase.id).label("purchases"))
            .join(Purchase, Purchase.article_key == Article.article_key)
            .group_by(Article.article_id)
            .order_by(desc("purchases"), Article.article_id)
            .limit(10)
        )
    ]
    manifest_matches = bool(
        manifest
        and manifest["purchases"] == total_purchases
        and manifest["catalog_customers"] == total_customers
        and manifest["catalog_articles"] == total_articles
    )
    return DatasetStatsResponse(
        total_purchases=total_purchases,
        unique_customers=total_customers,
        customers_with_purchases=customers_with_purchases,
        unique_articles=total_articles,
        purchased_articles=purchased_articles,
        average_purchases_per_customer=(
            total_purchases / total_customers if total_customers else 0.0
        ),
        median_purchases_per_customer=median,
        average_purchases_per_article=(
            total_purchases / total_articles if total_articles else 0.0
        ),
        date_range={"start": counts[3], "end": counts[4]},
        purchases_per_customer=histogram,
        age_band_counts=age_band_counts,
        top_articles=top_articles,
        sampled=not manifest_matches or bool(manifest and manifest["sampled"]),
    )


@router.get("/training/profiles", response_model=TrainingProfilesResponse, tags=["training"])
def training_profiles() -> TrainingProfilesResponse:
    return TrainingProfilesResponse(
        profiles=[
            TrainingProfileResponse(
                name=name,
                label=str(metadata["label"]),
                description=str(metadata["description"]),
                target_purchases=metadata["target_purchases"],
            )
            for name, metadata in DATA_PROFILES.items()
        ]
    )


def _snapshot_response(
    db: Session, snapshot: TrainingNetworkSnapshot | None, config: dict[str, Any]
) -> NetworkSnapshotResponse | None:
    if snapshot is None or snapshot.prediction is None:
        return None
    activations = snapshot.activations if isinstance(snapshot.activations, dict) else {}
    weights = snapshot.weights if isinstance(snapshot.weights, dict) else {}
    connections: list[NetworkConnectionResponse] = []
    embedding_dim = int(config.get("embedding_dim", 0))

    def add_connections(
        matrix_name: str, source_layer: str, target_layer: str, source_offset: int = 0
    ) -> None:
        matrix = weights.get(matrix_name, [])
        if not isinstance(matrix, list):
            return
        for row_index, row in enumerate(matrix):
            if not isinstance(row, list):
                continue
            for target_index, weight in enumerate(row):
                connections.append(
                    NetworkConnectionResponse(
                        source_layer=source_layer,
                        source_index=row_index - source_offset,
                        target_layer=target_layer,
                        target_index=target_index,
                        weight=float(weight),
                    )
                )

    dense_64_weights = weights.get("dense_64", [])
    if isinstance(dense_64_weights, list):
        customer_rows = dense_64_weights[:embedding_dim]
        article_rows = dense_64_weights[embedding_dim:]
        weights = {**weights, "dense_64_customer": customer_rows, "dense_64_article": article_rows}
        add_connections("dense_64_customer", "profile_embedding", "dense_64")
        add_connections("dense_64_article", "article_embedding", "dense_64")
    add_connections("dense_32", "dense_64", "dense_32")
    add_connections("score", "dense_32", "output")

    def activation(name: str) -> list[float]:
        values = activations.get(name, [])
        return [float(value) for value in values] if isinstance(values, list) else []

    customer = (
        db.get(Customer, snapshot.probe_customer_key)
        if snapshot.probe_customer_key is not None
        else None
    )
    article = (
        db.get(Article, snapshot.probe_article_key)
        if snapshot.probe_article_key is not None
        else None
    )
    return NetworkSnapshotResponse(
        epoch=snapshot.epoch,
        timestamp=snapshot.created_at,
        probe_profile_id=(_profile_id(customer) if customer is not None else None),
        probe_article_id=(str(article.article_id) if article is not None else None),
        profile_embedding=activation("customer_tower"),
        article_embedding=activation("article_tower"),
        dense_64=activation("dense_64"),
        dense_32=activation("dense_32"),
        output_score=float(snapshot.prediction),
        connections=connections,
    )


def _elapsed_seconds(run: TrainingRun) -> float | None:
    if run.started_at is None:
        return None
    started_at = run.started_at
    ended_at = run.completed_at or datetime.now(timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if ended_at.tzinfo is None:
        ended_at = ended_at.replace(tzinfo=timezone.utc)
    return max(0.0, (ended_at - started_at).total_seconds())


_UNSET = object()


def _training_run_response(
    db: Session,
    run: TrainingRun,
    *,
    prefetched_metrics: list[TrainingMetric] | None = None,
    prefetched_snapshot: TrainingNetworkSnapshot | None | object = _UNSET,
    prefetched_model: ModelVersion | None | object = _UNSET,
) -> TrainingRunResponse:
    metrics = prefetched_metrics
    if metrics is None:
        metrics = list(
            db.scalars(
                select(TrainingMetric)
                .where(
                    TrainingMetric.training_run_id == run.id,
                    TrainingMetric.phase == "BENCHMARK",
                )
                .order_by(TrainingMetric.epoch)
            )
        )
    snapshot = prefetched_snapshot
    if snapshot is _UNSET:
        snapshot = db.scalar(
            select(TrainingNetworkSnapshot)
            .where(TrainingNetworkSnapshot.training_run_id == run.id)
            .order_by(desc(TrainingNetworkSnapshot.epoch))
            .limit(1)
        )
    model = prefetched_model
    if model is _UNSET:
        model = db.scalar(
            select(ModelVersion)
            .where(ModelVersion.training_run_id == run.id)
            .order_by(desc(ModelVersion.created_at))
            .limit(1)
        )
    assert snapshot is None or isinstance(snapshot, TrainingNetworkSnapshot)
    assert model is None or isinstance(model, ModelVersion)
    ranking = model.metrics.get("ranking") if model and isinstance(model.metrics, dict) else None
    evaluation = None
    if isinstance(ranking, dict) and all(
        ranking.get(name) is not None
        for name in ("recall_at_k", "hit_rate_at_k", "ndcg_at_k", "candidate_strategy")
    ):
        evaluation = EvaluationResponse(
            recall_at_10=float(ranking["recall_at_k"]),
            hit_rate_at_10=float(ranking["hit_rate_at_k"]),
            ndcg_at_10=float(ranking["ndcg_at_k"]),
            candidate_strategy=str(ranking["candidate_strategy"]),
        )
    hardware = None
    if run.device is not None:
        hardware = HardwareInfoResponse(
            device=run.device,
            ram_gb=(run.available_ram_bytes / 1024**3 if run.available_ram_bytes is not None else None),
            tensorflow_version=run.tensorflow_version,
            python_version=run.python_version,
        )
    return TrainingRunResponse(
        id=run.id,
        status=run.status,
        config=_public_training_config(run.config),
        current_epoch=run.current_epoch,
        progress=run.progress,
        device=run.device,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        updated_at=run.updated_at,
        metrics=[
            TrainingMetricResponse(
                epoch=metric.epoch,
                timestamp=metric.created_at,
                loss=metric.loss,
                val_loss=metric.validation_loss,
                learning_rate=metric.learning_rate,
                epoch_duration_seconds=metric.epoch_duration_seconds,
                examples_per_second=metric.examples_per_second,
            )
            for metric in metrics
        ],
        network_snapshot=_snapshot_response(db, snapshot, run.config),
        hardware=hardware,
        elapsed_seconds=_elapsed_seconds(run),
        model_version=model.version if model else None,
        evaluation=evaluation,
    )


@router.post(
    "/training/runs",
    response_model=TrainingRunResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["training"],
)
def create_training_run(config: TrainingConfig, db: DbSession) -> TrainingRunResponse:
    existing = db.scalar(
        select(TrainingRun.id).where(TrainingRun.status.in_(ACTIVE_RUN_STATUSES)).limit(1)
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Training already running")

    settings = get_settings()
    if settings.require_full_dataset:
        purchases, customers, articles = _training_dataset_counts(db)
        dataset_is_full, reason = _dataset_validation(
            manifest_path=settings.dataset_manifest_path,
            purchases=purchases,
            customers=customers,
            articles=articles,
        )
        if not dataset_is_full:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Full validated dataset required: {reason}",
            )
        _training_preflight(
            db,
            config,
            customers=customers,
            articles=articles,
            artifacts_dir=settings.artifacts_dir,
        )

    training_run = TrainingRun(status="QUEUED", config=config.model_dump(mode="json"))
    db.add(training_run)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Training already running") from exc
    db.refresh(training_run)
    return _training_run_response(db, training_run)


@router.get("/training/runs", response_model=TrainingRunListResponse, tags=["training"])
def list_training_runs(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TrainingRunListResponse:
    total = int(db.scalar(select(func.count(TrainingRun.id))) or 0)
    runs = list(
        db.scalars(
            select(TrainingRun).order_by(desc(TrainingRun.created_at)).offset(offset).limit(limit)
        )
    )
    run_ids = [run.id for run in runs]
    metrics_by_run: dict[str, list[TrainingMetric]] = {run_id: [] for run_id in run_ids}
    snapshots_by_run: dict[str, TrainingNetworkSnapshot] = {}
    models_by_run: dict[str, ModelVersion] = {}
    if run_ids:
        for metric in db.scalars(
            select(TrainingMetric)
            .where(
                TrainingMetric.training_run_id.in_(run_ids),
                TrainingMetric.phase == "BENCHMARK",
            )
            .order_by(TrainingMetric.training_run_id, TrainingMetric.epoch)
        ):
            metrics_by_run[metric.training_run_id].append(metric)
        for snapshot in db.scalars(
            select(TrainingNetworkSnapshot)
            .where(TrainingNetworkSnapshot.training_run_id.in_(run_ids))
            .order_by(
                TrainingNetworkSnapshot.training_run_id,
                desc(TrainingNetworkSnapshot.epoch),
            )
        ):
            snapshots_by_run.setdefault(snapshot.training_run_id, snapshot)
        for model in db.scalars(
            select(ModelVersion)
            .where(ModelVersion.training_run_id.in_(run_ids))
            .order_by(ModelVersion.training_run_id, desc(ModelVersion.created_at))
        ):
            models_by_run.setdefault(model.training_run_id, model)
    return TrainingRunListResponse(
        total=total,
        items=[
            _training_run_response(
                db,
                run,
                prefetched_metrics=metrics_by_run[run.id],
                prefetched_snapshot=snapshots_by_run.get(run.id),
                prefetched_model=models_by_run.get(run.id),
            )
            for run in runs
        ],
    )


@router.get("/training/runs/{run_id}", response_model=TrainingRunResponse, tags=["training"])
def get_training_run(run_id: str, db: DbSession) -> TrainingRunResponse:
    training_run = db.get(TrainingRun, run_id)
    if training_run is None:
        raise HTTPException(status_code=404, detail="Training run not found")
    return _training_run_response(db, training_run)


def _sse(event: str, data: dict[str, Any], event_id: str) -> str:
    return f"id: {event_id}\nevent: {event}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


@router.get("/training/runs/{run_id}/events", tags=["training"])
async def training_events(run_id: str, request: Request) -> StreamingResponse:
    with get_session() as db:
        if db.get(TrainingRun, run_id) is None:
            raise HTTPException(status_code=404, detail="Training run not found")
    poll_interval = get_settings().sse_poll_interval_seconds
    last_event_id = request.headers.get("last-event-id", "")

    async def persisted_events() -> AsyncIterator[str]:
        last_metric_id = 0
        last_snapshot_id = 0
        terminal_seen = last_event_id.startswith(("completed-", "failed-"))
        if last_event_id.startswith("metric-"):
            try:
                last_metric_id = int(last_event_id.removeprefix("metric-"))
            except ValueError:
                pass
        elif last_event_id.startswith("snapshot-"):
            try:
                last_snapshot_id = int(last_event_id.removeprefix("snapshot-"))
            except ValueError:
                pass
        last_status: str | None = None
        if last_event_id:
            with get_session() as initial_db:
                initial_run = initial_db.get(TrainingRun, run_id)
                last_status = initial_run.status if initial_run is not None else None

        while not await request.is_disconnected():
            messages: list[str] = []
            terminal = False
            with get_session() as event_db:
                run = event_db.get(TrainingRun, run_id)
                if run is None:
                    return
                if terminal_seen and run.status in {"COMPLETED", "FAILED"}:
                    return

                if run.status != last_status and run.status not in {"COMPLETED", "FAILED"}:
                    response = _training_run_response(event_db, run)
                    messages.append(
                        _sse(
                            "status_changed",
                            {
                                "type": "status_changed",
                                "run_id": run.id,
                                "status": run.status,
                                "timestamp": run.updated_at.isoformat(),
                                "current_epoch": run.current_epoch,
                                "progress": run.progress,
                                "elapsed_seconds": response.elapsed_seconds,
                                "hardware": (
                                    response.hardware.model_dump(mode="json")
                                    if response.hardware is not None
                                    else None
                                ),
                                "error_message": run.error_message,
                            },
                            f"status-{run.updated_at.isoformat()}",
                        )
                    )
                    last_status = run.status

                metrics = list(
                    event_db.scalars(
                        select(TrainingMetric)
                        .where(
                            TrainingMetric.training_run_id == run_id,
                            TrainingMetric.phase == "BENCHMARK",
                            TrainingMetric.id > last_metric_id,
                        )
                        .order_by(TrainingMetric.id)
                    )
                )
                for metric in metrics:
                    messages.append(
                        _sse(
                            "epoch_completed",
                            {
                                "type": "epoch_completed",
                                "epoch": metric.epoch,
                                "timestamp": metric.created_at.isoformat(),
                                "phase": metric.phase,
                                "loss": metric.loss,
                                "val_loss": metric.validation_loss,
                                "auc": metric.auc,
                                "learning_rate": metric.learning_rate,
                                "epoch_duration_seconds": metric.epoch_duration_seconds,
                                "examples_per_second": metric.examples_per_second,
                                "elapsed_seconds": _elapsed_seconds(run),
                            },
                            f"metric-{metric.id}",
                        )
                    )
                    last_metric_id = metric.id

                snapshots = list(
                    event_db.scalars(
                        select(TrainingNetworkSnapshot)
                        .where(
                            TrainingNetworkSnapshot.training_run_id == run_id,
                            TrainingNetworkSnapshot.id > last_snapshot_id,
                        )
                        .order_by(TrainingNetworkSnapshot.id)
                    )
                )
                for snapshot in snapshots:
                    snapshot_response = _snapshot_response(event_db, snapshot, run.config)
                    if snapshot_response is None:
                        last_snapshot_id = snapshot.id
                        continue
                    messages.append(
                        _sse(
                            "network_snapshot",
                            {
                                "type": "network_snapshot",
                                "epoch": snapshot.epoch,
                                "snapshot": snapshot_response.model_dump(mode="json"),
                            },
                            f"snapshot-{snapshot.id}",
                        )
                    )
                    last_snapshot_id = snapshot.id

                if run.status in {"COMPLETED", "FAILED"} and not terminal_seen:
                    response = _training_run_response(event_db, run)
                    if run.status == "COMPLETED":
                        messages.append(
                            _sse(
                                "completed",
                                {"type": "completed", "run": response.model_dump(mode="json")},
                                f"completed-{run.updated_at.isoformat()}",
                            )
                        )
                    else:
                        messages.append(
                            _sse(
                                "failed",
                                {
                                    "type": "failed",
                                    "error_message": run.error_message or "Training failed",
                                },
                                f"failed-{run.updated_at.isoformat()}",
                            )
                        )
                    terminal = True
                    terminal_seen = True
                elif run.status in {"COMPLETED", "FAILED"}:
                    terminal = True

            for message in messages:
                yield message
            if terminal:
                return
            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        persisted_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/models/current", response_model=ModelVersionResponse, tags=["models"])
def current_model(db: DbSession) -> ModelVersionResponse:
    model = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    if model is None:
        raise HTTPException(status_code=404, detail="No active model")
    response = ModelVersionResponse.model_validate(model)
    if response.config is not None:
        response.config = _public_training_config(response.config)
    return response


def _article_response(article: Article) -> ArticleResponse:
    return ArticleResponse(
        article_id=str(article.article_id),
        product_name=article.product_name,
        product_type_name=article.product_type_name,
        product_group_name=article.product_group_name,
        colour_group_name=article.colour_group_name,
        department_name=article.department_name,
        section_name=article.section_name,
        garment_group_name=article.garment_group_name,
        detail_desc=article.detail_desc,
    )


def _profile_id(customer: Customer) -> str:
    return customer.profile_id


def _customer_response(db: Session, customer: Customer) -> CustomerResponse:
    row = db.execute(
        select(
            func.count(Purchase.id),
            func.min(Purchase.t_dat),
            func.max(Purchase.t_dat),
        ).where(Purchase.customer_key == customer.customer_key)
    ).one()
    purchase_count = int(row[0] or 0)
    return CustomerResponse(
        profile_id=_profile_id(customer),
        age_band=customer.age_band,
        purchase_count=purchase_count,
        first_purchase_date=row[1],
        last_purchase_date=row[2],
        personalizable=purchase_count > 0,
    )


@router.get("/customers/random", response_model=CustomerResponse, tags=["customers"])
def random_customer(
    db: DbSession,
    personalizable: bool = True,
) -> CustomerResponse:
    candidates = select(Customer.customer_key)
    if personalizable:
        purchase_customers = select(Purchase.customer_key).distinct()
        candidates = candidates.where(Customer.customer_key.in_(purchase_customers))
    candidate_keys = candidates.subquery()
    total = int(db.scalar(select(func.count()).select_from(candidate_keys)) or 0)
    if not total:
        detail = "No personalizable customer profiles available" if personalizable else "No customer profiles available"
        raise HTTPException(status_code=404, detail=detail)
    customer = db.scalar(
        select(Customer)
        .where(Customer.customer_key.in_(select(candidate_keys.c.customer_key)))
        .order_by(Customer.customer_key)
        .offset(secrets.randbelow(total))
        .limit(1)
    )
    if customer is None:
        raise HTTPException(status_code=404, detail="No customer profiles available")
    return _customer_response(db, customer)


def _get_customer(db: Session, profile_id: str) -> Customer:
    customer = db.scalar(select(Customer).where(Customer.profile_id == profile_id))
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer profile not found")
    return customer


@router.get("/customers/{profile_id}", response_model=CustomerResponse, tags=["customers"])
def get_customer(profile_id: ProfileId, db: DbSession) -> CustomerResponse:
    return _customer_response(db, _get_customer(db, profile_id))


@router.get("/customers/{profile_id}/history", response_model=CustomerHistoryResponse, tags=["customers"])
def customer_history(
    profile_id: ProfileId,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CustomerHistoryResponse:
    customer = _get_customer(db, profile_id)
    total = int(
        db.scalar(
            select(func.count(Purchase.id)).where(
                Purchase.customer_key == customer.customer_key
            )
        )
        or 0
    )
    purchases = list(
        db.scalars(
            select(Purchase)
            .where(Purchase.customer_key == customer.customer_key)
            .order_by(desc(Purchase.t_dat), desc(Purchase.id))
            .offset(offset)
            .limit(limit)
        )
    )
    article_keys = {purchase.article_key for purchase in purchases}
    articles = {
        article.article_key: article
        for article in db.scalars(select(Article).where(Article.article_key.in_(article_keys)))
    }
    items = [
        CustomerHistoryItem(
            purchased_on=purchase.t_dat,
            price=purchase.price,
            sales_channel_id=purchase.sales_channel_id,
            article_id=str(articles[purchase.article_key].article_id),
            article=_article_response(articles[purchase.article_key]),
        )
        for purchase in purchases
        if purchase.article_key in articles
    ]
    return CustomerHistoryResponse(
        profile_id=profile_id, total=total, limit=limit, offset=offset, items=items
    )


def _popular_rows(
    db: Session,
    k: int,
    excluded_article_keys: set[int] | None = None,
) -> list[tuple[int, float]]:
    purchase_count = func.count(Purchase.id).label("purchase_count")
    query = select(Purchase.article_key, purchase_count).group_by(Purchase.article_key)
    if excluded_article_keys:
        query = query.where(Purchase.article_key.not_in(excluded_article_keys))
    rows = list(
        db.execute(query.order_by(desc(purchase_count), Purchase.article_key).limit(k))
    )
    maximum = int(rows[0][1]) if rows else 0
    return [
        (int(article_key), float(count) / maximum)
        for article_key, count in rows
        if maximum
    ]


def _recommendation_items(
    db: Session,
    customer: Customer | None,
    rows: Sequence[tuple[int, float]],
    history_articles: dict[int, Article],
    *,
    strategy: str,
) -> list[RecommendationItem]:
    recommendation_rows = rows
    item_keys = {item_key for item_key, _ in recommendation_rows}
    articles = {
        article.article_key: article
        for article in db.scalars(select(Article).where(Article.article_key.in_(item_keys)))
    }
    purchase_counts = {
        int(article_key): (int(purchases), int(customers))
        for article_key, purchases, customers in db.execute(
            select(
                Purchase.article_key,
                func.count(Purchase.id),
                func.count(func.distinct(Purchase.customer_key)),
            )
            .where(Purchase.article_key.in_(item_keys))
            .group_by(Purchase.article_key)
        )
    }
    cohort_counts: dict[int, int] = {}
    age_band = (
        customer.age_band
        if customer is not None
        else None
    )
    if age_band and item_keys:
        age_column = _age_band_column()
        cohort_rows = db.execute(
            select(Purchase.article_key, func.count(func.distinct(Purchase.customer_key)))
            .join(Customer, Customer.customer_key == Purchase.customer_key)
            .where(age_column == age_band, Purchase.article_key.in_(item_keys))
            .group_by(Purchase.article_key)
        )
        cohort_counts = {int(article_key): int(count) for article_key, count in cohort_rows}
    history_by_department: dict[str, list[str]] = {}
    for article in history_articles.values():
        if article.department_name:
            history_by_department.setdefault(article.department_name, []).append(
                str(article.article_id)
            )
    output: list[RecommendationItem] = []
    for item_key, score in recommendation_rows:
        article = articles.get(item_key)
        if article is None:
            continue
        reasons: list[RecommendationEvidence] = []
        purchases, buyers = purchase_counts.get(item_key, (0, 0))
        if strategy == "model":
            reasons.append(
                RecommendationEvidence(
                    kind="model_score",
                    description="Active model assigned this ranking score.",
                )
            )
        else:
            reasons.append(
                RecommendationEvidence(
                    kind="global_popularity",
                    description="Observed purchase count in this dataset.",
                    purchase_count=purchases,
                    customer_count=buyers,
                )
            )
        related_ids = history_by_department.get(article.department_name or "", [])
        if related_ids:
            reasons.append(
                RecommendationEvidence(
                    kind="purchase_history_department",
                    description="Article shares a catalog department with purchase history.",
                    related_article_ids=sorted(related_ids),
                )
            )
        cohort_buyers = cohort_counts.get(item_key, 0)
        if cohort_buyers:
            reasons.append(
                RecommendationEvidence(
                    kind="age_band_popularity",
                    description="Distinct buyers observed in profile age band.",
                    customer_count=cohort_buyers,
                )
            )
        output.append(
            RecommendationItem(
                rank=len(output) + 1,
                article_id=str(article.article_id),
                score=score,
                article=_article_response(article),
                evidence=reasons,
            )
        )
    return output


def _popularity_semantics() -> ScoreSemanticsResponse:
    return ScoreSemanticsResponse(
        name="normalized_purchase_popularity",
        description=(
            "Observed purchase count divided by the highest count among eligible articles; "
            "not a purchase probability or causal effect."
        ),
    )


def _model_semantics() -> ScoreSemanticsResponse:
    return ScoreSemanticsResponse(
        name="model_ranking_score",
        description=(
            "Model output used only for ranking; not a calibrated purchase probability or causal effect."
        ),
    )


def _assert_model_dataset_matches(db: Session, model: ModelVersion) -> None:
    manifest_path = FilePath(model.artifact_path) / "dataset_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail="Active model dataset metadata is invalid") from exc
    actual = {
        "purchases": int(db.scalar(select(func.count(Purchase.id))) or 0),
        "catalog_customers": int(db.scalar(select(func.count(Customer.customer_key))) or 0),
        "catalog_articles": int(db.scalar(select(func.count(Article.article_key))) or 0),
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("dataset") != "hm"
        or manifest.get("semantics") != "purchase_only"
        or manifest.get("sampled") is not False
        or any(manifest.get(name) != value for name, value in actual.items())
    ):
        raise HTTPException(
            status_code=409,
            detail="Active model artifact does not match current H&M dataset",
        )


def _normalize_backend_rows(
    recommendations: Sequence[dict[str, Any] | tuple[int, float]], k: int
) -> list[tuple[int, float]]:
    normalized: dict[int, float] = {}
    for recommendation in recommendations:
        if isinstance(recommendation, dict):
            article_key = int(recommendation["article_key"])
            score = float(recommendation["score"])
        else:
            article_key, score = int(recommendation[0]), float(recommendation[1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise HTTPException(status_code=502, detail="Recommendation model returned invalid scores")
        normalized[article_key] = max(score, normalized.get(article_key, 0.0))
    return sorted(normalized.items(), key=lambda item: (-item[1], item[0]))[:k]


@router.get(
    "/recommendations/popular",
    response_model=RecommendationResponse,
    tags=["recommendations"],
)
def popular_recommendations(
    db: DbSession,
    k: Annotated[int, Query(ge=1, le=100)] = 10,
) -> RecommendationResponse:
    started = time.perf_counter()
    rows = _popular_rows(db, k)
    return RecommendationResponse(
        strategy="popularity",
        score_semantics=_popularity_semantics(),
        exclude_seen_articles=False,
        inference_duration_ms=(time.perf_counter() - started) * 1000,
        items=_recommendation_items(db, None, rows, {}, strategy="popularity"),
    )


@router.get(
    "/customers/{profile_id}/recommendations",
    response_model=RecommendationResponse,
    tags=["recommendations"],
)
def customer_recommendations(
    profile_id: ProfileId,
    db: DbSession,
    backend: Annotated[RecommendationBackend, Depends(get_recommendation_backend)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    exclude_seen_articles: bool = True,
) -> RecommendationResponse:
    started = time.perf_counter()
    customer = _get_customer(db, profile_id)
    seen_keys = set(
        db.scalars(
            select(Purchase.article_key)
            .where(Purchase.customer_key == customer.customer_key)
            .distinct()
        )
    )
    history_articles = {
        article.article_key: article
        for article in db.scalars(select(Article).where(Article.article_key.in_(seen_keys)))
    }
    model = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)))
    fallback_reason: str | None = None
    model_metadata: ModelMetadataResponse | None = None
    if not seen_keys:
        rows = _popular_rows(db, k)
        fallback_reason = "no_purchase_history"
        strategy = "popularity"
    elif model is None:
        rows = _popular_rows(db, k, seen_keys if exclude_seen_articles else None)
        fallback_reason = "no_active_model"
        strategy = "popularity"
    else:
        _assert_model_dataset_matches(db, model)
        try:
            result = backend.recommend(
                db=db,
                model_version=model,
                customer_key=customer.customer_key,
                k=k,
                exclude_item_ids=seen_keys if exclude_seen_articles else set(),
            )
        except RecommendationUnavailable:
            rows = _popular_rows(db, k, seen_keys if exclude_seen_articles else None)
            fallback_reason = "model_unavailable"
            strategy = "popularity"
        else:
            if result.strategy == "neural":
                rows = _normalize_backend_rows(result.items, k)
                strategy = "model"
                model_metadata = ModelMetadataResponse(
                    version=model.version,
                    trained_at=model.created_at,
                    data_profile=(model.config or {}).get("data_profile"),
                )
            elif result.strategy == "popularity":
                rows = _popular_rows(db, k, seen_keys if exclude_seen_articles else None)
                strategy = "popularity"
                fallback_reason = "profile_not_in_model"
            else:
                raise HTTPException(status_code=502, detail="Recommendation backend returned invalid strategy")
    items = _recommendation_items(
        db, customer, rows, history_articles, strategy=strategy
    )
    if exclude_seen_articles and any(
        article_key in seen_keys
        for article_key in (
            db.scalar(select(Article.article_key).where(Article.article_id == item.article_id))
            for item in items
        )
    ):
        raise HTTPException(status_code=502, detail="Recommendation backend returned purchased articles")
    return RecommendationResponse(
        profile_id=profile_id,
        strategy=strategy,
        fallback_reason=fallback_reason,
        model=model_metadata,
        score_semantics=(
            _model_semantics() if strategy == "model" else _popularity_semantics()
        ),
        exclude_seen_articles=exclude_seen_articles,
        inference_duration_ms=(time.perf_counter() - started) * 1000,
        items=items,
    )
