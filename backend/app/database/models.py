from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


RUN_STATUSES = (
    "QUEUED",
    "PREPARING_DATA",
    "TRAINING",
    "EVALUATING",
    "SAVING_MODEL",
    "COMPLETED",
    "FAILED",
)
ACTIVE_RUN_STATUSES = RUN_STATUSES[:-2]
MODEL_STATUSES = ("CREATING", "READY", "FAILED", "ARCHIVED")

BIGINT_PRIMARY_KEY = BigInteger().with_variant(Integer, "sqlite")


class Customer(Base):
    """Synthetic customer profile with public and internal identifiers."""

    __tablename__ = "customers"

    customer_key: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    profile_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    age_band: Mapped[str] = mapped_column(String(16), nullable=False)
    state: Mapped[str] = mapped_column(String(2), nullable=False)
    region: Mapped[str] = mapped_column(String(32), nullable=False)

    __table_args__ = (
        CheckConstraint("length(customer_id) > 0", name="ck_customers_customer_id"),
        CheckConstraint("age >= 18 AND age <= 80", name="ck_customers_age"),
        Index("ix_customers_profile_id", "profile_id"),
    )


class Article(Base):
    """Generic product catalog used by recommendation models."""

    __tablename__ = "articles"

    article_key: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    product_name: Mapped[str] = mapped_column(Text, nullable=False)
    department_name: Mapped[str] = mapped_column(String(128), nullable=False)
    category_name: Mapped[str] = mapped_column(String(128), nullable=False)
    subcategory_name: Mapped[str] = mapped_column(String(128), nullable=False)
    brand: Mapped[str] = mapped_column(String(128), nullable=False)
    price_band: Mapped[str] = mapped_column(String(32), nullable=False)
    detail_desc: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint("length(article_id) > 0", name="ck_articles_article_id"),
        Index("ix_articles_department_name", "department_name"),
        Index("ix_articles_category_name", "category_name"),
    )


class Purchase(Base):
    """Observed purchase with BRL price and sales channel."""

    __tablename__ = "purchases"

    id: Mapped[int] = mapped_column(BIGINT_PRIMARY_KEY, primary_key=True, autoincrement=True)
    t_dat: Mapped[date] = mapped_column(Date, nullable=False)
    customer_key: Mapped[int] = mapped_column(
        ForeignKey("customers.customer_key", ondelete="RESTRICT"), nullable=False
    )
    article_key: Mapped[int] = mapped_column(
        ForeignKey("articles.article_key", ondelete="RESTRICT"), nullable=False
    )
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    sales_channel_id: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("price > 0", name="ck_purchases_price"),
        CheckConstraint("sales_channel_id IN (1, 2)", name="ck_purchases_sales_channel"),
        Index("ix_purchases_customer_date", "customer_key", "t_dat", "id"),
        Index("ix_purchases_customer_article", "customer_key", "article_key"),
        Index("ix_purchases_article_date", "article_key", "t_dat"),
    )


class DatasetState(Base):
    __tablename__ = "dataset_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="QUEUED", index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    device: Mapped[str | None] = mapped_column(String(255))
    tensorflow_version: Mapped[str | None] = mapped_column(String(64))
    python_version: Mapped[str | None] = mapped_column(String(64))
    available_ram_bytes: Mapped[int | None] = mapped_column(BigInteger)
    error_message: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(255))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    metrics: Mapped[list[TrainingMetric]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    snapshots: Mapped[list[TrainingNetworkSnapshot]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    model_versions: Mapped[list[ModelVersion]] = relationship(back_populates="training_run")

    __table_args__ = (
        CheckConstraint(
            "status IN ('QUEUED', 'PREPARING_DATA', 'TRAINING', 'EVALUATING', "
            "'SAVING_MODEL', 'COMPLETED', 'FAILED')",
            name="ck_training_runs_status",
        ),
        CheckConstraint("progress >= 0 AND progress <= 1", name="ck_training_runs_progress"),
        Index("ix_training_runs_created_at", "created_at"),
        Index("ix_training_runs_status_created_at", "status", "created_at"),
    )


class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id: Mapped[int] = mapped_column(BIGINT_PRIMARY_KEY, primary_key=True, autoincrement=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    phase: Mapped[str] = mapped_column(String(32), nullable=False, default="BENCHMARK")
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    loss: Mapped[float] = mapped_column(Float, nullable=False)
    validation_loss: Mapped[float | None] = mapped_column(Float)
    auc: Mapped[float | None] = mapped_column(Float)
    learning_rate: Mapped[float | None] = mapped_column(Float)
    epoch_duration_seconds: Mapped[float | None] = mapped_column(Float)
    examples_per_second: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    run: Mapped[TrainingRun] = relationship(back_populates="metrics")

    __table_args__ = (
        UniqueConstraint("training_run_id", "phase", "epoch", name="uq_training_metric_epoch"),
        CheckConstraint("epoch >= 1", name="ck_training_metrics_epoch"),
        CheckConstraint("loss >= 0", name="ck_training_metrics_loss"),
        Index("ix_training_metrics_run_phase_id", "training_run_id", "phase", "id"),
    )


class TrainingNetworkSnapshot(Base):
    __tablename__ = "training_network_snapshots"

    id: Mapped[int] = mapped_column(BIGINT_PRIMARY_KEY, primary_key=True, autoincrement=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    probe_customer_key: Mapped[int | None] = mapped_column(BigInteger)
    probe_article_key: Mapped[int | None] = mapped_column(BigInteger)
    activations: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    weights: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    prediction: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    run: Mapped[TrainingRun] = relationship(back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("training_run_id", "epoch", name="uq_training_snapshot_epoch"),
        CheckConstraint("epoch >= 1", name="ck_training_snapshots_epoch"),
        CheckConstraint(
            "prediction IS NULL OR (prediction >= 0 AND prediction <= 1)",
            name="ck_training_snapshots_prediction",
        ),
        Index("ix_training_snapshots_run_id", "training_run_id", "id"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="CREATING")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    training_run: Mapped[TrainingRun] = relationship(back_populates="model_versions")

    __table_args__ = (
        CheckConstraint(
            "status IN ('CREATING', 'READY', 'FAILED', 'ARCHIVED')",
            name="ck_model_versions_status",
        ),
        CheckConstraint("NOT is_active OR status = 'READY'", name="ck_active_model_ready"),
    )


_active_run_predicate = TrainingRun.status.in_(ACTIVE_RUN_STATUSES)
Index(
    "uq_training_runs_one_active",
    TrainingRun.id.is_not(None),
    unique=True,
    postgresql_where=_active_run_predicate,
    sqlite_where=_active_run_predicate,
)

_active_model_predicate = ModelVersion.is_active.is_(True)
Index(
    "uq_model_versions_one_active",
    ModelVersion.id.is_not(None),
    unique=True,
    postgresql_where=_active_model_predicate,
    sqlite_where=_active_model_predicate,
)
