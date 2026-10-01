"""Validation, manifests, and PostgreSQL COPY import for H&M data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import uuid
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .importer import BulkImportSink
from .models import DatasetError, DatasetValidationError

HM_FILES = ("customers.csv", "articles.csv", "transactions_train.csv")
HM_FULL_COUNTS: Mapping[str, int] = {
    "customers.csv": 1_371_980,
    "articles.csv": 105_542,
    "transactions_train.csv": 31_788_324,
}
HM_HEADERS = {
    "customers.csv": (
        "customer_id",
        "FN",
        "Active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "postal_code",
    ),
    "articles.csv": (
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_no",
        "graphical_appearance_name",
        "colour_group_code",
        "colour_group_name",
        "perceived_colour_value_id",
        "perceived_colour_value_name",
        "perceived_colour_master_id",
        "perceived_colour_master_name",
        "department_no",
        "department_name",
        "index_code",
        "index_name",
        "index_group_no",
        "index_group_name",
        "section_no",
        "section_name",
        "garment_group_no",
        "garment_group_name",
        "detail_desc",
    ),
    "transactions_train.csv": (
        "t_dat",
        "customer_id",
        "article_id",
        "price",
        "sales_channel_id",
    ),
}

CUSTOMER_COLUMNS = (
    "customer_key",
    "customer_id",
    "profile_id",
    "fn",
    "active",
    "club_member_status",
    "fashion_news_frequency",
    "age",
    "age_band",
    "postal_code",
)
ARTICLE_COLUMNS = ("article_key", *HM_HEADERS["articles.csv"])
PURCHASE_COLUMNS = ("t_dat", "customer_key", "article_key", "price", "sales_channel_id")
PROFILE_NAMESPACE = uuid.UUID("f54aaec4-0297-5f5e-a755-638e6971d48d")


def is_hm_dataset(raw_dir: Path | str) -> bool:
    root = Path(raw_dir)
    return all((root / name).is_file() for name in HM_FILES)


def require_hm_files(raw_dir: Path | str) -> dict[str, Path]:
    root = Path(raw_dir)
    missing = [name for name in HM_FILES if not (root / name).is_file()]
    if missing:
        raise DatasetValidationError(
            f"missing required H&M files: {', '.join(missing)}", file=root
        )
    return {name: root / name for name in HM_FILES}


def public_profile_id(customer_id: str) -> str:
    """Return stable opaque ID without exposing source customer identifier."""
    return str(uuid.uuid5(PROFILE_NAMESPACE, customer_id))


def _reader(path: Path) -> tuple[Any, csv.DictReader[str]]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle)
    actual = tuple(reader.fieldnames or ())
    expected = HM_HEADERS[path.name]
    if actual != expected:
        handle.close()
        raise DatasetValidationError(
            f"schema mismatch: expected {expected}, got {actual}", file=path, row=1
        )
    return handle, reader


def _required(value: str | None, path: Path, row: int, column: str) -> str:
    raw = value or ""
    if not raw.strip():
        raise DatasetValidationError("must not be empty", file=path, row=row, column=column)
    return raw


def _optional(value: str | None) -> str | None:
    raw = value or ""
    return raw if raw.strip() else None


def _validate_row(row: dict[str, str], path: Path, row_number: int) -> None:
    if None in row or any(value is None for value in row.values()):
        raise DatasetValidationError("malformed CSV row", file=path, row=row_number)


def _integer(value: str | None, path: Path, row: int, column: str) -> int:
    raw = _required(value, path, row, column)
    try:
        return int(raw)
    except ValueError as exc:
        raise DatasetValidationError("must be an integer", file=path, row=row, column=column) from exc


def _indicator(value: str | None, path: Path, row: int, column: str) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    if raw in {"0", "0.0"}:
        return 0
    if raw in {"1", "1.0"}:
        return 1
    raise DatasetValidationError("must be 0, 1, or empty", file=path, row=row, column=column)


def _age(value: str | None, path: Path, row: int) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        age = int(raw)
    except ValueError as exc:
        raise DatasetValidationError("must be an integer", file=path, row=row, column="age") from exc
    if not 0 < age < 130:
        raise DatasetValidationError("must be between 1 and 129", file=path, row=row, column="age")
    return age


def _age_band(age: int | None) -> str | None:
    if age is None:
        return None
    if age < 18:
        return "under 18"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    return "55+"


def _transaction_date(value: str | None, path: Path, row: int) -> date:
    raw = _required(value, path, row, "t_dat")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise DatasetValidationError("must be an ISO calendar date", file=path, row=row, column="t_dat") from exc
    if parsed.isoformat() != raw:
        raise DatasetValidationError("must use YYYY-MM-DD format", file=path, row=row, column="t_dat")
    return parsed


def _price(value: str | None, path: Path, row: int) -> Decimal:
    raw = _required(value, path, row, "price")
    try:
        parsed = Decimal(raw)
    except InvalidOperation as exc:
        raise DatasetValidationError("must be a decimal", file=path, row=row, column="price") from exc
    if not parsed.is_finite() or parsed < 0:
        raise DatasetValidationError("must be a finite non-negative decimal", file=path, row=row, column="price")
    return parsed


def _channel(value: str | None, path: Path, row: int) -> int:
    channel = _integer(value, path, row, "sales_channel_id")
    if channel not in (1, 2):
        raise DatasetValidationError("must be 1 or 2", file=path, row=row, column="sales_channel_id")
    return channel


def _article_values(row: dict[str, str], path: Path, row_number: int) -> tuple[Any, ...]:
    values: list[Any] = []
    integer_fields = {
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_value_id",
        "perceived_colour_master_id",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    }
    for column in HM_HEADERS["articles.csv"]:
        if column == "detail_desc":
            values.append(_optional(row[column]))
        elif column in integer_fields:
            values.append(_integer(row[column], path, row_number, column))
        else:
            values.append(_required(row[column], path, row_number, column))
    return tuple(values)


def _checksum(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _source_fingerprint(source_files: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name in HM_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(source_files[name]["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(source_files[name]["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_counts(
    rows: Mapping[str, int], paths: Mapping[str, Path], expected: Mapping[str, int] | None
) -> None:
    if expected is None:
        return
    unknown = set(expected) - set(HM_FILES)
    if unknown:
        raise ValueError(f"unknown expected count files: {sorted(unknown)}")
    for name, expected_count in expected.items():
        actual = rows[name]
        if actual != expected_count:
            raise DatasetValidationError(
                f"row count mismatch: expected {expected_count}, got {actual}", file=paths[name]
            )


def inspect_hm(
    raw_dir: Path | str,
    *,
    full: bool = False,
    expected_counts: Mapping[str, int] | None = None,
    top_items_limit: int = 20,
) -> dict[str, Any]:
    """Validate every source row and produce exact catalog and interaction counts."""
    if full and expected_counts is not None:
        raise ValueError("full and expected_counts cannot be combined")
    paths = require_hm_files(raw_dir)
    customer_ids: set[str] = set()
    article_ids: set[str] = set()
    interacting_customers: set[str] = set()
    interacting_articles: set[str] = set()
    customer_purchase_counts: Counter[str] = Counter()
    article_purchase_counts: Counter[str] = Counter()
    age_band_counts: Counter[str] = Counter()
    missing_age = 0
    file_rows: dict[str, int] = {}

    handle, reader = _reader(paths["customers.csv"])
    try:
        for row_number, row in enumerate(reader, start=2):
            _validate_row(row, paths["customers.csv"], row_number)
            customer_id = _required(row["customer_id"], paths["customers.csv"], row_number, "customer_id")
            if customer_id in customer_ids:
                raise DatasetValidationError("duplicate customer_id", file=paths["customers.csv"], row=row_number, column="customer_id")
            customer_ids.add(customer_id)
            _indicator(row["FN"], paths["customers.csv"], row_number, "FN")
            _indicator(row["Active"], paths["customers.csv"], row_number, "Active")
            age = _age(row["age"], paths["customers.csv"], row_number)
            age_band_counts[_age_band(age) or "unknown"] += 1
            if age is None:
                missing_age += 1
        file_rows["customers.csv"] = len(customer_ids)
    finally:
        handle.close()

    handle, reader = _reader(paths["articles.csv"])
    try:
        for row_number, row in enumerate(reader, start=2):
            _validate_row(row, paths["articles.csv"], row_number)
            values = _article_values(row, paths["articles.csv"], row_number)
            article_id = str(values[0])
            if article_id in article_ids:
                raise DatasetValidationError("duplicate article_id", file=paths["articles.csv"], row=row_number, column="article_id")
            article_ids.add(article_id)
        file_rows["articles.csv"] = len(article_ids)
    finally:
        handle.close()

    min_date: date | None = None
    max_date: date | None = None
    purchases = 0
    handle, reader = _reader(paths["transactions_train.csv"])
    try:
        for row_number, row in enumerate(reader, start=2):
            _validate_row(row, paths["transactions_train.csv"], row_number)
            customer_id = _required(row["customer_id"], paths["transactions_train.csv"], row_number, "customer_id")
            article_id = _required(row["article_id"], paths["transactions_train.csv"], row_number, "article_id")
            if customer_id not in customer_ids:
                raise DatasetValidationError("unknown customer_id", file=paths["transactions_train.csv"], row=row_number, column="customer_id")
            if article_id not in article_ids:
                raise DatasetValidationError("unknown article_id", file=paths["transactions_train.csv"], row=row_number, column="article_id")
            purchased_on = _transaction_date(row["t_dat"], paths["transactions_train.csv"], row_number)
            _price(row["price"], paths["transactions_train.csv"], row_number)
            _channel(row["sales_channel_id"], paths["transactions_train.csv"], row_number)
            purchases += 1
            interacting_customers.add(customer_id)
            interacting_articles.add(article_id)
            customer_purchase_counts[customer_id] += 1
            article_purchase_counts[article_id] += 1
            min_date = purchased_on if min_date is None else min(min_date, purchased_on)
            max_date = purchased_on if max_date is None else max(max_date, purchased_on)
        file_rows["transactions_train.csv"] = purchases
    finally:
        handle.close()

    if purchases == 0 or min_date is None or max_date is None:
        raise DatasetValidationError(
            "must contain at least one purchase", file=paths["transactions_train.csv"]
        )
    expected = HM_FULL_COUNTS if full else expected_counts
    _validate_counts(file_rows, paths, expected)

    source_files = {
        name: {"rows": file_rows[name], **_checksum(paths[name])} for name in HM_FILES
    }
    manifest = {
        "schema_version": 1,
        "dataset": "hm",
        "dataset_name": "H&M Personalized Fashion Recommendations",
        "source": "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations",
        "files": file_rows,
        "source_files": source_files,
        "source_fingerprint": _source_fingerprint(source_files),
        "official_counts_validated": full,
        "sampled": not full,
        "catalog_customers": len(customer_ids),
        "catalog_articles": len(article_ids),
        "interacting_customers": len(interacting_customers),
        "interacting_articles": len(interacting_articles),
        "purchases": purchases,
        "date_range": {"start": min_date.isoformat(), "end": max_date.isoformat()},
        "semantics": "purchase_only",
    }
    purchase_counts = list(customer_purchase_counts.values())
    all_purchase_counts = [0] * (len(customer_ids) - len(purchase_counts)) + purchase_counts
    purchase_histogram = {
        "0": sum(value == 0 for value in all_purchase_counts),
        "1": sum(value == 1 for value in all_purchase_counts),
        "2-5": sum(2 <= value <= 5 for value in all_purchase_counts),
        "6-10": sum(6 <= value <= 10 for value in all_purchase_counts),
        "11-50": sum(11 <= value <= 50 for value in all_purchase_counts),
        "51-100": sum(51 <= value <= 100 for value in all_purchase_counts),
        "101+": sum(value >= 101 for value in all_purchase_counts),
    }
    stats = {
        "dataset_name": manifest["dataset_name"],
        "catalog_customers": manifest["catalog_customers"],
        "catalog_articles": manifest["catalog_articles"],
        "interacting_customers": manifest["interacting_customers"],
        "interacting_articles": manifest["interacting_articles"],
        "purchases": purchases,
        "source_fingerprint": manifest["source_fingerprint"],
        "average_purchases_per_interacting_customer": purchases / len(interacting_customers),
        "median_purchases_per_interacting_customer": float(statistics.median(purchase_counts)),
        "median_purchases_per_customer": float(statistics.median(all_purchase_counts)),
        "purchases_per_customer": purchase_histogram,
        "customers_without_age": missing_age,
        "age_coverage_ratio": (len(customer_ids) - missing_age) / len(customer_ids),
        "age_band_counts": dict(sorted(age_band_counts.items())),
        "date_range": manifest["date_range"],
        "top_articles": [
            {"article_id": article_id, "purchases": count}
            for article_id, count in sorted(
                article_purchase_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )[:top_items_limit]
        ],
        "semantics": "purchase_only",
    }
    return {"manifest": manifest, "stats": stats}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_hm_artifacts(
    raw_dir: Path | str, processed_dir: Path | str, *, full: bool = False
) -> dict[str, Any]:
    result = inspect_hm(raw_dir, full=full)
    output = Path(processed_dir)
    _write_json(output / "dataset_manifest.json", result["manifest"])
    _write_json(output / "dataset_stats.json", result["stats"])
    return result


def _write_checked(
    sink: BulkImportSink,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> int:
    written = sink.write_rows(table, columns, rows)
    if written != len(rows):
        raise DatasetError(f"partial import into {table}: sent {len(rows)}, wrote {written}")
    return written


def import_hm(
    raw_dir: Path | str,
    sink: BulkImportSink,
    *,
    full: bool = False,
    chunk_size: int = 50_000,
) -> dict[str, dict[str, int]]:
    """COPY only canonical customers, articles, and purchases in one transaction."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    paths = require_hm_files(raw_dir)
    customer_keys: dict[str, int] = {}
    article_keys: dict[str, int] = {}
    source_rows = {name: 0 for name in HM_FILES}
    table_rows = {"customers": 0, "articles": 0, "purchases": 0}
    sink.begin()
    try:
        handle, reader = _reader(paths["customers.csv"])
        try:
            rows: list[tuple[Any, ...]] = []
            for key, row in enumerate(reader, start=1):
                row_number = key + 1
                _validate_row(row, paths["customers.csv"], row_number)
                customer_id = _required(row["customer_id"], paths["customers.csv"], row_number, "customer_id")
                if customer_id in customer_keys:
                    raise DatasetValidationError("duplicate customer_id", file=paths["customers.csv"], row=row_number, column="customer_id")
                customer_keys[customer_id] = key
                age = _age(row["age"], paths["customers.csv"], row_number)
                rows.append(
                    (
                        key,
                        customer_id,
                        public_profile_id(customer_id),
                        _indicator(row["FN"], paths["customers.csv"], row_number, "FN"),
                        _indicator(row["Active"], paths["customers.csv"], row_number, "Active"),
                        _optional(row["club_member_status"]),
                        _optional(row["fashion_news_frequency"]),
                        age,
                        _age_band(age),
                        _optional(row["postal_code"]),
                    )
                )
                if len(rows) == chunk_size:
                    table_rows["customers"] += _write_checked(sink, "customers", CUSTOMER_COLUMNS, rows)
                    rows.clear()
            if rows:
                table_rows["customers"] += _write_checked(sink, "customers", CUSTOMER_COLUMNS, rows)
            source_rows["customers.csv"] = len(customer_keys)
        finally:
            handle.close()

        handle, reader = _reader(paths["articles.csv"])
        try:
            article_rows: list[tuple[Any, ...]] = []
            for key, row in enumerate(reader, start=1):
                row_number = key + 1
                _validate_row(row, paths["articles.csv"], row_number)
                values = _article_values(row, paths["articles.csv"], row_number)
                article_id = str(values[0])
                if article_id in article_keys:
                    raise DatasetValidationError("duplicate article_id", file=paths["articles.csv"], row=row_number, column="article_id")
                article_keys[article_id] = key
                article_rows.append((key, *values))
                if len(article_rows) == chunk_size:
                    table_rows["articles"] += _write_checked(sink, "articles", ARTICLE_COLUMNS, article_rows)
                    article_rows.clear()
            if article_rows:
                table_rows["articles"] += _write_checked(sink, "articles", ARTICLE_COLUMNS, article_rows)
            source_rows["articles.csv"] = len(article_keys)
        finally:
            handle.close()

        handle, reader = _reader(paths["transactions_train.csv"])
        try:
            purchase_rows: list[tuple[Any, ...]] = []
            for row_number, row in enumerate(reader, start=2):
                _validate_row(row, paths["transactions_train.csv"], row_number)
                customer_id = _required(row["customer_id"], paths["transactions_train.csv"], row_number, "customer_id")
                article_id = _required(row["article_id"], paths["transactions_train.csv"], row_number, "article_id")
                if customer_id not in customer_keys:
                    raise DatasetValidationError("unknown customer_id", file=paths["transactions_train.csv"], row=row_number, column="customer_id")
                if article_id not in article_keys:
                    raise DatasetValidationError("unknown article_id", file=paths["transactions_train.csv"], row=row_number, column="article_id")
                purchase_rows.append(
                    (
                        _transaction_date(row["t_dat"], paths["transactions_train.csv"], row_number),
                        customer_keys[customer_id],
                        article_keys[article_id],
                        _price(row["price"], paths["transactions_train.csv"], row_number),
                        _channel(row["sales_channel_id"], paths["transactions_train.csv"], row_number),
                    )
                )
                if len(purchase_rows) == chunk_size:
                    table_rows["purchases"] += _write_checked(sink, "purchases", PURCHASE_COLUMNS, purchase_rows)
                    purchase_rows.clear()
            if purchase_rows:
                table_rows["purchases"] += _write_checked(sink, "purchases", PURCHASE_COLUMNS, purchase_rows)
            source_rows["transactions_train.csv"] = table_rows["purchases"]
        finally:
            handle.close()

        _validate_counts(source_rows, paths, HM_FULL_COUNTS if full else None)
        sink.commit()
    except BaseException:
        sink.rollback()
        raise
    return {"source_rows": source_rows, "table_rows": table_rows}
