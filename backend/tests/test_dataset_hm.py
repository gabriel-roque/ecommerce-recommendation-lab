from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence

import pytest
from sqlalchemy import Date, Numeric

from app.database.base import Base
from app.database.models import Article, Customer, Purchase
from backend.app.dataset import (
    DatasetError,
    DatasetValidationError,
    HM_HEADERS,
    import_hm,
    inspect_hm,
    public_profile_id,
)


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _fixture(root: Path) -> None:
    _write_csv(
        root / "customers.csv",
        HM_HEADERS["customers.csv"],
        (
            ("customer-source-1", "1.0", "1.0", "ACTIVE", "Regularly", "30", "postal-a"),
            ("customer-source-2", "", "", "PRE-CREATE", "NONE", "", "postal-b"),
            ("customer-source-3", "", "", "ACTIVE", "NONE", "45", "postal-c"),
        ),
    )
    _write_csv(
        root / "articles.csv",
        HM_HEADERS["articles.csv"],
        (
            (
                "0000000101", "0000001", "Basic Tee", "255", "T-shirt", "Garment Upper body",
                "1010016", "Solid", "9", "Black", "4", "Dark", "5", "Black", "1234",
                "Menswear", "A", "Menswear", "1", "Menswear", "16", "Mens Basics", "1002",
                "Jersey Basic", "A basic tee",
            ),
            (
                "0000000102", "0000001", "Basic Tee Blue", "255", "T-shirt", "Garment Upper body",
                "1010016", "Solid", "71", "Light Blue", "1", "Dusty Light", "2", "Blue", "1234",
                "Menswear", "A", "Menswear", "1", "Menswear", "16", "Mens Basics", "1002",
                "Jersey Basic", "",
            ),
        ),
    )
    _write_csv(
        root / "transactions_train.csv",
        HM_HEADERS["transactions_train.csv"],
        (
            ("2020-09-01", "customer-source-1", "0000000101", "0.10000000000000001", "2"),
            ("2020-09-02", "customer-source-1", "0000000102", "0.2", "1"),
            ("2020-09-02", "customer-source-2", "0000000101", "0.3", "2"),
        ),
    )


class RecordingSink:
    def __init__(self, *, partial_table: str | None = None) -> None:
        self.rows: dict[str, list[tuple[Any, ...]]] = {}
        self.columns: dict[str, tuple[str, ...]] = {}
        self.partial_table = partial_table
        self.committed = False
        self.rolled_back = False

    def begin(self) -> None:
        return None

    def write_rows(
        self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> int:
        self.columns[table] = tuple(columns)
        self.rows.setdefault(table, []).extend(tuple(row) for row in rows)
        return len(rows) - 1 if table == self.partial_table else len(rows)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_inspection_distinguishes_catalog_and_interaction_counts(tmp_path: Path) -> None:
    _fixture(tmp_path)

    result = inspect_hm(tmp_path)
    manifest = result["manifest"]

    assert manifest["catalog_customers"] == 3
    assert manifest["catalog_articles"] == 2
    assert manifest["interacting_customers"] == 2
    assert manifest["interacting_articles"] == 2
    assert manifest["purchases"] == 3
    assert manifest["date_range"] == {"start": "2020-09-01", "end": "2020-09-02"}
    assert len(manifest["source_fingerprint"]) == 64
    assert set(manifest["source_files"]) == set(HM_HEADERS)
    assert all(len(value["sha256"]) == 64 for value in manifest["source_files"].values())
    assert manifest["official_counts_validated"] is False


def test_import_preserves_all_fields_and_writes_only_canonical_tables(tmp_path: Path) -> None:
    _fixture(tmp_path)
    sink = RecordingSink()

    result = import_hm(tmp_path, sink, chunk_size=1)

    assert result["table_rows"] == {"customers": 3, "articles": 2, "purchases": 3}
    assert set(sink.rows) == {"customers", "articles", "purchases"}
    assert sink.committed is True
    customer = sink.rows["customers"][0]
    assert customer[1] == "customer-source-1"
    assert customer[2] == public_profile_id("customer-source-1")
    assert customer[-1] == "postal-a"
    assert len(sink.columns["articles"]) == len(HM_HEADERS["articles.csv"]) + 1
    assert sink.rows["articles"][0][-1] == "A basic tee"
    purchase = sink.rows["purchases"][0]
    assert purchase[0] == date(2020, 9, 1)
    assert purchase[3] == Decimal("0.10000000000000001")


def test_partial_copy_rolls_back(tmp_path: Path) -> None:
    _fixture(tmp_path)
    sink = RecordingSink(partial_table="articles")

    with pytest.raises(DatasetError, match="partial import into articles"):
        import_hm(tmp_path, sink)

    assert sink.rolled_back is True
    assert sink.committed is False


def test_full_mode_enforces_official_counts(tmp_path: Path) -> None:
    _fixture(tmp_path)

    with pytest.raises(DatasetValidationError, match="expected 1371980, got 3"):
        inspect_hm(tmp_path, full=True)


def test_rejects_non_exact_schema_and_invalid_calendar_date(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write_csv(tmp_path / "customers.csv", HM_HEADERS["customers.csv"][:-1], ())
    with pytest.raises(DatasetValidationError, match="schema mismatch"):
        inspect_hm(tmp_path)

    _fixture(tmp_path)
    _write_csv(
        tmp_path / "transactions_train.csv",
        HM_HEADERS["transactions_train.csv"],
        (("2020-02-30", "customer-source-1", "0000000101", "0.1", "2"),),
    )
    with pytest.raises(DatasetValidationError, match="ISO calendar date"):
        inspect_hm(tmp_path)


def test_canonical_schema_has_foreign_keys_checks_and_exact_types() -> None:
    assert {"customers", "articles", "purchases"}.issubset(Base.metadata.tables)
    assert {"customers", "articles", "purchases"}.issubset(Base.metadata.tables)
    assert Customer.__table__.c.profile_id.unique is True
    assert Article.__table__.c.prod_name.nullable is False
    assert isinstance(Purchase.__table__.c.t_dat.type, Date)
    assert isinstance(Purchase.__table__.c.price.type, Numeric)
    assert Purchase.__table__.c.price.type.scale == 18
    targets = {
        foreign_key.target_fullname
        for column in (Purchase.__table__.c.customer_key, Purchase.__table__.c.article_key)
        for foreign_key in column.foreign_keys
    }
    assert targets == {"customers.customer_key", "articles.article_key"}
    check_names = {constraint.name for constraint in Purchase.__table__.constraints}
    assert {"ck_purchases_price", "ck_purchases_sales_channel"}.issubset(check_names)
