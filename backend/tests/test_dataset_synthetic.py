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
from app.dataset import (
    DATASET_COUNTS,
    DATASET_HEADERS,
    DatasetError,
    DatasetValidationError,
    generate_dataset,
    import_synthetic,
    inspect_synthetic,
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
        DATASET_HEADERS["customers.csv"],
        (
            ("CLI1", public_profile_id("CLI1"), "Lucas Almeida", "30", "25-34", "SP", "Sudeste"),
            ("CLI2", public_profile_id("CLI2"), "Mariana Souza", "41", "35-44", "PE", "Nordeste"),
        ),
    )
    _write_csv(
        root / "products.csv",
        DATASET_HEADERS["products.csv"],
        (
            ("PROD1", "Notebook Prático", "Eletrônicos", "Informática", "Notebook", "Aurora", "premium", "Notebook para uso cotidiano."),
            ("PROD2", "Caderno Essencial", "Livros", "Papelaria", "Caderno", "Ipê", "econômico", "Caderno para estudo diário."),
        ),
    )
    _write_csv(
        root / "purchases.csv",
        DATASET_HEADERS["purchases.csv"],
        (
            ("2025-01-01", "CLI1", "PROD1", "1999.90", "2"),
            ("2025-01-02", "CLI2", "PROD2", "29.90", "1"),
            ("2025-01-03", "CLI1", "PROD2", "31.50", "2"),
        ),
    )


class RecordingSink:
    def __init__(self, partial_table: str | None = None) -> None:
        self.rows: dict[str, list[tuple[Any, ...]]] = {}
        self.columns: dict[str, tuple[str, ...]] = {}
        self.partial_table = partial_table
        self.committed = False
        self.rolled_back = False

    def begin(self) -> None:
        pass

    def write_rows(self, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> int:
        self.columns[table] = tuple(columns)
        self.rows.setdefault(table, []).extend(tuple(row) for row in rows)
        return len(rows) - 1 if table == self.partial_table else len(rows)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_generator_is_deterministic_and_has_exact_versioned_counts(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = generate_dataset(first)
    two = generate_dataset(second)

    assert one["manifest"]["files"] == DATASET_COUNTS
    assert one["manifest"]["total_records"] == 50_000
    assert one["manifest"]["source_fingerprint"] == two["manifest"]["source_fingerprint"]
    assert one["manifest"]["synthetic"] is True
    assert one["manifest"]["sampled"] is False
    assert one["manifest"]["currency"] == "BRL"
    assert one["stats"]["personas"] == [
        "Lucas Almeida", "Mariana Souza", "Rafael Oliveira", "Camila Santos", "Juliana Costa", "Bruno Ferreira"
    ]
    for filename in DATASET_COUNTS:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    with (first / "products.csv").open(encoding="utf-8", newline="") as stream:
        products = {
            row["article_id"]: (row["department_name"], row["category_name"])
            for row in csv.DictReader(stream)
        }
    persona_preferences = {
        "CLI000001": ("Eletrônicos", "Informática"),
        "CLI000002": ("Livros", "Papelaria"),
        "CLI000003": ("Esporte", "Esporte"),
        "CLI000004": ("Casa", "Cozinha"),
        "CLI000005": ("Beleza", "Cuidados pessoais"),
        "CLI000006": ("Games", "Informática"),
    }
    totals = {customer_id: 0 for customer_id in persona_preferences}
    matches = {customer_id: 0 for customer_id in persona_preferences}
    with (first / "purchases.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            customer_id = row["customer_id"]
            if customer_id in persona_preferences:
                totals[customer_id] += 1
                matches[customer_id] += products[row["article_id"]] == persona_preferences[customer_id]
    assert all(matches[key] / totals[key] >= 0.65 for key in persona_preferences)


def test_inspection_validates_foreign_keys_and_brl_values(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = inspect_synthetic(
        tmp_path,
        expected_counts={"customers.csv": 2, "products.csv": 2, "purchases.csv": 3},
    )
    assert result["manifest"]["interacting_customers"] == 2
    assert result["manifest"]["interacting_articles"] == 2
    assert result["manifest"]["date_range"] == {"start": "2025-01-01", "end": "2025-01-03"}

    _write_csv(
        tmp_path / "purchases.csv",
        DATASET_HEADERS["purchases.csv"],
        (("2025-01-01", "DESCONHECIDO", "PROD1", "10.00", "1"),),
    )
    with pytest.raises(DatasetValidationError, match="chave estrangeira"):
        inspect_synthetic(tmp_path, expected_counts=None)


def test_import_writes_only_generic_canonical_tables(tmp_path: Path) -> None:
    _fixture(tmp_path)
    sink = RecordingSink()
    result = import_synthetic(
        tmp_path,
        sink,
        expected_counts={"customers.csv": 2, "products.csv": 2, "purchases.csv": 3},
        chunk_size=1,
    )
    assert result["table_rows"] == {"customers": 2, "articles": 2, "purchases": 3}
    assert set(sink.rows) == {"customers", "articles", "purchases"}
    assert sink.rows["customers"][0][3] == "Lucas Almeida"
    assert sink.rows["articles"][0][2] == "Notebook Prático"
    assert sink.rows["purchases"][0] == (date(2025, 1, 1), 1, 1, Decimal("1999.90"), 2)
    assert sink.committed is True


def test_partial_copy_rolls_back(tmp_path: Path) -> None:
    _fixture(tmp_path)
    sink = RecordingSink("articles")
    with pytest.raises(DatasetError, match="importação parcial"):
        import_synthetic(tmp_path, sink, expected_counts=None)
    assert sink.rolled_back is True
    assert sink.committed is False


def test_schema_is_generic_and_keeps_exact_brl_type() -> None:
    assert {"customers", "articles", "purchases"}.issubset(Base.metadata.tables)
    assert Customer.__table__.c.name.nullable is False
    assert Article.__table__.c.product_name.nullable is False
    assert "product_group_name" not in Article.__table__.c
    assert isinstance(Purchase.__table__.c.t_dat.type, Date)
    assert isinstance(Purchase.__table__.c.price.type, Numeric)
    assert Purchase.__table__.c.price.type.scale == 2
    targets = {
        foreign_key.target_fullname
        for column in (Purchase.__table__.c.customer_key, Purchase.__table__.c.article_key)
        for foreign_key in column.foreign_keys
    }
    assert targets == {"customers.customer_key", "articles.article_key"}


def test_new_dataset_module_has_no_retailer_identity() -> None:
    source = (Path(__file__).parents[1] / "app/dataset/synthetic.py").read_text(encoding="utf-8")
    assert "H" + "&M" not in source
