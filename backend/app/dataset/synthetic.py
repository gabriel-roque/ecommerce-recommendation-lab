"""Deterministic Brazilian synthetic dataset generation, validation, and import."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import uuid
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from .importer import BulkImportSink
from .models import DatasetError, DatasetValidationError

DATASET_ID = "ecommerce_sintetico_br"
DATASET_NAME = "E-commerce Recommendation Lab"
DATASET_SEED = 20261001
DATASET_FILES = ("customers.csv", "products.csv", "purchases.csv")
DATASET_COUNTS: Mapping[str, int] = {
    "customers.csv": 2_000,
    "products.csv": 500,
    "purchases.csv": 47_500,
}
DATASET_HEADERS = {
    "customers.csv": (
        "customer_id",
        "profile_id",
        "name",
        "age",
        "age_band",
        "state",
        "region",
    ),
    "products.csv": (
        "article_id",
        "product_name",
        "department_name",
        "category_name",
        "subcategory_name",
        "brand",
        "price_band",
        "detail_desc",
    ),
    "purchases.csv": (
        "purchased_on",
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
    "name",
    "age",
    "age_band",
    "state",
    "region",
)
ARTICLE_COLUMNS = (
    "article_key",
    "article_id",
    "product_name",
    "department_name",
    "category_name",
    "subcategory_name",
    "brand",
    "price_band",
    "detail_desc",
)
PURCHASE_COLUMNS = ("t_dat", "customer_key", "article_key", "price", "sales_channel_id")

PERSONAS = (
    ("Lucas Almeida", "Eletrônicos", "Informática"),
    ("Mariana Souza", "Livros", "Papelaria"),
    ("Rafael Oliveira", "Esporte", "Esporte"),
    ("Camila Santos", "Casa", "Cozinha"),
    ("Juliana Costa", "Beleza", "Cuidados pessoais"),
    ("Bruno Ferreira", "Games", "Informática"),
)

TAXONOMY = (
    ("Eletrônicos", "Informática", "Notebook"),
    ("Eletrônicos", "Informática", "Acessório para computador"),
    ("Eletrônicos", "Telefonia", "Celular"),
    ("Eletrônicos", "Áudio", "Fone de ouvido"),
    ("Livros", "Literatura", "Romance brasileiro"),
    ("Livros", "Educação", "Livro didático"),
    ("Livros", "Papelaria", "Caderno"),
    ("Livros", "Papelaria", "Caneta"),
    ("Esporte", "Esporte", "Tênis esportivo"),
    ("Esporte", "Treino", "Equipamento de exercício"),
    ("Esporte", "Lazer", "Bola"),
    ("Casa", "Cozinha", "Panela"),
    ("Casa", "Cozinha", "Utensílio"),
    ("Casa", "Decoração", "Luminária"),
    ("Casa", "Organização", "Organizador"),
    ("Beleza", "Cuidados pessoais", "Hidratante"),
    ("Beleza", "Maquiagem", "Batom"),
    ("Beleza", "Cabelos", "Xampu"),
    ("Games", "Informática", "Periférico para jogos"),
    ("Games", "Jogos", "Jogo de tabuleiro"),
    ("Games", "Consoles", "Controle"),
    ("Moda", "Vestuário", "Camiseta"),
    ("Moda", "Calçados", "Sapato"),
    ("Alimentos", "Mercearia", "Café"),
    ("Alimentos", "Saudáveis", "Lanche natural"),
)

FIRST_NAMES = (
    "Ana", "Beatriz", "Carla", "Daniel", "Eduardo", "Fernanda", "Gabriel",
    "Helena", "Igor", "João", "Larissa", "Márcia", "Nicolas", "Otávio",
    "Patrícia", "Renata", "Sérgio", "Talita", "Vinícius", "Yasmin",
)
LAST_NAMES = (
    "Alves", "Barbosa", "Cardoso", "Dias", "Esteves", "Freitas", "Gomes",
    "Lima", "Martins", "Mendes", "Nascimento", "Pereira", "Ribeiro",
    "Rocha", "Silva", "Teixeira", "Vieira",
)
STATES = (
    ("SP", "Sudeste"), ("RJ", "Sudeste"), ("MG", "Sudeste"),
    ("ES", "Sudeste"), ("PR", "Sul"), ("SC", "Sul"), ("RS", "Sul"),
    ("BA", "Nordeste"), ("PE", "Nordeste"), ("CE", "Nordeste"),
    ("GO", "Centro-Oeste"), ("DF", "Centro-Oeste"), ("PA", "Norte"),
    ("AM", "Norte"),
)
BRANDS = ("Aurora", "Ipê", "Jatobá", "Horizonte", "Sereno", "Caju", "Arara", "Vereda")
QUALITIES = ("Essencial", "Prático", "Conforto", "Premium", "Compacto", "Clássico")
PRICE_RANGES = {
    "econômico": (Decimal("12.90"), Decimal("79.90")),
    "intermediário": (Decimal("80.00"), Decimal("299.90")),
    "premium": (Decimal("300.00"), Decimal("2499.90")),
}
PROFILE_NAMESPACE = uuid.UUID("c72efcb2-f859-58a0-9327-301621297a65")


def age_band(age: int) -> str:
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    return "55+"


def public_profile_id(customer_id: str) -> str:
    return str(uuid.uuid5(PROFILE_NAMESPACE, customer_id))


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
    os.replace(temporary, path)


def _customer_rows(rng: random.Random) -> tuple[list[tuple[Any, ...]], list[tuple[str, str]]]:
    rows: list[tuple[Any, ...]] = []
    affinities: list[tuple[str, str]] = []
    for index in range(DATASET_COUNTS["customers.csv"]):
        customer_id = f"CLI{index + 1:06d}"
        if index < len(PERSONAS):
            name, department, category = PERSONAS[index]
        else:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            department, category, _ = rng.choice(TAXONOMY)
        age = rng.randint(18, 72)
        state, region = rng.choice(STATES)
        rows.append(
            (customer_id, public_profile_id(customer_id), name, age, age_band(age), state, region)
        )
        affinities.append((department, category))
    return rows, affinities


def _product_rows() -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    bands = tuple(PRICE_RANGES)
    for index in range(DATASET_COUNTS["products.csv"]):
        department, category, subcategory = TAXONOMY[index % len(TAXONOMY)]
        brand = BRANDS[(index * 3) % len(BRANDS)]
        quality = QUALITIES[(index * 5) % len(QUALITIES)]
        band = bands[(index // len(TAXONOMY)) % len(bands)]
        product_name = f"{subcategory} {quality} {index + 1:03d}"
        description = (
            f"{product_name} da marca {brand}, criado para uso cotidiano com acabamento nacional."
        )
        rows.append(
            (
                f"PROD{index + 1:05d}",
                product_name,
                department,
                category,
                subcategory,
                brand,
                band,
                description,
            )
        )
    return rows


def _purchase_rows(
    rng: random.Random,
    customers: Sequence[Sequence[Any]],
    affinities: Sequence[tuple[str, str]],
    products: Sequence[Sequence[Any]],
) -> list[tuple[Any, ...]]:
    by_department: dict[str, list[int]] = {}
    by_pair: dict[tuple[str, str], list[int]] = {}
    for index, product in enumerate(products):
        by_department.setdefault(str(product[2]), []).append(index)
        by_pair.setdefault((str(product[2]), str(product[3])), []).append(index)

    start = date(2024, 1, 1)
    days = 731
    customer_weights = [1.0 / ((index + 5) ** 0.55) for index in range(len(customers))]
    product_rank = {index: rank for indices in by_department.values() for rank, index in enumerate(indices)}
    rows: list[tuple[Any, ...]] = []

    def add_purchase(customer_index: int, serial: int) -> None:
        # Recency trend plus fixed retail peaks in June, November, and December.
        while True:
            day = int((rng.random() ** 0.88) * days)
            purchased_on = start + timedelta(days=min(day, days - 1))
            seasonal = 1.8 if purchased_on.month in {6, 11, 12} else 1.0
            if rng.random() <= seasonal / 1.8:
                break
        preferred_department, preferred_category = affinities[customer_index]
        signal = rng.random()
        if signal < 0.72:
            candidates = by_pair[(preferred_department, preferred_category)]
        elif signal < 0.90:
            candidates = by_department[preferred_department]
        elif signal < 0.96 and purchased_on.year == 2025:
            candidates = by_department["Eletrônicos"] + by_department["Games"]
        else:
            candidates = list(range(len(products)))
        weights = []
        for product_index in candidates:
            rank = product_rank[product_index]
            long_tail = 1.0 / ((rank + 1) ** 1.18)
            trend = 1.0 + (0.9 * (purchased_on - start).days / (days - 1))
            season = 1.0
            department = products[product_index][2]
            if purchased_on.month in {11, 12} and department in {"Eletrônicos", "Games"}:
                season = 1.8
            elif purchased_on.month in {1, 6, 7} and department in {"Livros", "Esporte"}:
                season = 1.5
            weights.append(long_tail * trend * season)
        product_index = rng.choices(candidates, weights=weights, k=1)[0]
        product = products[product_index]
        low, high = PRICE_RANGES[str(product[6])]
        cents = rng.randint(int(low * 100), int(high * 100))
        price = Decimal(cents) / 100
        rows.append(
            (
                purchased_on.isoformat(),
                customers[customer_index][0],
                product[0],
                f"{price:.2f}",
                1 if rng.random() < 0.38 else 2,
                serial,
            )
        )

    # Every customer interacts; remaining events follow a long-tail activity distribution.
    for customer_index in range(len(customers)):
        add_purchase(customer_index, customer_index)
    for serial in range(len(customers), DATASET_COUNTS["purchases.csv"]):
        customer_index = rng.choices(range(len(customers)), weights=customer_weights, k=1)[0]
        add_purchase(customer_index, serial)
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[5]))
    return [row[:5] for row in rows]


def generate_dataset(output_dir: Path | str, *, seed: int = DATASET_SEED) -> dict[str, Any]:
    """Generate byte-stable UTF-8 CSV files and their complete metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    customers, affinities = _customer_rows(rng)
    products = _product_rows()
    purchases = _purchase_rows(rng, customers, affinities, products)
    _write_csv(output / "customers.csv", DATASET_HEADERS["customers.csv"], customers)
    _write_csv(output / "products.csv", DATASET_HEADERS["products.csv"], products)
    _write_csv(output / "purchases.csv", DATASET_HEADERS["purchases.csv"], purchases)
    result = inspect_synthetic(output, expected_counts=DATASET_COUNTS)
    _write_json(output / "dataset_manifest.json", result["manifest"])
    _write_json(output / "dataset_stats.json", result["stats"])
    return result


def require_dataset_files(raw_dir: Path | str) -> dict[str, Path]:
    root = Path(raw_dir)
    missing = [name for name in DATASET_FILES if not (root / name).is_file()]
    if missing:
        raise DatasetValidationError(
            f"arquivos obrigatórios ausentes: {', '.join(missing)}", file=root
        )
    return {name: root / name for name in DATASET_FILES}


def _reader(path: Path) -> tuple[Any, csv.DictReader[str]]:
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(handle)
    actual = tuple(reader.fieldnames or ())
    expected = DATASET_HEADERS[path.name]
    if actual != expected:
        handle.close()
        raise DatasetValidationError(
            f"schema inválido: esperado {expected}, recebido {actual}", file=path, row=1
        )
    return handle, reader


def _required(value: str | None, path: Path, row: int, column: str) -> str:
    raw = value or ""
    if not raw.strip():
        raise DatasetValidationError("não pode ser vazio", file=path, row=row, column=column)
    return raw.strip()


def _integer(value: str | None, path: Path, row: int, column: str) -> int:
    raw = _required(value, path, row, column)
    try:
        return int(raw)
    except ValueError as exc:
        raise DatasetValidationError("deve ser inteiro", file=path, row=row, column=column) from exc


def _purchase_date(value: str | None, path: Path, row: int) -> date:
    raw = _required(value, path, row, "purchased_on")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise DatasetValidationError("deve ser data ISO válida", file=path, row=row, column="purchased_on") from exc
    if parsed.isoformat() != raw:
        raise DatasetValidationError("deve usar YYYY-MM-DD", file=path, row=row, column="purchased_on")
    return parsed


def _price(value: str | None, path: Path, row: int) -> Decimal:
    raw = _required(value, path, row, "price")
    try:
        parsed = Decimal(raw)
    except InvalidOperation as exc:
        raise DatasetValidationError("deve ser decimal", file=path, row=row, column="price") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed.as_tuple().exponent < -2:
        raise DatasetValidationError("deve ser valor BRL positivo com até 2 casas", file=path, row=row, column="price")
    return parsed


def _channel(value: str | None, path: Path, row: int) -> int:
    channel = _integer(value, path, row, "sales_channel_id")
    if channel not in (1, 2):
        raise DatasetValidationError("deve ser 1 ou 2", file=path, row=row, column="sales_channel_id")
    return channel


def _checksum(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _fingerprint(source_files: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name in DATASET_FILES:
        value = source_files[name]
        digest.update(f"{name}\0{value['bytes']}\0{value['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def inspect_synthetic(
    raw_dir: Path | str,
    *,
    expected_counts: Mapping[str, int] | None = DATASET_COUNTS,
    top_items_limit: int = 20,
) -> dict[str, Any]:
    paths = require_dataset_files(raw_dir)
    customer_ids: set[str] = set()
    article_ids: set[str] = set()
    product_names: dict[str, str] = {}
    profiles: set[str] = set()
    age_counts: Counter[str] = Counter()
    file_rows: dict[str, int] = {}
    customer_names: list[str] = []

    handle, reader = _reader(paths["customers.csv"])
    try:
        for row_number, row in enumerate(reader, 2):
            customer_id = _required(row["customer_id"], paths["customers.csv"], row_number, "customer_id")
            profile_id = _required(row["profile_id"], paths["customers.csv"], row_number, "profile_id")
            name = _required(row["name"], paths["customers.csv"], row_number, "name")
            age = _integer(row["age"], paths["customers.csv"], row_number, "age")
            if customer_id in customer_ids or profile_id in profiles:
                raise DatasetValidationError("cliente ou perfil duplicado", file=paths["customers.csv"], row=row_number)
            if not 18 <= age <= 80 or row["age_band"] != age_band(age):
                raise DatasetValidationError("idade ou faixa etária inválida", file=paths["customers.csv"], row=row_number)
            customer_ids.add(customer_id)
            profiles.add(profile_id)
            customer_names.append(name)
            age_counts[row["age_band"]] += 1
        file_rows["customers.csv"] = len(customer_ids)
    finally:
        handle.close()

    handle, reader = _reader(paths["products.csv"])
    try:
        for row_number, row in enumerate(reader, 2):
            article_id = _required(row["article_id"], paths["products.csv"], row_number, "article_id")
            if article_id in article_ids:
                raise DatasetValidationError("produto duplicado", file=paths["products.csv"], row=row_number)
            for column in DATASET_HEADERS["products.csv"][1:]:
                _required(row[column], paths["products.csv"], row_number, column)
            if row["price_band"] not in PRICE_RANGES:
                raise DatasetValidationError("faixa de preço inválida", file=paths["products.csv"], row=row_number)
            article_ids.add(article_id)
            product_names[article_id] = row["product_name"]
        file_rows["products.csv"] = len(article_ids)
    finally:
        handle.close()

    interacting_customers: set[str] = set()
    interacting_articles: set[str] = set()
    customer_counts: Counter[str] = Counter()
    article_counts: Counter[str] = Counter()
    minimum: date | None = None
    maximum: date | None = None
    handle, reader = _reader(paths["purchases.csv"])
    try:
        for row_number, row in enumerate(reader, 2):
            customer_id = _required(row["customer_id"], paths["purchases.csv"], row_number, "customer_id")
            article_id = _required(row["article_id"], paths["purchases.csv"], row_number, "article_id")
            if customer_id not in customer_ids or article_id not in article_ids:
                raise DatasetValidationError("chave estrangeira desconhecida", file=paths["purchases.csv"], row=row_number)
            purchased_on = _purchase_date(row["purchased_on"], paths["purchases.csv"], row_number)
            _price(row["price"], paths["purchases.csv"], row_number)
            _channel(row["sales_channel_id"], paths["purchases.csv"], row_number)
            interacting_customers.add(customer_id)
            interacting_articles.add(article_id)
            customer_counts[customer_id] += 1
            article_counts[article_id] += 1
            minimum = purchased_on if minimum is None else min(minimum, purchased_on)
            maximum = purchased_on if maximum is None else max(maximum, purchased_on)
        file_rows["purchases.csv"] = sum(customer_counts.values())
    finally:
        handle.close()
    if minimum is None or maximum is None:
        raise DatasetValidationError("deve conter compras", file=paths["purchases.csv"])
    if expected_counts is not None:
        for name, expected in expected_counts.items():
            if file_rows.get(name) != expected:
                raise DatasetValidationError(
                    f"contagem inválida: esperado {expected}, recebido {file_rows.get(name, 0)}",
                    file=paths[name],
                )

    source_files = {name: {"rows": file_rows[name], **_checksum(paths[name])} for name in DATASET_FILES}
    fingerprint = _fingerprint(source_files)
    purchases = file_rows["purchases.csv"]
    manifest = {
        "schema_version": 2,
        "dataset": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "locale": "pt-BR",
        "currency": "BRL",
        "synthetic": True,
        "sampled": False,
        "seed": DATASET_SEED,
        "source": "gerador determinístico do projeto",
        "source_fingerprint": fingerprint,
        "source_files": source_files,
        "files": file_rows,
        "total_records": sum(file_rows.values()),
        "catalog_customers": len(customer_ids),
        "catalog_articles": len(article_ids),
        "interacting_customers": len(interacting_customers),
        "interacting_articles": len(interacting_articles),
        "purchases": purchases,
        "date_range": {"start": minimum.isoformat(), "end": maximum.isoformat()},
        "semantics": "purchase_only",
    }
    counts = list(customer_counts.values())
    stats = {
        "dataset_name": DATASET_NAME,
        "currency": "BRL",
        "source_fingerprint": fingerprint,
        "catalog_customers": len(customer_ids),
        "catalog_articles": len(article_ids),
        "interacting_customers": len(interacting_customers),
        "interacting_articles": len(interacting_articles),
        "purchases": purchases,
        "average_purchases_per_interacting_customer": purchases / len(interacting_customers),
        "median_purchases_per_customer": float(sorted(counts)[len(counts) // 2]),
        "purchases_per_customer": {
            "0": len(customer_ids) - len(counts),
            "1": sum(value == 1 for value in counts),
            "2-5": sum(2 <= value <= 5 for value in counts),
            "6-10": sum(6 <= value <= 10 for value in counts),
            "11-50": sum(11 <= value <= 50 for value in counts),
            "51-100": sum(51 <= value <= 100 for value in counts),
            "101+": sum(value >= 101 for value in counts),
        },
        "age_band_counts": dict(sorted(age_counts.items())),
        "date_range": manifest["date_range"],
        "top_articles": [
            {
                "article_id": article_id,
                "product_name": product_names[article_id],
                "purchases": count,
            }
            for article_id, count in sorted(article_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:top_items_limit]
        ],
        "personas": customer_names[: len(PERSONAS)],
        "semantics": "purchase_only",
    }
    return {"manifest": manifest, "stats": stats}


def write_dataset_artifacts(raw_dir: Path | str) -> dict[str, Any]:
    result = inspect_synthetic(raw_dir)
    root = Path(raw_dir)
    _write_json(root / "dataset_manifest.json", result["manifest"])
    _write_json(root / "dataset_stats.json", result["stats"])
    return result


def _write_checked(
    sink: BulkImportSink, table: str, columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> int:
    written = sink.write_rows(table, columns, rows)
    if written != len(rows):
        raise DatasetError(f"importação parcial em {table}: enviados {len(rows)}, gravados {written}")
    return written


def import_synthetic(
    raw_dir: Path | str,
    sink: BulkImportSink,
    *,
    expected_counts: Mapping[str, int] | None = DATASET_COUNTS,
    chunk_size: int = 10_000,
) -> dict[str, dict[str, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    paths = require_dataset_files(raw_dir)
    customer_keys: dict[str, int] = {}
    article_keys: dict[str, int] = {}
    source_rows = {name: 0 for name in DATASET_FILES}
    table_rows = {"customers": 0, "articles": 0, "purchases": 0}
    sink.begin()
    try:
        handle, reader = _reader(paths["customers.csv"])
        try:
            rows: list[tuple[Any, ...]] = []
            for key, row in enumerate(reader, 1):
                customer_id = row["customer_id"]
                customer_keys[customer_id] = key
                rows.append((key, *(row[column] for column in DATASET_HEADERS["customers.csv"][:3]), int(row["age"]), *(row[column] for column in DATASET_HEADERS["customers.csv"][4:])))
                if len(rows) == chunk_size:
                    table_rows["customers"] += _write_checked(sink, "customers", CUSTOMER_COLUMNS, rows)
                    rows.clear()
            if rows:
                table_rows["customers"] += _write_checked(sink, "customers", CUSTOMER_COLUMNS, rows)
            source_rows["customers.csv"] = len(customer_keys)
        finally:
            handle.close()

        handle, reader = _reader(paths["products.csv"])
        try:
            rows = []
            for key, row in enumerate(reader, 1):
                article_keys[row["article_id"]] = key
                rows.append((key, *(row[column] for column in DATASET_HEADERS["products.csv"])))
                if len(rows) == chunk_size:
                    table_rows["articles"] += _write_checked(sink, "articles", ARTICLE_COLUMNS, rows)
                    rows.clear()
            if rows:
                table_rows["articles"] += _write_checked(sink, "articles", ARTICLE_COLUMNS, rows)
            source_rows["products.csv"] = len(article_keys)
        finally:
            handle.close()

        handle, reader = _reader(paths["purchases.csv"])
        try:
            rows = []
            for row_number, row in enumerate(reader, 2):
                rows.append(
                    (
                        _purchase_date(row["purchased_on"], paths["purchases.csv"], row_number),
                        customer_keys[row["customer_id"]],
                        article_keys[row["article_id"]],
                        _price(row["price"], paths["purchases.csv"], row_number),
                        _channel(row["sales_channel_id"], paths["purchases.csv"], row_number),
                    )
                )
                if len(rows) == chunk_size:
                    table_rows["purchases"] += _write_checked(sink, "purchases", PURCHASE_COLUMNS, rows)
                    rows.clear()
            if rows:
                table_rows["purchases"] += _write_checked(sink, "purchases", PURCHASE_COLUMNS, rows)
            source_rows["purchases.csv"] = table_rows["purchases"]
        finally:
            handle.close()
        if expected_counts is not None:
            for name, expected in expected_counts.items():
                if source_rows[name] != expected:
                    raise DatasetValidationError(
                        f"contagem inválida: esperado {expected}, recebido {source_rows[name]}", file=paths[name]
                    )
        sink.commit()
    except BaseException:
        sink.rollback()
        raise
    return {"source_rows": source_rows, "table_rows": table_rows}
