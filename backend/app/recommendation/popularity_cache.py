"""Shared process and disk cache for purchase popularity."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import Purchase

from .popularity import PopularityRecommender


_cache: dict[tuple[str, str], PopularityRecommender] = {}
_lock = Lock()

def _source_key(db: Session) -> str:
    settings = get_settings()
    manifest_path = Path(settings.dataset_manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            isinstance(manifest, dict)
            and manifest.get("dataset") == "ecommerce_sintetico_br"
            and manifest.get("semantics") == "purchase_only"
        ):
            source = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        else:
            raise ValueError("Popularidade requer manifesto completo do dataset sintético")
    except OSError:
        source = str(db.get_bind().url)
    except json.JSONDecodeError as exc:
        raise ValueError("Popularidade requer manifesto válido") from exc
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _load(path: Path, source_key: str) -> PopularityRecommender | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if payload.get("version") != 3 or payload.get("source_fingerprint") != source_key:
                return None
            rows = payload.get("articles")
        if not isinstance(rows, list) or not rows:
            return None
        item_ids = np.asarray([row["article_key"] for row in rows], dtype=np.int64)
        scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
        if scores[0] <= 0:
            return None
        return PopularityRecommender(item_ids, scores / scores[0])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _build(db: Session) -> PopularityRecommender:
    purchase_count = func.count(Purchase.id).label("score")
    rows = db.execute(
        select(Purchase.article_key, purchase_count)
        .group_by(Purchase.article_key)
        .order_by(desc(purchase_count), Purchase.article_key)
    ).all()
    if not rows:
        raise ValueError("Não é possível calcular popularidade sem compras")
    item_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    raw_scores = np.asarray([row[1] for row in rows], dtype=np.float64)
    return PopularityRecommender(item_ids, raw_scores / raw_scores[0])


def _write(path: Path, source_key: str, popularity: PopularityRecommender) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload: dict[str, Any] = {
        "version": 3,
        "source_fingerprint": source_key,
        "articles": [
            {"article_key": int(item), "score": float(score)}
            for item, score in zip(popularity.article_keys, popularity.scores, strict=True)
        ],
    }
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def get_popularity(db: Session) -> PopularityRecommender:
    settings = get_settings()
    path = Path(settings.popularity_path)
    source_key = _source_key(db)
    key = (str(path.resolve()), source_key)
    with _lock:
        popularity = _cache.get(key)
        if popularity is not None:
            return popularity
        popularity = _load(path, source_key)
        if popularity is None:
            popularity = _build(db)
            _write(path, source_key, popularity)
        _cache.clear()
        _cache[key] = popularity
        return popularity
