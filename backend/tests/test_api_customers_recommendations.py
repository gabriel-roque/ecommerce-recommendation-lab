import json
from datetime import date, datetime, timezone

import pytest

from app.api.recommendation_backend import (
    BackendRecommendationResult,
    RecommendationUnavailable,
    get_recommendation_backend,
)
from app.api import routes
from app.database.models import Article, Customer, ModelVersion, Purchase, TrainingRun
from app.main import app
from app.models.mappings import DenseMappings


class FakeRecommendationBackend:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def recommend(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def article(key: int, article_id: str, name: str, department: str) -> Article:
    return Article(
        article_key=key,
        article_id=article_id,
        product_name=name,
        department_name=department,
        category_name=department,
        subcategory_name="Produto",
        brand="Aurora",
        price_band="intermediário",
        detail_desc=f"{name} para uso cotidiano.",
    )


def seed_dataset(db):
    db.add_all(
        [
            Customer(customer_key=1, customer_id="source-customer-secret", profile_id="profile-one", name="Lucas Almeida", age=31, age_band="25-34", state="SP", region="Sudeste"),
            Customer(customer_key=2, customer_id="another-source-secret", profile_id="profile-two", name="Mariana Souza", age=47, age_band="45-54", state="PE", region="Nordeste"),
            Customer(customer_key=3, customer_id="cold-source-secret", profile_id="profile-three", name="Rafael Oliveira", age=22, age_band="18-24", state="RS", region="Sul"),
            article(10, "0000000010", "Camiseta", "Moda"),
            article(11, "0000000011", "Jaqueta", "Moda"),
            article(12, "0000000012", "Notebook", "Eletrônicos"),
            article(13, "0000000013", "Tênis", "Moda"),
            Purchase(t_dat=date(2020, 1, 1), customer_key=1, article_key=10, price=0.02, sales_channel_id=1),
            Purchase(t_dat=date(2020, 1, 2), customer_key=1, article_key=11, price=0.03, sales_channel_id=1),
            Purchase(t_dat=date(2020, 1, 3), customer_key=2, article_key=12, price=0.04, sales_channel_id=2),
            Purchase(t_dat=date(2020, 1, 4), customer_key=2, article_key=12, price=0.04, sales_channel_id=2),
            Purchase(t_dat=date(2020, 1, 5), customer_key=2, article_key=13, price=0.05, sales_channel_id=2),
        ]
    )
    db.commit()


def active_model(db, artifact_path: str) -> ModelVersion:
    run = TrainingRun(status="COMPLETED", config={"data_profile": "balanced"}, progress=1)
    db.add(run)
    db.flush()
    model = ModelVersion(
        version="ecommerce-v1",
        training_run_id=run.id,
        artifact_path=artifact_path,
        status="READY",
        is_active=True,
        config={"data_profile": "balanced"},
        activated_at=datetime.now(timezone.utc),
    )
    db.add(model)
    db.commit()
    return model


def test_customer_contract_hides_source_id_and_exact_age(client, db):
    seed_dataset(db)

    profile = client.get("/customers/profile-one")
    history = client.get("/customers/profile-one/history?limit=1&offset=1")

    assert profile.status_code == 200
    assert profile.json() == {
        "profile_id": "profile-one",
        "name": "Lucas Almeida",
        "age_band": "25-34",
        "state": "SP",
        "region": "Sudeste",
        "purchase_count": 2,
        "first_purchase_date": "2020-01-01",
        "last_purchase_date": "2020-01-02",
        "personalizable": True,
    }
    assert "source-customer-secret" not in profile.text
    assert "31" not in profile.text
    assert history.status_code == 200
    assert history.json()["total"] == 2
    assert history.json()["limit"] == 1
    assert history.json()["offset"] == 1
    assert history.json()["items"][0]["article_id"] == "0000000010"
    assert history.json()["items"][0]["price"] == 0.02
    assert history.json()["items"][0]["sales_channel_id"] == 1
    assert "available" not in history.text


def test_random_customer_honors_personalizable_filter(client, db):
    db.add(Customer(customer_key=3, customer_id="cold-source-secret", profile_id="profile-three", name="Rafael Oliveira", age=22, age_band="18-24", state="RS", region="Sul"))
    db.commit()

    assert client.get("/customers/random?personalizable=true").status_code == 404
    response = client.get("/customers/random?personalizable=false")
    assert response.status_code == 200
    assert response.json()["profile_id"] == "profile-three"
    assert response.json()["personalizable"] is False


def test_random_customer_can_be_limited_to_active_model_profiles(client, db, tmp_path, monkeypatch):
    seed_dataset(db)
    artifact = tmp_path / "artifact"
    DenseMappings.build([2], [10]).save(artifact)
    active_model(db, str(artifact))
    monkeypatch.setattr(routes, "_assert_model_dataset_matches", lambda db, model: None)

    response = client.get(
        "/customers/random?personalizable=true&model_eligible=true"
    )

    assert response.status_code == 200
    assert response.json()["profile_id"] == "profile-two"


def test_model_eligible_random_customer_requires_active_model(client, db):
    seed_dataset(db)

    response = client.get("/customers/random?model_eligible=true")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "No active model available for model-eligible profile selection"
    )


def test_popular_recommendations_are_generic_and_explain_scores(client, db):
    seed_dataset(db)

    response = client.get("/recommendations/popular?k=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile_id"] is None
    assert payload["strategy"] == "popularity"
    assert payload["fallback_reason"] is None
    assert payload["model"] is None
    assert payload["score_semantics"]["name"] == "normalized_purchase_popularity"
    assert "probabilidade" in payload["score_semantics"]["description"]
    assert [item["article_id"] for item in payload["items"]] == ["0000000012", "0000000010"]
    assert all(isinstance(item["article_id"], str) for item in payload["items"])
    assert payload["items"][0]["evidence"][0] == {
        "kind": "global_popularity",
        "description": "Produto registrou 2 compras no dataset.",
        "purchase_count": 2,
        "customer_count": 1,
        "related_article_ids": [],
    }


def test_customer_recommendations_report_fallback_and_exclude_history(client, db):
    seed_dataset(db)

    response = client.get("/customers/profile-one/recommendations?k=3")

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "popularity"
    assert payload["fallback_reason"] == "no_active_model"
    assert payload["exclude_seen_articles"] is True
    assert {item["article_id"] for item in payload["items"]}.isdisjoint(
        {"0000000010", "0000000011"}
    )


def test_model_recommendations_expose_weighted_breakdown(client, db, tmp_path, monkeypatch):
    seed_dataset(db)
    active_model(db, str(tmp_path / "missing-artifact"))
    backend = FakeRecommendationBackend(
        BackendRecommendationResult(strategy="neural", items=[(13, 0.7), (12, 0.9), (12, 0.8)])
    )
    app.dependency_overrides[get_recommendation_backend] = lambda: backend
    monkeypatch.setattr(routes, "_assert_model_dataset_matches", lambda db, model: None)
    try:
        response = client.get("/customers/profile-one/recommendations?k=2")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert backend.calls[0]["customer_key"] == 1
    assert backend.calls[0]["exclude_item_ids"] == {10, 11}
    assert payload["strategy"] == "model"
    assert payload["fallback_reason"] is None
    assert payload["model"]["version"] == "ecommerce-v1"
    assert payload["model"]["data_profile"] == "balanced"
    assert payload["score_semantics"]["name"] == "hybrid_ranking_score"
    assert payload["applied_weights"] == {"model": 45.0, "category": 30.0, "popularity": 15.0, "trend": 10.0}
    assert [item["article_id"] for item in payload["items"]] == ["0000000013", "0000000012"]
    first = payload["items"][0]
    assert first["score"] == pytest.approx(sum(part["contribution"] for part in first["breakdown"].values()))
    assert first["score"] <= 100
    assert "probabilidade" in payload["score_semantics"]["description"]
    assert first["evidence"][1]["kind"] == "purchase_history_department"
    assert first["evidence"][1]["related_article_ids"] == [
        "0000000010",
        "0000000011",
    ]


def test_model_unavailable_uses_explicit_popularity_fallback(client, db, tmp_path, monkeypatch):
    seed_dataset(db)
    active_model(db, str(tmp_path / "missing-artifact"))
    backend = FakeRecommendationBackend(error=RecommendationUnavailable("not loaded"))
    app.dependency_overrides[get_recommendation_backend] = lambda: backend
    monkeypatch.setattr(routes, "_assert_model_dataset_matches", lambda db, model: None)
    try:
        response = client.get("/customers/profile-one/recommendations?k=2")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["strategy"] == "popularity"
    assert response.json()["fallback_reason"] == "model_unavailable"
    assert response.json()["applied_weights"]["popularity"] == 100


def test_recommendation_weights_must_sum_to_100(client, db):
    seed_dataset(db)
    response = client.get(
        "/customers/profile-one/recommendations?model_weight=40&category_weight=30&popularity_weight=15&trend_weight=10"
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "A soma dos quatro pesos deve ser 100"


def test_dataset_artifact_mismatch_is_rejected_before_inference(client, db, tmp_path):
    seed_dataset(db)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "ecommerce_sintetico_br",
                "semantics": "purchase_only",
                "sampled": False,
                "source_fingerprint": "antigo",
                "purchases": 999,
                "catalog_customers": 3,
                "catalog_articles": 4,
            }
        ),
        encoding="utf-8",
    )
    active_model(db, str(artifact))
    backend = FakeRecommendationBackend(
        BackendRecommendationResult(strategy="neural", items=[(12, 0.9)])
    )
    app.dependency_overrides[get_recommendation_backend] = lambda: backend
    try:
        response = client.get("/customers/profile-one/recommendations")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert response.json()["detail"].startswith("Modelo ativo não corresponde ao dataset atual")
    assert backend.calls == []


def test_openapi_exposes_generic_public_fields(client):
    openapi = client.get("/openapi.json").json()
    customer_properties = openapi["components"]["schemas"]["CustomerResponse"]["properties"]
    history_properties = openapi["components"]["schemas"]["CustomerHistoryItem"]["properties"]
    assert "customer_id" not in customer_properties
    assert "age" not in customer_properties
    assert "name" in customer_properties
    assert "price" in history_properties
    assert "availability" not in history_properties
