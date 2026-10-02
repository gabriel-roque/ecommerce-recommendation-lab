import json
from datetime import date

from app.api import routes
from app.config import Settings
from app.database.models import Article, Customer, Purchase


def article(key: int, article_id: str) -> Article:
    return Article(
        article_key=key,
        article_id=article_id,
        product_name="Produto",
        department_name="Casa",
        category_name="Cozinha",
        subcategory_name="Utensílio",
        brand="Aurora",
        price_band="econômico",
        detail_desc="Produto brasileiro para uso cotidiano.",
    )


def seed_dataset(db):
    db.add_all(
        [
            Customer(customer_key=1, customer_id="source-1", profile_id="profile-1", name="Lucas Almeida", age=31, age_band="25-34", state="SP", region="Sudeste"),
            Customer(customer_key=2, customer_id="source-2", profile_id="profile-2", name="Mariana Souza", age=40, age_band="35-44", state="PE", region="Nordeste"),
            Customer(customer_key=3, customer_id="source-3", profile_id="profile-3", name="Rafael Oliveira", age=51, age_band="45-54", state="RS", region="Sul"),
            article(10, "0000000010"),
            article(11, "0000000011"),
            article(12, "0000000012"),
            Purchase(t_dat=date(2020, 1, 1), customer_key=1, article_key=10, price=0.01, sales_channel_id=1),
            Purchase(t_dat=date(2020, 1, 2), customer_key=1, article_key=10, price=0.01, sales_channel_id=1),
            Purchase(t_dat=date(2020, 1, 3), customer_key=2, article_key=11, price=0.02, sales_channel_id=2),
        ]
    )
    db.commit()


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_cors_allows_local_frontend_private_network(client):
    response = client.options(
        "/dataset/stats",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
            "Access-Control-Request-Private-Network": "true",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.headers["access-control-allow-private-network"] == "true"


def test_dataset_stats_use_synthetic_purchase_contract(client, db):
    seed_dataset(db)

    response = client.get("/dataset/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_name"] == "E-commerce Recommendation Lab"
    assert payload["locale"] == "pt-BR"
    assert payload["currency"] == "BRL"
    assert payload["semantics"] == "purchase_history"
    assert payload["total_purchases"] == 3
    assert payload["unique_customers"] == 3
    assert payload["customers_with_purchases"] == 2
    assert payload["unique_articles"] == 3
    assert payload["purchased_articles"] == 2
    assert payload["average_purchases_per_customer"] == 1.0
    assert payload["median_purchases_per_customer"] == 1.0
    assert payload["purchases_per_customer"] == {
        "0": 1,
        "1": 1,
        "2-5": 1,
        "6-10": 0,
        "11-50": 0,
        "51-100": 0,
        "101+": 0,
    }
    assert payload["age_band_counts"] == {"25-34": 1, "35-44": 1, "45-54": 1}
    assert payload["top_articles"][0] == {
        "article_id": "0000000010",
        "product_name": "Produto",
        "purchases": 2,
    }
    assert payload["date_range"] == {
        "start": "2020-01-01",
        "end": "2020-01-03",
    }
    assert payload["sampled"] is True
    for retail_field in ("views", "add_to_cart", "events_by_type", "top_items"):
        assert retail_field not in payload


def test_dataset_stats_only_trust_matching_synthetic_manifest(client, db, tmp_path, monkeypatch):
    seed_dataset(db)
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "ecommerce_sintetico_br",
                "dataset_name": "E-commerce Recommendation Lab",
                "semantics": "purchase_only",
                "synthetic": True,
                "currency": "BRL",
                "source_fingerprint": "abc",
                "catalog_customers": 3,
                "catalog_articles": 3,
                "purchases": 3,
                "sampled": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: Settings(require_full_dataset=False, dataset_manifest_path=str(manifest)),
    )

    assert client.get("/dataset/stats").json()["sampled"] is False

    manifest.write_text(
        json.dumps(
            {
                "dataset": "ecommerce_sintetico_br",
                "semantics": "purchase_only",
                "synthetic": True,
                "currency": "BRL",
                "source_fingerprint": "abc",
                "catalog_customers": 3,
                "catalog_articles": 3,
                "purchases": 4,
                "sampled": False,
            }
        ),
        encoding="utf-8",
    )
    assert client.get("/dataset/stats").json()["sampled"] is True
