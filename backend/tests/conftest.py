from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./backend/tests/test.db"
os.environ["SSE_POLL_INTERVAL_SECONDS"] = "0.01"
os.environ["REQUIRE_FULL_DATASET"] = "false"
os.environ["POPULARITY_PATH"] = "/tmp/ecommerce-recommendation-lab-test-popularity.json"

from app.database.base import Base
from app.database.session import configure_database, get_session
from app.main import app


@pytest.fixture(autouse=True)
def clean_database(tmp_path: pytest.TempPathFactory) -> Iterator[None]:
    engine = configure_database(f"sqlite:///{tmp_path}/api.db")
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db():
    with get_session() as session:
        yield session


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
