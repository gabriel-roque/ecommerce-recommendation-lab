from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.database.base import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_database_url: str | None = None


def configure_database(database_url: str | None = None) -> Engine:
    global _engine, _session_factory, _database_url

    url = database_url or get_settings().database_url
    if _engine is not None and _database_url == url:
        return _engine
    if _engine is not None:
        _engine.dispose()

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
    _database_url = url
    _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)

    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_engine() -> Engine:
    return _engine or configure_database()


def create_schema() -> None:
    # Import registers every mapped table before metadata creation.
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())


def get_session() -> Session:
    if _session_factory is None:
        configure_database()
    assert _session_factory is not None
    return _session_factory()


def get_db() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()
