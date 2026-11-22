"""Requires PostgreSQL + pgvector with migrations applied: `make test-integration`."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Document, DocumentChunk, QueryLog
from app.db.session import build_engine, build_session_factory
from app.rag.embeddings import HashingEmbeddings
from app.rag.ingest import ingest_file
from app.rag.retriever import PgVectorRetriever
from app.services.audit import AuditRecord, DbAuditSink

pytestmark = pytest.mark.integration

DOCS = Path(__file__).parent / "documents.jsonl"


@pytest.fixture
def settings() -> Settings:
    settings = Settings(llm_provider="fake")
    # The fixture wipes tables, so never point it at the demo database.
    assert settings.database_url.rsplit("/", 1)[-1].endswith("_test"), "use a *_test database"
    return settings


@pytest.fixture
def factory(settings: Settings) -> Iterator[sessionmaker[Session]]:
    engine = build_engine(settings.database_url)
    factory = build_session_factory(engine)
    with factory.begin() as s:
        s.execute(delete(QueryLog))
        s.execute(delete(Document))
    yield factory
    engine.dispose()


def _count(factory: sessionmaker[Session], model: type) -> int:
    with factory() as s:
        return int(s.scalar(select(func.count()).select_from(model)) or 0)


def test_ingestion_is_idempotent(factory: sessionmaker[Session], settings: Settings) -> None:
    emb = HashingEmbeddings(settings.embedding_dimensions)
    first = ingest_file(DOCS, factory, emb, settings)
    chunks_after_first = _count(factory, DocumentChunk)
    second = ingest_file(DOCS, factory, emb, settings)

    assert (first.inserted, first.failed) == (3, 0)
    assert (second.inserted, second.updated, second.skipped) == (0, 0, 3)
    assert second.chunks_embedded == 0
    assert _count(factory, DocumentChunk) == chunks_after_first


def test_new_version_replaces_old_chunks(
    factory: sessionmaker[Session], settings: Settings, tmp_path: Path
) -> None:
    emb = HashingEmbeddings(settings.embedding_dimensions)
    ingest_file(DOCS, factory, emb, settings)
    bumped = DOCS.read_text(encoding="utf-8").replace('"version": 1', '"version": 2', 1)
    path = tmp_path / "docs.jsonl"
    path.write_text(bumped, encoding="utf-8")

    report = ingest_file(path, factory, emb, settings)
    assert (report.updated, report.skipped) == (1, 2)
    with factory() as s:
        keys = s.scalars(select(DocumentChunk.chunk_key)).all()
    assert not any(k.startswith("HR-TEST-001:1:") for k in keys)
    assert any(k.startswith("HR-TEST-001:2:") for k in keys)


def test_retrieval_respects_scopes(factory: sessionmaker[Session], settings: Settings) -> None:
    emb = HashingEmbeddings(settings.embedding_dimensions)
    ingest_file(DOCS, factory, emb, settings)
    retriever = PgVectorRetriever(factory, emb, top_k=5)

    assert retriever.retrieve("ลาป่วยยื่นตรงไหน", ["public"]) == []
    hits = retriever.retrieve("ลาป่วยยื่นตรงไหน", ["internal"])
    assert hits
    assert {h["access_level"] for h in hits} == {"internal"}
    assert any(h["document_id"] == "HR-TEST-001" for h in hits)


def test_audit_record_persists(factory: sessionmaker[Session]) -> None:
    sink = DbAuditSink(factory)
    sink.record(
        AuditRecord(
            request_id="req_integration_1",
            actor_hash="a" * 64,
            query_hash="b" * 64,
            redacted_query=None,
            status="refused",
            reason_code="PROMPT_INJECTION",
            best_relevance=None,
            model=None,
            latency_ms=1.0,
        )
    )
    assert _count(factory, QueryLog) == 1
