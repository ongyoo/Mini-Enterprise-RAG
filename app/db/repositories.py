import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import (
    Document,
    DocumentChunk,
    EvaluationRun,
    IngestionRun,
    QueryLog,
    RetrievalHit,
)


@dataclass(frozen=True, slots=True)
class ChunkSearchRow:
    chunk_id: uuid.UUID
    chunk_key: str
    content: str
    document_code: str
    title: str
    version: int
    type: str
    access_level: str
    source_authority: str
    related_document_codes: list[str]
    similarity: float


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def get_by_code(self, code: str) -> Document | None:
        return self._s.scalar(select(Document).where(Document.document_code == code))

    def topics(self, scopes: Sequence[str]) -> dict[str, list[str]]:
        """Titles of active SOPs the caller may see, grouped by category."""
        grouped: dict[str, list[str]] = {}
        for doc in self.list_visible(scopes):
            if doc.type == "SOP":
                grouped.setdefault(doc.category, []).append(doc.title)
        return grouped

    def list_visible(self, scopes: Sequence[str]) -> Sequence[Document]:
        stmt = (
            select(Document)
            .where(Document.status == "active", Document.access_level.in_(scopes))
            .order_by(Document.document_code)
        )
        return self._s.scalars(stmt).all()

    def replace_chunks(self, document: Document, chunks: Sequence[DocumentChunk]) -> None:
        self._s.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        self._s.flush()
        self._s.add_all(chunks)

    def search_chunks(
        self,
        embedding: Sequence[float],
        scopes: Sequence[str],
        top_k: int,
        *,
        document_codes: Sequence[str] | None = None,
        authoritative_only: bool = False,
    ) -> list[ChunkSearchRow]:
        distance = DocumentChunk.embedding.cosine_distance(embedding)
        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.chunk_key,
                DocumentChunk.content,
                Document.document_code,
                Document.title,
                Document.version,
                Document.type,
                Document.access_level,
                Document.source_authority,
                Document.related_document_codes,
                (1 - distance).label("similarity"),
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == "active", Document.access_level.in_(scopes))
            .order_by(distance)
            .limit(top_k)
        )
        if document_codes is not None:
            stmt = stmt.where(Document.document_code.in_(document_codes))
        if authoritative_only:
            stmt = stmt.where(Document.source_authority == "authoritative")
        return [_to_row(r) for r in self._s.execute(stmt).all()]


def _to_row(r: Any) -> ChunkSearchRow:
    return ChunkSearchRow(
        chunk_id=r.id,
        chunk_key=r.chunk_key,
        content=r.content,
        document_code=r.document_code,
        title=r.title,
        version=r.version,
        type=r.type,
        access_level=r.access_level,
        source_authority=r.source_authority,
        related_document_codes=list(r.related_document_codes or []),
        similarity=float(r.similarity),
    )


class IngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add(self, run: IngestionRun) -> IngestionRun:
        self._s.add(run)
        self._s.flush()
        return run


class AuditRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add(self, log: QueryLog, hits: Sequence[RetrievalHit]) -> QueryLog:
        log.hits = list(hits)
        self._s.add(log)
        self._s.flush()
        return log

    def list_recent(self, limit: int) -> Sequence[QueryLog]:
        return self._s.scalars(
            select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit)
        ).all()


class EvaluationRepository:
    def __init__(self, session: Session) -> None:
        self._s = session

    def add(self, run: EvaluationRun) -> EvaluationRun:
        self._s.add(run)
        self._s.flush()
        return run

    def list_recent(self, limit: int) -> Sequence[EvaluationRun]:
        return self._s.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        ).all()
