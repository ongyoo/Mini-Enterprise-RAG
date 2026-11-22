import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import QueryLog, RetrievalHit
from app.db.repositories import AuditRepository
from app.db.session import session_scope


@dataclass(frozen=True)
class AuditHit:
    chunk_id: str
    chunk_key: str
    rank: int
    similarity: float


@dataclass(frozen=True)
class AuditRecord:
    request_id: str
    actor_hash: str
    query_hash: str
    redacted_query: str | None
    status: str
    reason_code: str | None
    best_relevance: float | None
    model: str | None
    latency_ms: float
    hits: list[AuditHit] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class AuditSink(Protocol):
    def record(self, record: AuditRecord) -> str: ...


class DbAuditSink:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def record(self, record: AuditRecord) -> str:
        with session_scope(self._session_factory) as session:
            log = AuditRepository(session).add(
                QueryLog(
                    request_id=record.request_id,
                    actor_hash=record.actor_hash,
                    query_hash=record.query_hash,
                    redacted_query=record.redacted_query,
                    status=record.status,
                    reason_code=record.reason_code,
                    best_relevance=record.best_relevance,
                    model=record.model,
                    latency_ms=record.latency_ms,
                    details=record.details,
                ),
                [
                    RetrievalHit(
                        chunk_id=uuid.UUID(h.chunk_id),
                        chunk_key=h.chunk_key,
                        rank=h.rank,
                        cosine_similarity=h.similarity,
                    )
                    for h in record.hits
                ],
            )
            return str(log.id)


class InMemoryAuditSink:
    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def record(self, record: AuditRecord) -> str:
        self.records.append(record)
        return f"mem-{len(self.records)}"
