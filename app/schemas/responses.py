from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel

from app.domain import Status


class SourceRef(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    version: int
    snippet: str


class RetrievalInfo(BaseModel):
    best_similarity: float | None
    threshold: float


class RetrievalHitInfo(BaseModel):
    document_id: str
    chunk_id: str
    similarity: float
    source_authority: str


class DebugInfo(BaseModel):
    trace: list[str]
    hits: list[RetrievalHitInfo]
    gate_passed: bool | None
    rewritten_query: str | None = None
    latency_ms: float
    model: str | None


class ChatResponse(BaseModel):
    request_id: str
    status: Status
    answer: str
    sources: list[SourceRef]
    retrieval: RetrievalInfo
    reason_code: str | None
    decision_type: Literal["document_grounded", "system_policy", "system_fallback"]
    audit_status: Literal["recorded", "degraded"]
    debug: DebugInfo


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    type: str
    category: str
    version: int
    access_level: str
    source_authority: str
    synthetic: bool
    updated_at: date


class IngestionResponse(BaseModel):
    run_id: str
    status: str
    inserted: int
    updated: int
    skipped: int
    failed: int
    chunks_embedded: int
    elapsed_ms: float
    errors: list[str]


class AuditLogEntry(BaseModel):
    request_id: str
    actor_hash: str
    redacted_query: str | None
    status: str
    reason_code: str | None
    best_relevance: float | None
    model: str | None
    latency_ms: float
    details: dict[str, Any]
    created_at: datetime


class EvaluationSummary(BaseModel):
    id: str
    dataset_version: str
    created_at: datetime
    config_snapshot: dict[str, Any]
    metrics: dict[str, Any]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorBody
