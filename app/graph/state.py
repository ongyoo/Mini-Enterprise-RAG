import operator
from typing import Annotated, Literal, TypedDict

from app.domain import Status

Authority = Literal["authoritative", "example"]


class RetrievedChunk(TypedDict):
    chunk_id: str
    chunk_key: str
    document_id: str
    title: str
    version: int
    doc_type: str
    access_level: str
    source_authority: Authority
    text: str
    similarity: float


class TokenUsage(TypedDict, total=False):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class RAGState(TypedDict, total=False):
    request_id: str
    actor_id: str
    allowed_scopes: list[str]
    threshold: float
    started_at: float

    query: str
    sanitized_query: str
    rewritten_query: str | None
    pii_types: list[str]
    matched_rule: str | None

    retrieved_chunks: list[RetrievedChunk]
    context_chunks: list[RetrievedChunk]
    best_relevance: float | None
    gate_passed: bool | None

    draft_answer: str
    draft_citations: list[str]
    cannot_answer: bool
    needs_clarification: bool
    model: str | None
    token_usage: TokenUsage

    answer: str
    source_ids: list[str]
    status: Status
    reason_code: str | None
    verification_detail: str | None

    audit_id: str | None
    audit_status: Literal["recorded", "degraded"]
    latency_ms: float
    trace: Annotated[list[str], operator.add]
