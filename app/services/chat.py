import time
from typing import Any

from app.auth import Actor
from app.domain import ReasonCode
from app.graph.state import RAGState, RetrievedChunk
from app.schemas.responses import (
    ChatResponse,
    DebugInfo,
    RetrievalHitInfo,
    RetrievalInfo,
    SourceRef,
)

_POLICY_REASONS = {ReasonCode.PROMPT_INJECTION, ReasonCode.OUT_OF_SCOPE, ReasonCode.INVALID_INPUT}


class ChatService:
    def __init__(self, graph: Any, default_threshold: float) -> None:
        self._graph = graph
        self.default_threshold = default_threshold

    def ask(
        self, query: str, actor: Actor, request_id: str, threshold: float | None = None
    ) -> ChatResponse:
        initial: RAGState = {
            "request_id": request_id,
            "actor_id": actor.actor_id,
            "allowed_scopes": list(actor.scopes),
            "threshold": self.default_threshold if threshold is None else threshold,
            "started_at": time.perf_counter(),
            "query": query,
            "status": "pending",
            "reason_code": None,
            "trace": [],
        }
        final: RAGState = self._graph.invoke(initial)
        return to_response(final, actor)


def to_response(state: RAGState, actor: Actor) -> ChatResponse:
    status = state["status"]
    sources: list[SourceRef] = []
    if status == "answered":
        by_id = {c["chunk_id"]: c for c in state.get("context_chunks", [])}
        for chunk_id in state.get("source_ids", []):
            chunk = by_id.get(chunk_id)
            # Second authorization check before showing a source to the user.
            if chunk and chunk["access_level"] in actor.scopes:
                sources.append(_source_ref(chunk))

    reason = state.get("reason_code")
    if status == "answered":
        decision = "document_grounded"
    elif reason in _POLICY_REASONS:
        decision = "system_policy"
    else:
        decision = "system_fallback"

    hits = sorted(state.get("retrieved_chunks", []), key=lambda c: c["similarity"], reverse=True)
    return ChatResponse(
        request_id=state["request_id"],
        status=status,
        answer=state.get("answer", ""),
        sources=sources,
        retrieval=RetrievalInfo(
            best_similarity=state.get("best_relevance"), threshold=state["threshold"]
        ),
        reason_code=reason,
        decision_type=decision,
        audit_status=state.get("audit_status", "degraded"),
        debug=DebugInfo(
            trace=state.get("trace", []),
            rewritten_query=state.get("rewritten_query"),
            hits=[
                RetrievalHitInfo(
                    document_id=c["document_id"],
                    chunk_id=c["chunk_id"],
                    similarity=c["similarity"],
                    source_authority=c["source_authority"],
                )
                for c in hits
            ],
            gate_passed=state.get("gate_passed"),
            latency_ms=state.get("latency_ms", 0.0),
            model=state.get("model"),
        ),
    )


def _source_ref(chunk: RetrievedChunk) -> SourceRef:
    body = chunk["text"].split("\n", 1)[-1].strip()
    return SourceRef(
        document_id=chunk["document_id"],
        chunk_id=chunk["chunk_id"],
        title=chunk["title"],
        version=chunk["version"],
        snippet=body[:300],
    )
