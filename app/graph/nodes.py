import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.domain import (
    ERROR_REASONS,
    GenerationParseError,
    ProviderUnavailableError,
    ReasonCode,
    RetrievalUnavailableError,
    fallback_message,
)
from app.graph.state import RAGState
from app.guardrails.input_policy import evaluate_input
from app.guardrails.output_policy import verify_answer
from app.guardrails.redaction import hash_identifier
from app.rag.generator import Generator
from app.rag.retriever import Retriever
from app.rag.rewriter import QueryRewriter
from app.services.audit import AuditHit, AuditRecord, AuditSink

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphDeps:
    retriever: Retriever
    generator: Generator
    audit_sink: AuditSink
    hash_salt: str
    max_query_chars: int
    max_answer_chars: int
    max_context_chunks: int
    log_raw_queries: bool
    query_rewriter: QueryRewriter | None = None
    list_topics: Callable[[list[str]], dict[str, list[str]]] | None = None


def validate_input(state: RAGState, deps: GraphDeps) -> RAGState:
    decision = evaluate_input(state["query"], deps.max_query_chars)
    update: RAGState = {"trace": ["validate_input"], "matched_rule": decision.matched_rule}
    if not decision.allowed:
        reason = decision.reason_code or ReasonCode.INVALID_INPUT
        update |= {
            "status": "refused",
            "reason_code": reason,
            "answer": fallback_message(reason, _topics(state, deps, reason)),
            "source_ids": [],
        }
        return update
    update |= {"sanitized_query": decision.sanitized_query, "pii_types": list(decision.pii_types)}
    return update


def retrieve(state: RAGState, deps: GraphDeps) -> RAGState:
    update: RAGState = {"trace": ["retrieve"]}
    try:
        query = state.get("rewritten_query") or state["sanitized_query"]
        chunks = deps.retriever.retrieve(query, state["allowed_scopes"])
    except ProviderUnavailableError:
        return update | {"reason_code": ReasonCode.PROVIDER_UNAVAILABLE}
    except RetrievalUnavailableError:
        return update | {"reason_code": ReasonCode.RETRIEVAL_UNAVAILABLE}

    if not chunks:
        return update | {"retrieved_chunks": [], "reason_code": ReasonCode.NO_RESULTS}
    return update | {
        "retrieved_chunks": chunks,
        "best_relevance": max(c["similarity"] for c in chunks),
    }


def evidence_gate(state: RAGState, deps: GraphDeps) -> RAGState:
    """Pass only when the best hit clears the threshold and at least one SOP chunk is present.

    The threshold is a retrieval heuristic, not a calibrated probability.
    """
    update: RAGState = {"trace": ["evidence_gate"], "gate_passed": False}
    best = state.get("best_relevance") or 0.0
    if best < state["threshold"]:
        return update | {"reason_code": ReasonCode.LOW_EVIDENCE}

    sops = [c for c in state["retrieved_chunks"] if c["source_authority"] == "authoritative"]
    if not sops:
        return update | {"reason_code": ReasonCode.NO_AUTHORITATIVE_SOURCE}

    sops.sort(key=lambda c: c["similarity"], reverse=True)
    return {
        "trace": ["evidence_gate"],
        "gate_passed": True,
        "context_chunks": sops[: deps.max_context_chunks],
    }


def rewrite_query(state: RAGState, deps: GraphDeps) -> RAGState:
    """Weak evidence is often a typo or slang. Normalise the query once and search again."""
    update: RAGState = {"trace": ["rewrite_query"], "rewritten_query": None}
    original = state["sanitized_query"]
    if deps.query_rewriter is None:
        return update
    titles = [t for group in _topics(state, deps, ReasonCode.LOW_EVIDENCE).values() for t in group]
    result = deps.query_rewriter.rewrite(original, titles)
    if result is None:
        return update
    if not result.understood:
        return update | {"reason_code": ReasonCode.UNCLEAR_QUERY}
    if result.query == original:
        return update
    return update | {"rewritten_query": result.query, "reason_code": None}


def generate(state: RAGState, deps: GraphDeps) -> RAGState:
    update: RAGState = {"trace": ["generate"], "model": deps.generator.model_name}
    try:
        draft = deps.generator.generate(state["sanitized_query"], state["context_chunks"])
    except ProviderUnavailableError:
        return update | {"reason_code": ReasonCode.PROVIDER_UNAVAILABLE}
    except GenerationParseError:
        return update | {"reason_code": ReasonCode.GENERATION_PARSE_ERROR}

    return update | {
        "draft_answer": draft.answer,
        "draft_citations": draft.cited_chunk_ids,
        "cannot_answer": draft.cannot_answer,
        "needs_clarification": draft.needs_clarification,
        "token_usage": draft.token_usage,
    }


def verify_output(state: RAGState, deps: GraphDeps) -> RAGState:
    update: RAGState = {"trace": ["verify_output"]}

    if state.get("needs_clarification") and state.get("draft_answer", "").strip():
        return update | {
            "status": "fallback",
            "reason_code": ReasonCode.NEEDS_CLARIFICATION,
            "answer": state["draft_answer"].strip()[: deps.max_answer_chars],
            "source_ids": [],
        }
    if state.get("cannot_answer"):
        return update | {"reason_code": ReasonCode.MODEL_CANNOT_ANSWER}

    allowed = {c["chunk_id"]: c["text"] for c in state["context_chunks"]}
    result = verify_answer(
        answer=state.get("draft_answer", ""),
        cited_chunk_ids=state.get("draft_citations", []),
        allowed_chunks=allowed,
        query=state["sanitized_query"],
        max_answer_chars=deps.max_answer_chars,
    )
    if not result.ok:
        return update | {"reason_code": result.reason_code, "verification_detail": result.detail}
    return update | {
        "status": "answered",
        "reason_code": None,
        "answer": state["draft_answer"].strip(),
        "source_ids": list(result.cited_chunk_ids),
    }


def _topics(state: RAGState, deps: GraphDeps, reason: ReasonCode) -> dict[str, list[str]]:
    if deps.list_topics is None or reason in ERROR_REASONS or reason is ReasonCode.PROMPT_INJECTION:
        return {}
    try:
        return deps.list_topics(state["allowed_scopes"])
    except Exception:
        log.warning("could not load topic list", exc_info=True)
        return {}


def fallback(state: RAGState, deps: GraphDeps) -> RAGState:
    reason = ReasonCode(state.get("reason_code") or ReasonCode.INTERNAL_ERROR)
    return {
        "trace": ["fallback"],
        "status": "error" if reason in ERROR_REASONS else "fallback",
        "reason_code": reason,
        "answer": fallback_message(reason, _topics(state, deps, reason)),
        "source_ids": [],
    }


def audit_and_finalize(state: RAGState, deps: GraphDeps) -> RAGState:
    latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 1)
    hits = sorted(state.get("retrieved_chunks", []), key=lambda c: c["similarity"], reverse=True)
    record = AuditRecord(
        request_id=state["request_id"],
        actor_hash=hash_identifier(state["actor_id"], deps.hash_salt),
        query_hash=hash_identifier(state["query"], deps.hash_salt),
        redacted_query=state.get("sanitized_query") if deps.log_raw_queries else None,
        status=state["status"],
        reason_code=state.get("reason_code"),
        best_relevance=state.get("best_relevance"),
        model=state.get("model"),
        latency_ms=latency_ms,
        hits=[
            AuditHit(c["chunk_id"], c["chunk_key"], rank, c["similarity"])
            for rank, c in enumerate(hits, start=1)
        ],
        details={
            "trace": [*state.get("trace", []), "audit_and_finalize"],
            "threshold": state.get("threshold"),
            "matched_rule": state.get("matched_rule"),
            "rewritten_query": state.get("rewritten_query") if deps.log_raw_queries else None,
            "pii_types": state.get("pii_types", []),
            "token_usage": state.get("token_usage", {}),
            "verification_detail": state.get("verification_detail"),
            "cited_chunk_ids": state.get("source_ids", []),
        },
    )
    audit_status: Literal["recorded", "degraded"] = "recorded"
    try:
        audit_id: str | None = deps.audit_sink.record(record)
    except Exception:
        # Audit failure must not hide the answer, but the response is flagged as degraded.
        log.exception("audit write failed", extra={"request_id": state["request_id"]})
        audit_id, audit_status = None, "degraded"

    return {
        "trace": ["audit_and_finalize"],
        "latency_ms": latency_ms,
        "audit_id": audit_id,
        "audit_status": audit_status,
    }
