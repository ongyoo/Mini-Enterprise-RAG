from typing import Literal

from app.graph.state import RAGState


def after_validate(state: RAGState) -> Literal["retrieve", "audit_and_finalize"]:
    return "audit_and_finalize" if state.get("status") == "refused" else "retrieve"


def after_retrieve(state: RAGState) -> Literal["evidence_gate", "fallback"]:
    return "fallback" if state.get("reason_code") else "evidence_gate"


def after_gate(state: RAGState) -> Literal["generate", "rewrite_query", "fallback"]:
    if state.get("gate_passed"):
        return "generate"
    if state.get("reason_code") == "LOW_EVIDENCE" and "rewritten_query" not in state:
        return "rewrite_query"
    return "fallback"


def after_rewrite(state: RAGState) -> Literal["retrieve", "fallback"]:
    return "retrieve" if state.get("rewritten_query") else "fallback"


def after_generate(state: RAGState) -> Literal["verify_output", "fallback"]:
    return "fallback" if state.get("reason_code") else "verify_output"


def after_verify(state: RAGState) -> Literal["audit_and_finalize", "rewrite_query", "fallback"]:
    if state.get("reason_code") == "MODEL_CANNOT_ANSWER" and "rewritten_query" not in state:
        return "rewrite_query"
    return "audit_and_finalize" if state.get("status") in ("answered", "fallback") else "fallback"
