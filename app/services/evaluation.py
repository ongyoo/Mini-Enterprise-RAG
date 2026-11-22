import hashlib
import statistics
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.auth import Actor
from app.schemas.responses import ChatResponse
from app.services.chat import ChatService

Behavior = Literal["answered", "fallback", "refused", "clarify", "error"]

FACT_MATCH_METHOD = "lexical: whitespace/comma-insensitive substring match of expected_facts"


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    query: str
    expected_behavior: Literal["answered", "fallback", "refused", "clarify"]
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    category: str


@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    expected_behavior: str
    observed_behavior: str
    passed: bool
    reason_code: str | None
    retrieved_document_ids: list[str]
    cited_document_ids: list[str]
    facts_found: int
    facts_total: int
    best_similarity: float | None
    latency_ms: float
    answer: str


def load_cases(path: Path) -> tuple[list[EvalCase], str]:
    raw = path.read_bytes()
    cases = [EvalCase.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    return cases, hashlib.sha256(raw).hexdigest()[:12]


def observed_behavior(response: ChatResponse) -> Behavior:
    if response.status == "fallback" and response.reason_code in (
        "NEEDS_CLARIFICATION",
        "UNCLEAR_QUERY",
    ):
        return "clarify"
    return response.status  # type: ignore[return-value]


def _normalize(text: str) -> str:
    return "".join(text.lower().split()).replace(",", "")


def _ranked_documents(response: ChatResponse) -> list[str]:
    return list(dict.fromkeys(h.document_id for h in response.debug.hits))


def run_case(service: ChatService, actor: Actor, case: EvalCase, threshold: float) -> CaseResult:
    response = service.ask(case.query, actor, f"eval_{uuid.uuid4().hex[:12]}", threshold)
    behavior = observed_behavior(response)
    answer = _normalize(response.answer) if behavior == "answered" else ""
    found = sum(1 for fact in case.expected_facts if _normalize(fact) in answer)
    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        expected_behavior=case.expected_behavior,
        observed_behavior=behavior,
        passed=behavior == case.expected_behavior,
        reason_code=response.reason_code,
        retrieved_document_ids=_ranked_documents(response),
        cited_document_ids=[s.document_id for s in response.sources],
        facts_found=found,
        facts_total=len(case.expected_facts),
        best_similarity=response.retrieval.best_similarity,
        latency_ms=response.debug.latency_ms,
        answer=response.answer,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_metrics(cases: list[EvalCase], results: list[CaseResult]) -> dict[str, Any]:
    by_id = {c.case_id: c for c in cases}

    answerable = [r for r in results if by_id[r.case_id].expected_behavior == "answered"]
    with_docs = [r for r in answerable if by_id[r.case_id].expected_document_ids]
    hits_at_5, reciprocal_ranks = 0, []
    for r in with_docs:
        expected = set(by_id[r.case_id].expected_document_ids)
        top5 = r.retrieved_document_ids[:5]
        hits_at_5 += any(d in expected for d in top5)
        rank = next((i for i, d in enumerate(r.retrieved_document_ids, 1) if d in expected), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    answered = [r for r in results if r.observed_behavior == "answered"]
    emitted = [(r, d) for r in answered for d in r.cited_document_ids]
    relevant = sum(1 for r, d in emitted if d in by_id[r.case_id].expected_document_ids)

    injections = [r for r in results if r.category == "injection"]
    benign = [r for r in results if by_id[r.case_id].expected_behavior != "refused"]
    expected_fallback = [r for r in results if r.expected_behavior == "fallback"]
    expected_non_answer = [r for r in results if r.expected_behavior != "answered"]
    latencies = sorted(r.latency_ms for r in results)

    per_category: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = per_category.setdefault(r.category, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += r.passed

    return {
        "cases": len(results),
        "behavior_accuracy": _rate(sum(r.passed for r in results), len(results)),
        "recall_at_5": _rate(hits_at_5, len(with_docs)),
        "mrr": round(statistics.fmean(reciprocal_ranks), 4) if reciprocal_ranks else None,
        "citation_precision": _rate(relevant, len(emitted)),
        "citation_coverage": _rate(sum(1 for r in answered if r.cited_document_ids), len(answered)),
        "fact_match_rate": _rate(
            sum(r.facts_found for r in answered), sum(r.facts_total for r in answered)
        ),
        "fact_match_method": FACT_MATCH_METHOD,
        "injection_refusal_recall": _rate(
            sum(r.observed_behavior == "refused" for r in injections), len(injections)
        ),
        "false_refusal_rate": _rate(
            sum(r.observed_behavior == "refused" for r in benign), len(benign)
        ),
        "fallback_accuracy": _rate(
            sum(r.observed_behavior == "fallback" for r in expected_fallback),
            len(expected_fallback),
        ),
        "false_answer_rate": _rate(
            sum(r.observed_behavior == "answered" for r in expected_non_answer),
            len(expected_non_answer),
        ),
        "unnecessary_fallback_rate": _rate(
            sum(r.observed_behavior in ("fallback", "clarify") for r in answerable),
            len(answerable),
        ),
        "error_rate": _rate(sum(r.observed_behavior == "error" for r in results), len(results)),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "per_category": per_category,
    }


def _percentile(sorted_values: list[float], pct: int) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[index]


def _selection_cost(metrics: dict[str, Any]) -> float:
    # A wrong answer is treated as twice as costly as an unnecessary fallback.
    return 2 * (metrics["false_answer_rate"] or 0) + (metrics["unnecessary_fallback_rate"] or 0)


def run_evaluation(
    service: ChatService, actor: Actor, cases: list[EvalCase], thresholds: list[float]
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for threshold in thresholds:
        results = [run_case(service, actor, case, threshold) for case in cases]
        metrics = compute_metrics(cases, results)
        runs.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "failures": [asdict(r) for r in results if not r.passed],
                "results": [asdict(r) for r in results],
            }
        )
    best = min(runs, key=lambda run: _selection_cost(run["metrics"]))
    return {
        "runs": runs,
        "recommended_threshold": best["threshold"],
        "selection_rule": "min(2 * false_answer_rate + unnecessary_fallback_rate)",
    }
