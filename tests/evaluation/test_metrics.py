from pathlib import Path

from app.services.evaluation import CaseResult, EvalCase, compute_metrics, load_cases

DATA = Path(__file__).parents[2] / "data"


def test_dataset_shape_matches_spec() -> None:
    cases, version = load_cases(DATA / "evaluation.jsonl")
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.category] = counts.get(c.category, 0) + 1
    assert len(cases) == 40
    assert counts == {
        "sop": 10,
        "slang_typo": 10,
        "ambiguous": 5,
        "out_of_scope": 5,
        "injection": 5,
        "missing_knowledge": 5,
    }
    assert len({c.case_id for c in cases}) == 40
    assert len(version) == 12


def _result(case: EvalCase, observed: str, retrieved: list[str], cited: list[str]) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        expected_behavior=case.expected_behavior,
        observed_behavior=observed,
        passed=observed == case.expected_behavior,
        reason_code=None,
        retrieved_document_ids=retrieved,
        cited_document_ids=cited,
        facts_found=1 if observed == "answered" else 0,
        facts_total=1 if observed == "answered" else 0,
        best_similarity=0.8,
        latency_ms=100.0,
        answer="",
    )


def test_metrics_math() -> None:
    a = EvalCase(
        case_id="a",
        query="q",
        expected_behavior="answered",
        expected_document_ids=["D1"],
        category="sop",
    )
    b = EvalCase(
        case_id="b",
        query="q",
        expected_behavior="answered",
        expected_document_ids=["D2"],
        category="sop",
    )
    i = EvalCase(case_id="i", query="q", expected_behavior="refused", category="injection")
    m = EvalCase(case_id="m", query="q", expected_behavior="fallback", category="missing_knowledge")
    results = [
        _result(a, "answered", ["D1", "D2"], ["D1"]),
        _result(b, "answered", ["D1", "D2"], ["D1"]),
        _result(i, "refused", [], []),
        _result(m, "answered", ["D3"], ["D3"]),
    ]
    metrics = compute_metrics([a, b, i, m], results)
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.75
    assert metrics["citation_precision"] == round(1 / 3, 4)
    assert metrics["injection_refusal_recall"] == 1.0
    assert metrics["false_refusal_rate"] == 0.0
    assert metrics["fallback_accuracy"] == 0.0
    assert metrics["false_answer_rate"] == 0.5
    assert metrics["behavior_accuracy"] == 0.75
