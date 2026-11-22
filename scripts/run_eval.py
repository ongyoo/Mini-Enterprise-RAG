import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.container import EVAL_ACTOR, build_container
from app.db.models import EvaluationRun
from app.db.repositories import EvaluationRepository
from app.db.session import session_scope
from app.logging_config import configure_logging
from app.services.evaluation import load_cases, run_evaluation

METRICS_TO_PRINT = (
    "behavior_accuracy",
    "recall_at_5",
    "mrr",
    "citation_precision",
    "fact_match_rate",
    "injection_refusal_recall",
    "false_refusal_rate",
    "fallback_accuracy",
    "false_answer_rate",
    "unnecessary_fallback_rate",
    "error_rate",
    "latency_p50_ms",
    "latency_p95_ms",
)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the frozen 40-case evaluation")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.28, 0.35])
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    configure_logging("WARNING")
    container = build_container(settings)
    cases, dataset_version = load_cases(settings.evaluation_path)
    report = run_evaluation(container.chat_service, EVAL_ACTOR, cases, args.thresholds)

    config = {
        "provider": settings.llm_provider,
        "chat_model": settings.chat_model_name,
        "embedding_model": settings.embedding_model_name,
        "top_k": settings.rag_top_k,
        "thresholds": args.thresholds,
    }
    with session_scope(container.session_factory) as session:
        EvaluationRepository(session).add(
            EvaluationRun(
                dataset_version=dataset_version, config_snapshot=config, metrics_json=report
            )
        )

    args.out_dir.mkdir(exist_ok=True)
    out = args.out_dir / f"eval_{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.write_text(json.dumps({"config": config, **report}, ensure_ascii=False, indent=2))

    for run in report["runs"]:
        print(f"\n== threshold {run['threshold']} ==")
        for key in METRICS_TO_PRINT:
            print(f"  {key:28} {run['metrics'][key]}")
        print(f"  failures: {len(run['failures'])}")
        for f in run["failures"][:10]:
            print(
                f"    {f['case_id']:9} expected={f['expected_behavior']:8} "
                f"got={f['observed_behavior']:8} reason={f['reason_code']}"
            )
    print(
        f"\nrecommended threshold: {report['recommended_threshold']} ({report['selection_rule']})"
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
