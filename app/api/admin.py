from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import AdminDep, ContainerDep
from app.container import EVAL_ACTOR
from app.db.models import EvaluationRun
from app.db.repositories import AuditRepository, EvaluationRepository
from app.db.session import session_scope
from app.rag.ingest import ingest_file
from app.schemas.requests import EvaluationRunRequest
from app.schemas.responses import AuditLogEntry, EvaluationSummary, IngestionResponse
from app.services.evaluation import load_cases, run_evaluation

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/ingest", response_model=IngestionResponse)
def ingest(_: AdminDep, container: ContainerDep) -> IngestionResponse:
    s = container.settings
    report = ingest_file(s.documents_path, container.session_factory, container.embeddings, s)
    return IngestionResponse(
        run_id=report.run_id,
        status=report.status,
        inserted=report.inserted,
        updated=report.updated,
        skipped=report.skipped,
        failed=report.failed,
        chunks_embedded=report.chunks_embedded,
        elapsed_ms=report.elapsed_ms,
        errors=report.errors,
    )


@router.post("/evaluations/run", response_model=EvaluationSummary)
def run_eval(body: EvaluationRunRequest, _: AdminDep, container: ContainerDep) -> EvaluationSummary:
    s = container.settings
    cases, dataset_version = load_cases(s.evaluation_path)
    report = run_evaluation(container.chat_service, EVAL_ACTOR, cases, body.thresholds)
    config = {
        "provider": s.llm_provider,
        "chat_model": s.chat_model_name,
        "embedding_model": s.embedding_model_name,
        "embedding_dimensions": s.embedding_dimensions,
        "top_k": s.rag_top_k,
        "thresholds": body.thresholds,
    }
    with session_scope(container.session_factory) as session:
        run = EvaluationRepository(session).add(
            EvaluationRun(
                dataset_version=dataset_version, config_snapshot=config, metrics_json=report
            )
        )
        return _summary(run)


@router.get("/evaluations", response_model=list[EvaluationSummary])
def list_evals(
    _: AdminDep, container: ContainerDep, limit: Annotated[int, Query(ge=1, le=50)] = 10
) -> list[EvaluationSummary]:
    with container.session_factory() as session:
        return [_summary(r) for r in EvaluationRepository(session).list_recent(limit)]


@router.get("/audit-logs", response_model=list[AuditLogEntry])
def audit_logs(
    _: AdminDep, container: ContainerDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[AuditLogEntry]:
    with container.session_factory() as session:
        return [
            AuditLogEntry(
                request_id=log.request_id,
                actor_hash=log.actor_hash[:16],
                redacted_query=log.redacted_query,
                status=log.status,
                reason_code=log.reason_code,
                best_relevance=log.best_relevance,
                model=log.model,
                latency_ms=log.latency_ms,
                details=log.details,
                created_at=log.created_at,
            )
            for log in AuditRepository(session).list_recent(limit)
        ]


def _summary(run: EvaluationRun) -> EvaluationSummary:
    return EvaluationSummary(
        id=str(run.id),
        dataset_version=run.dataset_version,
        created_at=run.created_at,
        config_snapshot=run.config_snapshot,
        metrics=run.metrics_json,
    )
