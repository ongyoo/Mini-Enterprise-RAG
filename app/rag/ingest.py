import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.models import Document, DocumentChunk, IngestionRun
from app.db.repositories import DocumentRepository, IngestionRunRepository
from app.db.session import session_scope
from app.rag.splitter import split_content

log = logging.getLogger(__name__)

_CHUNK_NAMESPACE = uuid.UUID("6f1c2a8e-3b7d-4c55-9a51-2f0e8b7d4c11")
DOCUMENT_ID_PATTERN = r"^[A-Z]+(-[A-Z]+)*-\d{3}$"


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    title: str = Field(min_length=1, max_length=300)
    type: Literal["SOP", "Chat"]
    category: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=10, max_length=20_000)
    version: int = Field(ge=1)
    status: Literal["active", "inactive"]
    access_level: Literal["public", "internal", "confidential"]
    source_authority: Literal["authoritative", "example"]
    synthetic: Literal[True]
    updated_at: date
    related_document_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _authority_matches_type(self) -> "SourceDocument":
        expected = "authoritative" if self.type == "SOP" else "example"
        if self.source_authority != expected:
            raise ValueError(f"{self.type} documents must have source_authority={expected}")
        return self

    @property
    def content_hash(self) -> str:
        return sha256(self.model_dump_json())


@dataclass
class IngestionReport:
    run_id: str = ""
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_embedded: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def status(self) -> str:
        if not self.failed:
            return "succeeded"
        return "partial" if self.inserted + self.updated + self.skipped else "failed"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_TYPE_ALIASES = {"policy": "SOP", "sop": "SOP", "chat": "Chat"}


def _parse_version(value: Any) -> Any:
    """Accept 2 or "2.0"; anything else is left for validation to reject."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else value


def normalize_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Map the handbook export formats onto SourceDocument. Native records pass through.

    Exports carry no `synthetic` flag; they are treated as synthetic demo data.
    """
    if "type" in raw and "source_authority" in raw:
        return raw
    kind = str(raw.get("source_type") or raw.get("record_type") or raw.get("type") or "")
    doc_type = _TYPE_ALIASES.get(kind.lower(), kind)
    return {
        "document_id": raw.get("document_id") or raw.get("id"),
        "title": raw.get("title"),
        "type": doc_type,
        "category": raw.get("category") or raw.get("department"),
        "content": raw.get("content"),
        "version": _parse_version(raw.get("version", 1)),
        "status": raw.get("status", "active"),
        "access_level": str(raw.get("access_level") or raw.get("classification") or "").lower(),
        "source_authority": "authoritative" if doc_type == "SOP" else "example",
        "synthetic": raw.get("synthetic", True),
        "updated_at": raw.get("updated_at") or raw.get("effective_date"),
        "related_document_ids": raw.get("related_document_ids") or [],
    }


def load_documents(path: Path) -> tuple[list[SourceDocument], list[str]]:
    """Parse and validate JSONL. Conflicting duplicate IDs are rejected entirely."""
    docs: dict[str, SourceDocument] = {}
    conflicts: set[str] = set()
    errors: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            doc = SourceDocument.model_validate(normalize_record(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {line_no}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        existing = docs.get(doc.document_id)
        if existing is not None and existing != doc:
            conflicts.add(doc.document_id)
        docs[doc.document_id] = doc

    for code in sorted(conflicts):
        errors.append(f"{code}: duplicate document_id with conflicting content")
        del docs[code]
    return list(docs.values()), errors


def build_chunks(doc: SourceDocument, settings: Settings) -> list[tuple[str, str, int]]:
    texts = split_content(
        title=doc.title,
        content=doc.content,
        doc_type=doc.type,
        chunk_size=settings.rag_chunk_size_chars,
        chunk_overlap=settings.rag_chunk_overlap_chars,
    )
    return [
        (f"{doc.document_id}:{doc.version}:{i}:{sha256(t)[:12]}", t, i) for i, t in enumerate(texts)
    ]


def ingest_file(
    path: Path,
    session_factory: sessionmaker[Session],
    embeddings: Embeddings,
    settings: Settings,
) -> IngestionReport:
    started = time.perf_counter()
    report = IngestionReport()
    docs, report.errors = load_documents(path)
    report.failed = len(report.errors)

    for doc in docs:
        try:
            outcome = _ingest_one(doc, session_factory, embeddings, settings, report)
        except Exception as exc:
            log.exception("ingest failed", extra={"document_id": doc.document_id})
            report.failed += 1
            report.errors.append(f"{doc.document_id}: {type(exc).__name__}")
            continue
        if outcome == "inserted":
            report.inserted += 1
        elif outcome == "updated":
            report.updated += 1
        else:
            report.skipped += 1

    report.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    with session_scope(session_factory) as session:
        run = IngestionRunRepository(session).add(
            IngestionRun(
                finished_at=datetime.now(UTC),
                status=report.status,
                inserted_count=report.inserted,
                updated_count=report.updated,
                skipped_count=report.skipped,
                failed_count=report.failed,
                error_summary="\n".join(report.errors)[:4000] or None,
            )
        )
        report.run_id = str(run.id)
    return report


def _ingest_one(
    doc: SourceDocument,
    session_factory: sessionmaker[Session],
    embeddings: Embeddings,
    settings: Settings,
    report: IngestionReport,
) -> str:
    with session_scope(session_factory) as session:
        existing = DocumentRepository(session).get_by_code(doc.document_id)
        if existing is not None:
            if existing.version > doc.version:
                raise ValueError("incoming version is older than stored version")
            if existing.version == doc.version:
                if existing.content_hash == doc.content_hash:
                    return "skipped"
                raise ValueError("content changed without a version bump")

    chunks = build_chunks(doc, settings)
    # Embed outside the DB transaction so the transaction stays short.
    vectors = embeddings.embed_documents([text for _, text, _ in chunks])
    report.chunks_embedded += len(vectors)
    if any(len(v) != settings.embedding_dimensions for v in vectors):
        raise ValueError("embedding dimension does not match EMBEDDING_DIMENSIONS")

    with session_scope(session_factory) as session:
        repo = DocumentRepository(session)
        row = repo.get_by_code(doc.document_id)
        outcome = "updated" if row else "inserted"
        if row is None:
            row = Document(document_code=doc.document_id)
            session.add(row)
        row.title = doc.title
        row.type = doc.type
        row.category = doc.category
        row.version = doc.version
        row.status = doc.status
        row.access_level = doc.access_level
        row.source_authority = doc.source_authority
        row.related_document_codes = doc.related_document_ids
        row.synthetic = doc.synthetic
        row.content_hash = doc.content_hash
        row.source_updated_at = doc.updated_at
        session.flush()

        repo.replace_chunks(
            row,
            [
                DocumentChunk(
                    id=uuid.uuid5(_CHUNK_NAMESPACE, key),
                    document_id=row.id,
                    chunk_index=index,
                    chunk_key=key,
                    content=text,
                    embedding=vector,
                    content_hash=sha256(text),
                )
                for (key, text, index), vector in zip(chunks, vectors, strict=True)
            ],
        )
    return outcome
