import os

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LLM_PROVIDER", "fake")

from dataclasses import dataclass, field

import pytest

from app.auth import Actor
from app.graph.build import build_graph
from app.graph.nodes import GraphDeps
from app.graph.state import RetrievedChunk
from app.rag.generator import Draft
from app.services.audit import InMemoryAuditSink
from app.services.chat import ChatService

EMPLOYEE = Actor("emp_test", ("employee",), ("public", "internal"))


def make_chunk(
    document_id: str = "HR-LEAVE-001",
    text: str = "ขั้นตอนการลาป่วย\n## การแจ้งลาป่วย\nแจ้งหัวหน้างานก่อน 09:00 น.",
    similarity: float = 0.82,
    authority: str = "authoritative",
    chunk_id: str = "00000000-0000-0000-0000-000000000001",
    access_level: str = "internal",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        chunk_key=f"{document_id}:1:0:abc",
        document_id=document_id,
        title="ขั้นตอนการลาป่วย (จำลอง)",
        version=1,
        doc_type="SOP" if authority == "authoritative" else "Chat",
        access_level=access_level,
        source_authority=authority,  # type: ignore[typeddict-item]
        text=text,
        similarity=similarity,
    )


@dataclass
class FakeRetriever:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    error: Exception | None = None
    calls: int = 0

    def retrieve(self, query: str, scopes: list[str]) -> list[RetrievedChunk]:
        self.calls += 1
        if self.error:
            raise self.error
        return [c for c in self.chunks if c["access_level"] in scopes]


@dataclass
class ScriptedGenerator:
    draft: Draft | None = None
    error: Exception | None = None
    model_name: str = "scripted"
    calls: int = 0

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Draft:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.draft is not None
        return self.draft


def draft(
    answer: str = "ตามนโยบายจำลอง แจ้งหัวหน้างานก่อน 09:00 น.",
    cited: list[str] | None = None,
    cannot_answer: bool = False,
    needs_clarification: bool = False,
) -> Draft:
    return Draft(
        answer=answer,
        cited_chunk_ids=cited if cited is not None else ["00000000-0000-0000-0000-000000000001"],
        cannot_answer=cannot_answer,
        needs_clarification=needs_clarification,
        model="scripted",
    )


@dataclass
class Harness:
    retriever: FakeRetriever
    generator: ScriptedGenerator
    audit: InMemoryAuditSink
    service: ChatService


@pytest.fixture
def harness() -> Harness:
    retriever = FakeRetriever(chunks=[make_chunk()])
    generator = ScriptedGenerator(draft=draft())
    audit = InMemoryAuditSink()
    deps = GraphDeps(
        retriever=retriever,
        generator=generator,
        audit_sink=audit,
        hash_salt="test-salt",
        max_query_chars=1000,
        max_answer_chars=2000,
        max_context_chunks=5,
        log_raw_queries=False,
    )
    return Harness(retriever, generator, audit, ChatService(build_graph(deps), 0.70))
