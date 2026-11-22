from dataclasses import dataclass, field

from app.graph.build import build_graph
from app.graph.nodes import GraphDeps
from app.graph.state import RetrievedChunk
from app.rag.rewriter import RewriteResult
from app.services.audit import InMemoryAuditSink
from app.services.chat import ChatService
from tests.conftest import EMPLOYEE, ScriptedGenerator, draft, make_chunk

TYPO = "ลาป๊วยทำไง"
FIXED = "ลาป่วยทำอย่างไร"


@dataclass
class QueryAwareRetriever:
    """Finds the sick-leave SOP only when the query is spelled correctly."""

    queries: list[str] = field(default_factory=list)

    def retrieve(self, query: str, scopes: list[str]) -> list[RetrievedChunk]:
        self.queries.append(query)
        return [make_chunk(similarity=0.82 if query == FIXED else 0.10)]


@dataclass
class FixedRewriter:
    query: str
    understood: bool = True
    calls: int = 0

    def rewrite(self, query: str, topics: list[str]) -> RewriteResult | None:
        self.calls += 1
        return RewriteResult(understood=self.understood, query=self.query)


def make_service(retriever: QueryAwareRetriever, rewriter: FixedRewriter | None) -> ChatService:
    deps = GraphDeps(
        retriever=retriever,
        generator=ScriptedGenerator(draft=draft()),
        audit_sink=InMemoryAuditSink(),
        hash_salt="test-salt",
        max_query_chars=1000,
        max_answer_chars=2000,
        max_context_chunks=5,
        log_raw_queries=False,
        query_rewriter=rewriter,
    )
    return ChatService(build_graph(deps), 0.70)


def test_typo_is_rewritten_and_searched_again() -> None:
    retriever = QueryAwareRetriever()
    r = make_service(retriever, FixedRewriter(FIXED)).ask(TYPO, EMPLOYEE, "req_test_0001")

    assert r.status == "answered"
    assert retriever.queries == [TYPO, FIXED]
    assert r.debug.rewritten_query == FIXED
    assert r.debug.trace.count("retrieve") == 2


def test_rewrite_happens_at_most_once() -> None:
    retriever = QueryAwareRetriever()
    rewriter = FixedRewriter("ยังสะกดผิดอยู่")
    r = make_service(retriever, rewriter).ask(TYPO, EMPLOYEE, "req_test_0001")

    assert (r.status, r.reason_code) == ("fallback", "LOW_EVIDENCE")
    assert rewriter.calls == 1
    assert len(retriever.queries) == 2


def test_no_rewriter_falls_back_directly() -> None:
    retriever = QueryAwareRetriever()
    r = make_service(retriever, None).ask(TYPO, EMPLOYEE, "req_test_0001")

    assert (r.status, r.reason_code) == ("fallback", "LOW_EVIDENCE")
    assert retriever.queries == [TYPO]


def test_good_query_skips_rewrite() -> None:
    retriever = QueryAwareRetriever()
    rewriter = FixedRewriter(FIXED)
    r = make_service(retriever, rewriter).ask(FIXED, EMPLOYEE, "req_test_0001")

    assert r.status == "answered"
    assert rewriter.calls == 0


def test_unreadable_query_asks_user_to_rephrase() -> None:
    retriever = QueryAwareRetriever()
    rewriter = FixedRewriter("", understood=False)
    r = make_service(retriever, rewriter).ask("กดฟหกด", EMPLOYEE, "req_test_0001")

    assert (r.status, r.reason_code) == ("fallback", "UNCLEAR_QUERY")
    assert "พิมพ์ใหม่" in r.answer
    assert r.sources == []
    assert len(retriever.queries) == 1


class CannotAnswerFirstGenerator(ScriptedGenerator):
    """Says "no answer" until it sees the chunk found by the rewritten query."""

    def generate(self, question, chunks):  # type: ignore[no-untyped-def]
        self.calls += 1
        if chunks[0]["chunk_id"] == HR_004["chunk_id"]:
            return draft(cited=[HR_004["chunk_id"]], answer="ตามนโยบายจำลอง ดูได้ใน Employee Portal")
        return draft(cannot_answer=True)


HR_004 = make_chunk(document_id="HR-004", chunk_id="00000000-0000-0000-0000-000000000004")


def test_cannot_answer_triggers_one_rewrite() -> None:
    @dataclass
    class Retriever:
        queries: list[str] = field(default_factory=list)

        def retrieve(self, query: str, scopes: list[str]) -> list[RetrievedChunk]:
            self.queries.append(query)
            return [HR_004] if query == "ตรวจสอบวันลาพักร้อนคงเหลือ" else [make_chunk()]

    retriever = Retriever()
    deps = GraphDeps(
        retriever=retriever,
        generator=CannotAnswerFirstGenerator(),
        audit_sink=InMemoryAuditSink(),
        hash_salt="test-salt",
        max_query_chars=1000,
        max_answer_chars=2000,
        max_context_chunks=5,
        log_raw_queries=False,
        query_rewriter=FixedRewriter("ตรวจสอบวันลาพักร้อนคงเหลือ"),
    )
    r = ChatService(build_graph(deps), 0.70).ask("ลาพักล้อนเหลือกี่วันดูไง", EMPLOYEE, "req_test_0001")

    assert r.status == "answered"
    assert [s.document_id for s in r.sources] == ["HR-004"]
    assert len(retriever.queries) == 2


def test_fallback_lists_available_topics() -> None:
    deps = GraphDeps(
        retriever=QueryAwareRetriever(),
        generator=ScriptedGenerator(draft=draft()),
        audit_sink=InMemoryAuditSink(),
        hash_salt="test-salt",
        max_query_chars=1000,
        max_answer_chars=2000,
        max_context_chunks=5,
        log_raw_queries=False,
        list_topics=lambda scopes: {"leave": ["ลาป่วย", "ลากิจ"]},
    )
    r = ChatService(build_graph(deps), 0.70).ask("ลาคลอดได้กี่วัน", EMPLOYEE, "req_test_0001")

    assert r.reason_code == "LOW_EVIDENCE"
    assert "ไม่มีข้อมูลส่วนนี้" in r.answer
    assert "• การลา: ลาป่วย, ลากิจ" in r.answer
