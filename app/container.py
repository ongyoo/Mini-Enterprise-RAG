from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.auth import Actor
from app.config import Settings
from app.db.repositories import DocumentRepository
from app.db.session import build_engine, build_session_factory, session_scope
from app.graph.build import build_graph
from app.graph.nodes import GraphDeps
from app.rag.embeddings import build_embeddings
from app.rag.generator import build_generator
from app.rag.retriever import PgVectorRetriever
from app.rag.rewriter import build_rewriter
from app.services.audit import DbAuditSink
from app.services.chat import ChatService

EVAL_ACTOR = Actor("eval_runner", ("employee",), ("public", "internal"))


@dataclass
class Container:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    embeddings: Embeddings
    chat_service: ChatService


def _list_topics(session_factory: sessionmaker[Session], scopes: list[str]) -> dict[str, list[str]]:
    with session_scope(session_factory) as session:
        return DocumentRepository(session).topics(scopes)


def build_container(settings: Settings) -> Container:
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)
    embeddings = build_embeddings(settings)
    deps = GraphDeps(
        retriever=PgVectorRetriever(session_factory, embeddings, settings.rag_top_k),
        generator=build_generator(settings),
        audit_sink=DbAuditSink(session_factory),
        hash_salt=settings.audit_hash_salt.get_secret_value(),
        max_query_chars=settings.max_query_chars,
        max_answer_chars=settings.max_answer_chars,
        max_context_chunks=settings.rag_max_context_chunks,
        log_raw_queries=settings.log_raw_queries,
        query_rewriter=build_rewriter(settings),
        list_topics=lambda scopes: _list_topics(session_factory, scopes),
    )
    chat_service = ChatService(build_graph(deps), settings.rag_similarity_threshold)
    return Container(settings, engine, session_factory, embeddings, chat_service)
