from typing import Protocol

import httpx
import openai
from langchain_core.embeddings import Embeddings
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db.repositories import ChunkSearchRow, DocumentRepository
from app.domain import ProviderUnavailableError, RetrievalUnavailableError
from app.graph.state import RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, scopes: list[str]) -> list[RetrievedChunk]: ...


class PgVectorRetriever:
    """Cosine search over pgvector, filtered by status and the caller's access scopes.

    Chat examples are not citable, but they map slang to the SOPs they relate to.
    When a chat example is hit, chunks of its related SOPs are fetched as well.
    """

    def __init__(
        self, session_factory: sessionmaker[Session], embeddings: Embeddings, top_k: int
    ) -> None:
        self._session_factory = session_factory
        self._embeddings = embeddings
        self._top_k = top_k

    def retrieve(self, query: str, scopes: list[str]) -> list[RetrievedChunk]:
        try:
            vector = self._embeddings.embed_query(query)
        except (openai.APIError, httpx.HTTPError) as exc:
            raise ProviderUnavailableError("embedding call failed") from exc

        try:
            with self._session_factory() as session:
                repo = DocumentRepository(session)
                hits = repo.search_chunks(vector, scopes, self._top_k)

                found_docs = {h.document_code for h in hits}
                related = {
                    code
                    for h in hits
                    if h.source_authority == "example"
                    for code in h.related_document_codes
                    if code not in found_docs
                }
                if related:
                    hits += repo.search_chunks(
                        vector,
                        scopes,
                        self._top_k,
                        document_codes=sorted(related),
                        authoritative_only=True,
                    )
        except SQLAlchemyError as exc:
            raise RetrievalUnavailableError("vector search failed") from exc

        return [_to_chunk(h) for h in hits]


def _to_chunk(row: ChunkSearchRow) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=str(row.chunk_id),
        chunk_key=row.chunk_key,
        document_id=row.document_code,
        title=row.title,
        version=row.version,
        doc_type=row.type,
        access_level=row.access_level,
        source_authority="authoritative" if row.source_authority == "authoritative" else "example",
        text=row.content,
        similarity=round(row.similarity, 4),
    )
