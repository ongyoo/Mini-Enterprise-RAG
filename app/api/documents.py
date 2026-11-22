from fastapi import APIRouter

from app.api.deps import ActorDep, ContainerDep
from app.db.repositories import DocumentRepository
from app.schemas.responses import DocumentInfo

router = APIRouter(prefix="/api/v1", tags=["documents"])


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(actor: ActorDep, container: ContainerDep) -> list[DocumentInfo]:
    with container.session_factory() as session:
        docs = DocumentRepository(session).list_visible(actor.scopes)
        return [
            DocumentInfo(
                document_id=d.document_code,
                title=d.title,
                type=d.type,
                category=d.category,
                version=d.version,
                access_level=d.access_level,
                source_authority=d.source_authority,
                synthetic=d.synthetic,
                updated_at=d.source_updated_at,
            )
            for d in docs
        ]
