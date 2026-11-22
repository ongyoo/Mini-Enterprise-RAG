from fastapi import APIRouter

from app.api.deps import ActorDep, ContainerDep, RequestIdDep
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest, actor: ActorDep, container: ContainerDep, request_id: RequestIdDep
) -> ChatResponse:
    return container.chat_service.ask(body.query, actor, request_id)
