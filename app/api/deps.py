from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth import Actor, resolve_token
from app.container import Container

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_request_id(request: Request) -> str:
    request_id: str = request.state.request_id
    return request_id


def current_actor(
    container: Annotated[Container, Depends(get_container)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Actor:
    actor = resolve_token(credentials.credentials, container.settings) if credentials else None
    if actor is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing token")
    return actor


def require_admin(actor: Annotated[Actor, Depends(current_actor)]) -> Actor:
    if not actor.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
    return actor


ContainerDep = Annotated[Container, Depends(get_container)]
ActorDep = Annotated[Actor, Depends(current_actor)]
AdminDep = Annotated[Actor, Depends(require_admin)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
