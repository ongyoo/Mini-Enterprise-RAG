import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import text

from app.api import admin, chat, documents
from app.config import get_settings
from app.container import Container, build_container
from app.logging_config import configure_logging
from app.schemas.responses import ErrorBody, ErrorResponse

log = logging.getLogger(__name__)
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_CHAT_PAGE = Path(__file__).parent / "web" / "chat.html"


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is None:
            settings = get_settings()
            configure_logging(settings.log_level)
            app.state.container = build_container(settings)
        else:
            app.state.container = container
        yield
        app.state.container.engine.dispose()

    app = FastAPI(title="Mini Enterprise RAG (synthetic data)", version="0.1.0", lifespan=lifespan)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(admin.router)

    @app.middleware("http")
    async def request_id_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = (
            incoming if _SAFE_REQUEST_ID.match(incoming) else f"req_{uuid.uuid4().hex}"
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def error(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
        body = ErrorResponse(
            request_id=getattr(request.state, "request_id", "unknown"),
            error=ErrorBody(code=code, message=message),
        )
        return JSONResponse(status_code=status_code, content=body.model_dump())

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return error(request, exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = ", ".join(".".join(map(str, e["loc"])) for e in exc.errors())
        return error(request, 422, "VALIDATION_ERROR", f"invalid fields: {fields}")

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"request_id": request.state.request_id})
        return error(request, 500, "INTERNAL_ERROR", "internal error")

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(_CHAT_PAGE)

    @app.get("/health/live", tags=["health"])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request) -> JSONResponse:
        c: Container = request.app.state.container
        try:
            with c.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
        payload = {
            "database": "ok" if db_ok else "unavailable",
            "provider": c.settings.llm_provider,
            "chat_model": c.settings.chat_model_name,
        }
        return JSONResponse(status_code=200 if db_ok else 503, content=payload)

    return app


app = create_app()
