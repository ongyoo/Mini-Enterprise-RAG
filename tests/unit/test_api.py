from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import Harness

EMPLOYEE_TOKEN = {"Authorization": "Bearer employee-demo-token"}


@pytest.fixture
def client(harness: Harness) -> TestClient:
    container = SimpleNamespace(
        settings=Settings(),
        engine=MagicMock(),
        session_factory=MagicMock(),
        embeddings=MagicMock(),
        chat_service=harness.service,
    )
    with TestClient(create_app(container)) as c:  # type: ignore[arg-type]
        yield c


def test_chat_requires_auth(client: TestClient) -> None:
    r = client.post("/api/v1/chat", json={"query": "ลาป่วยทำไง"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "HTTP_401"
    assert r.json()["request_id"].startswith("req_")


def test_chat_response_contract(client: TestClient) -> None:
    r = client.post("/api/v1/chat", json={"query": "ลาป่วยทำไง"}, headers=EMPLOYEE_TOKEN)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert body["sources"][0]["document_id"] == "HR-LEAVE-001"
    assert body["retrieval"] == {"best_similarity": 0.82, "threshold": 0.70}


def test_caller_cannot_inject_role_or_scopes(client: TestClient) -> None:
    r = client.post(
        "/api/v1/chat",
        json={"query": "ลาป่วยทำไง", "role": "admin", "scopes": ["confidential"]},
        headers=EMPLOYEE_TOKEN,
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_admin_routes_reject_employees(client: TestClient) -> None:
    assert client.post("/api/v1/admin/ingest", headers=EMPLOYEE_TOKEN).status_code == 403
    assert client.get("/api/v1/admin/audit-logs", headers=EMPLOYEE_TOKEN).status_code == 403


def test_unsafe_request_id_is_replaced(client: TestClient) -> None:
    r = client.get("/health/live", headers={"X-Request-ID": "<script>"})
    assert r.headers["X-Request-ID"].startswith("req_")


def test_chat_page_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
