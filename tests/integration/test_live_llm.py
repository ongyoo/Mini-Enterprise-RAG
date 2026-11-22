"""Calls the real provider configured in .env: `make test-live`. Costs a few cents."""

import pytest

from app.config import Settings
from app.rag.embeddings import build_embeddings
from app.rag.generator import build_generator
from tests.conftest import make_chunk

pytestmark = pytest.mark.live


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    if s.llm_provider == "fake":
        pytest.skip("set LLM_PROVIDER=openai|azure for live tests")
    return s


def test_embedding_dimensions_match_config(settings: Settings) -> None:
    vector = build_embeddings(settings).embed_query("ลาป่วยทำไง")
    assert len(vector) == settings.embedding_dimensions


def test_structured_generation_cites_context(settings: Settings) -> None:
    chunk = make_chunk()
    out = build_generator(settings).generate("ลาป่วยต้องแจ้งก่อนกี่โมง", [chunk])
    assert not out.cannot_answer
    assert out.cited_chunk_ids == [chunk["chunk_id"]]
    assert "09:00" in out.answer
