from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from app.config import Settings
from app.domain import GenerationParseError, ProviderUnavailableError
from app.rag.embeddings import HashingEmbeddings, build_embeddings
from app.rag.generator import ClaudeGenerator, GenerationOutput
from tests.conftest import make_chunk


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def make_generator(messages: FakeMessages) -> ClaudeGenerator:
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeGenerator(client, "claude-opus-5-5", "low")  # type: ignore[arg-type]


def response(parsed: GenerationOutput | None, stop_reason: str = "end_turn") -> Any:
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


def test_maps_source_labels_to_chunk_ids() -> None:
    chunk = make_chunk()
    output = GenerationOutput(
        answer="ตามนโยบายจำลอง แจ้งหัวหน้างานก่อน 09:00 น.",
        cited_source_ids=["S1"],
        cannot_answer=False,
        needs_clarification=False,
    )
    messages = FakeMessages(response(output))

    draft = make_generator(messages).generate("ลาป่วยทำไง", [chunk])

    assert draft.cited_chunk_ids == [chunk["chunk_id"]]
    assert draft.token_usage["total_tokens"] == 120
    call = messages.calls[0]
    assert call["output_format"] is GenerationOutput
    assert "<question>ลาป่วยทำไง</question>" in call["messages"][0]["content"]


def test_refusal_is_a_parse_error() -> None:
    messages = FakeMessages(response(None, stop_reason="refusal"))
    with pytest.raises(GenerationParseError):
        make_generator(messages).generate("q", [make_chunk()])


def test_api_error_is_provider_unavailable() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    messages = FakeMessages(error=anthropic.APIConnectionError(request=request))
    with pytest.raises(ProviderUnavailableError):
        make_generator(messages).generate("q", [make_chunk()])


def test_embedding_backend_defaults() -> None:
    def backend(provider: str) -> str:
        return Settings(llm_provider=provider, _env_file=None).embedding_backend  # type: ignore[call-arg,arg-type]

    assert backend("anthropic") == "local"
    assert backend("fake") == "hashing"
    assert backend("openai") == "openai"


def test_fake_provider_uses_hashing_embeddings() -> None:
    settings = Settings(llm_provider="fake", _env_file=None)  # type: ignore[call-arg]
    assert isinstance(build_embeddings(settings), HashingEmbeddings)
