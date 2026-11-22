from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import anthropic
import httpx
import openai
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, Field

from app.config import Settings
from app.domain import ConfigurationError, GenerationParseError, ProviderUnavailableError
from app.graph.state import RetrievedChunk, TokenUsage
from app.rag.prompts import (
    HUMAN_PROMPT,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    format_sources,
    source_label,
)


class GenerationOutput(BaseModel):
    answer: str = Field(description="คำตอบภาษาไทย หรือคำถามกลับเมื่อ needs_clarification")
    cited_source_ids: list[str] = Field(description="source ids ที่ใช้ เช่น ['S1']")
    cannot_answer: bool
    needs_clarification: bool


@dataclass(frozen=True)
class Draft:
    answer: str
    cited_chunk_ids: list[str]
    cannot_answer: bool
    needs_clarification: bool
    model: str
    token_usage: TokenUsage = field(default_factory=lambda: TokenUsage())


class Generator(Protocol):
    model_name: str

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Draft: ...


def _labels_to_chunk_ids(labels: list[str], chunks: list[RetrievedChunk]) -> list[str]:
    by_label = {source_label(i): c["chunk_id"] for i, c in enumerate(chunks)}
    # Unknown labels are passed through so output verification rejects them.
    return [by_label.get(label.strip(), label) for label in labels]


class LLMGenerator:
    def __init__(self, llm: BaseChatModel, model_name: str) -> None:
        self.model_name = model_name
        self._chain = RAG_PROMPT | llm.with_structured_output(GenerationOutput, include_raw=True)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Draft:
        try:
            result: dict[str, Any] = self._chain.invoke(  # type: ignore[assignment]
                {"sources": format_sources(chunks), "question": question}
            )
        except (openai.APIError, httpx.HTTPError) as exc:
            raise ProviderUnavailableError("chat model call failed") from exc

        parsed = result.get("parsed")
        if not isinstance(parsed, GenerationOutput):
            raise GenerationParseError("model returned unparseable output")

        usage = getattr(result.get("raw"), "usage_metadata", None) or {}
        return Draft(
            answer=parsed.answer,
            cited_chunk_ids=_labels_to_chunk_ids(parsed.cited_source_ids, chunks),
            cannot_answer=parsed.cannot_answer,
            needs_clarification=parsed.needs_clarification,
            model=self.model_name,
            token_usage=TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )


class ClaudeGenerator:
    """Calls Claude through the Anthropic SDK with a Pydantic output schema."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model_name: str,
        effort: Literal["low", "medium", "high"],
    ) -> None:
        self.model_name = model_name
        self._client = client
        self._effort = effort

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Draft:
        prompt = HUMAN_PROMPT.format(sources=format_sources(chunks), question=question)
        try:
            response = self._client.beta.messages.parse(
                model=self.model_name,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=GenerationOutput,
                output_config={"effort": self._effort},
                # On a safety decline, Anthropic retries on its recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.APIError as exc:
            raise ProviderUnavailableError("chat model call failed") from exc

        parsed = response.parsed_output
        if response.stop_reason == "refusal" or not isinstance(parsed, GenerationOutput):
            raise GenerationParseError(f"no structured output (stop_reason={response.stop_reason})")

        usage = response.usage
        return Draft(
            answer=parsed.answer,
            cited_chunk_ids=_labels_to_chunk_ids(parsed.cited_source_ids, chunks),
            cannot_answer=parsed.cannot_answer,
            needs_clarification=parsed.needs_clarification,
            model=response.model,
            token_usage=TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
        )


class ExtractiveFakeGenerator:
    """Offline generator: quotes the best chunk. For tests and keyless demos only."""

    model_name = "fake-extractive"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> Draft:
        best = chunks[0]
        body = best["text"].split("\n", 1)[-1].strip()
        return Draft(
            answer=f"ตามนโยบายจำลอง {body[:400]}",
            cited_chunk_ids=[best["chunk_id"]],
            cannot_answer=False,
            needs_clarification=False,
            model=self.model_name,
        )


def build_generator(settings: Settings) -> Generator:
    common = {
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
    match settings.llm_provider:
        case "openai":
            if settings.openai_api_key is None or not settings.openai_chat_model:
                raise ConfigurationError("OPENAI_API_KEY and OPENAI_CHAT_MODEL are required")
            llm: BaseChatModel = ChatOpenAI(
                model=settings.openai_chat_model, api_key=settings.openai_api_key, **common
            )
            return LLMGenerator(llm, settings.openai_chat_model)
        case "azure":
            if settings.azure_openai_api_key is None or not settings.azure_openai_chat_deployment:
                raise ConfigurationError(
                    "AZURE_OPENAI_API_KEY and AZURE_OPENAI_CHAT_DEPLOYMENT are required"
                )
            llm = AzureChatOpenAI(
                azure_deployment=settings.azure_openai_chat_deployment,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                **common,
            )
            return LLMGenerator(llm, settings.azure_openai_chat_deployment)
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise ConfigurationError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value(),
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
            return ClaudeGenerator(client, settings.anthropic_chat_model, settings.anthropic_effort)
        case "fake":
            return ExtractiveFakeGenerator()
