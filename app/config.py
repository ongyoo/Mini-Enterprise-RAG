from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_SALT = "dev-only-change-me"


class DemoUser(BaseModel):
    actor_id: str
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)


def _default_demo_users() -> dict[str, DemoUser]:
    return {
        "employee-demo-token": DemoUser(
            actor_id="emp_demo", roles=["employee"], scopes=["public", "internal"]
        ),
        "admin-demo-token": DemoUser(
            actor_id="admin_demo",
            roles=["employee", "admin"],
            scopes=["public", "internal", "confidential"],
        ),
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "test", "prod"] = "local"
    log_level: str = "INFO"

    llm_provider: Literal["openai", "azure", "anthropic", "fake"] = "openai"
    embedding_provider: Literal["openai", "azure", "local", "hashing"] | None = None
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = ""
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""
    anthropic_api_key: SecretStr | None = None
    anthropic_chat_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    local_embedding_cache_dir: Path = Path(".cache/fastembed")
    embedding_dimensions: int = Field(default=1536, gt=0)

    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_timeout_s: float = Field(default=20.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

    database_url: str = "postgresql+psycopg://app:local_password@localhost:5432/rag"

    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_similarity_threshold: float = Field(default=0.70, ge=-1.0, le=1.0)
    rag_max_context_chunks: int = Field(default=5, ge=1, le=10)
    rag_chunk_size_chars: int = Field(default=800, ge=100)
    rag_chunk_overlap_chars: int = Field(default=80, ge=0)

    documents_path: Path = Path("data/documents.jsonl")
    evaluation_path: Path = Path("data/evaluation.jsonl")

    max_query_chars: int = Field(default=1000, ge=10, le=5000)
    max_answer_chars: int = Field(default=2000, ge=100)
    log_raw_queries: bool = False
    audit_hash_salt: SecretStr = SecretStr(_DEV_SALT)

    demo_users: dict[str, DemoUser] = Field(default_factory=_default_demo_users)

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.app_env == "prod" and self.audit_hash_salt.get_secret_value() == _DEV_SALT:
            raise ValueError("AUDIT_HASH_SALT must be set in prod")
        if self.rag_chunk_overlap_chars >= self.rag_chunk_size_chars:
            raise ValueError("RAG_CHUNK_OVERLAP_CHARS must be smaller than RAG_CHUNK_SIZE_CHARS")
        return self

    @property
    def chat_model_name(self) -> str:
        match self.llm_provider:
            case "openai":
                return self.openai_chat_model
            case "azure":
                return self.azure_openai_chat_deployment
            case "anthropic":
                return self.anthropic_chat_model
            case "fake":
                return "fake-extractive"

    @property
    def embedding_backend(self) -> Literal["openai", "azure", "local", "hashing"]:
        if self.embedding_provider:
            return self.embedding_provider
        match self.llm_provider:
            case "openai" | "azure":
                return self.llm_provider
            case "anthropic":
                # Anthropic has no embeddings API.
                return "local"
            case "fake":
                return "hashing"

    @property
    def embedding_model_name(self) -> str:
        match self.embedding_backend:
            case "openai":
                return self.openai_embedding_model
            case "azure":
                return self.azure_openai_embedding_deployment
            case "local":
                return self.local_embedding_model
            case "hashing":
                return "hashing-ngram"


@lru_cache
def get_settings() -> Settings:
    return Settings()
