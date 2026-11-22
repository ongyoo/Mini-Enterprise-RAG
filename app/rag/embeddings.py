import hashlib
import math

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from app.config import Settings
from app.domain import ConfigurationError


class HashingEmbeddings(Embeddings):
    """Offline character n-gram embeddings (no API key needed).

    Only good enough to exercise the pipeline; similarity scores are not comparable
    with OpenAI embeddings, so calibrate the threshold separately.
    """

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        compact = "".join(text.lower().split())
        for n in (2, 3):
            for i in range(len(compact) - n + 1):
                digest = hashlib.blake2b(compact[i : i + n].encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vec[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LocalEmbeddings(Embeddings):
    """Multilingual sentence-transformer run locally with fastembed (ONNX, CPU)."""

    def __init__(self, model_name: str, cache_dir: str, dimensions: int) -> None:
        model_dim = next(
            (m["dim"] for m in TextEmbedding.list_supported_models() if m["model"] == model_name),
            None,
        )
        if model_dim != dimensions:
            raise ConfigurationError(
                f"{model_name} outputs {model_dim} dimensions, EMBEDDING_DIMENSIONS={dimensions}"
            )
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(settings: Settings) -> Embeddings:
    match settings.embedding_backend:
        case "openai":
            if settings.openai_api_key is None:
                raise ConfigurationError("OPENAI_API_KEY is required for OpenAI embeddings")
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                dimensions=settings.embedding_dimensions,
                openai_api_key=settings.openai_api_key,
                request_timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        case "azure":
            if settings.azure_openai_api_key is None or not settings.azure_openai_endpoint:
                raise ConfigurationError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY required")
            return AzureOpenAIEmbeddings(
                deployment=settings.azure_openai_embedding_deployment,
                azure_endpoint=settings.azure_openai_endpoint,
                openai_api_key=settings.azure_openai_api_key,
                openai_api_version=settings.azure_openai_api_version,
                dimensions=settings.embedding_dimensions,
                request_timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            )
        case "local":
            return LocalEmbeddings(
                settings.local_embedding_model,
                str(settings.local_embedding_cache_dir),
                settings.embedding_dimensions,
            )
        case "hashing":
            return HashingEmbeddings(settings.embedding_dimensions)
