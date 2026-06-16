"""
Dense embedder implementations.

All implementations:
- batch requests to respect provider limits
- retry on transient failures with exponential backoff
- expose `.dimension` so the vector store can validate collection config
"""
from __future__ import annotations

import logging
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import DenseEmbedder

logger = logging.getLogger(__name__)


class OpenAIDenseEmbedder(DenseEmbedder):
    """Dense embeddings via the OpenAI API (or any OpenAI-compatible endpoint)."""

    # Known dimensions for common OpenAI embedding models.
    _KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 96,
        dimension: int | None = None,
    ):
        from openai import OpenAI  # imported lazily so this is an optional dep

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._batch_size = batch_size
        self._dimension = dimension or self._KNOWN_DIMENSIONS.get(model)
        if self._dimension is None:
            raise ValueError(
                f"Unknown dimension for model '{model}'. Pass `dimension=` explicitly."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self._model, input=batch)
        # API preserves input order, but sort by index defensively.
        ordered = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in ordered]

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors


class OllamaDenseEmbedder(DenseEmbedder):
    """Dense embeddings via a local Ollama instance.

    Ollama's /api/embeddings endpoint embeds one input at a time, so
    batching here is purely about controlled sequential calls + retries,
    not request batching.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,  # nomic-embed-text default
        timeout: float = 60.0,
    ):
        import requests

        self._requests = requests
        self._model = model
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._dimension = dimension
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _embed_one(self, text: str) -> List[float]:
        resp = self._requests.post(
            self._url,
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data.get("embedding")
        if embedding is None:
            raise RuntimeError(f"Ollama returned no embedding: {data}")
        return embedding

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerDenseEmbedder(DenseEmbedder):
    """Dense embeddings via a local sentence-transformers model.

    Useful when neither an OpenAI-compatible API nor Ollama is desired --
    runs fully in-process on CPU or GPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
