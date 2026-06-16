"""
Core data models and abstract provider interfaces for the RAG pipeline.

Everything downstream (ingestion, retrieval) is written against these
interfaces only. Swapping embedding models, vector stores, or LLM
providers means writing a new class here and wiring it in config.py --
no changes to ingestion.py or retrieval.py are required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A single unit of text produced by the chunking stage.

    By the time a Chunk reaches the embedding stage it is fully enriched:
    metadata has been attached and hypothetical questions (if enabled)
    have been generated.
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    hypothetical_questions: List[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, with provenance and score."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    matched_via: str  # "dense" | "sparse" | "hypothetical_question" | "fused"


class DenseEmbedder(ABC):
    """Embeds text into a fixed-size dense vector (semantic similarity)."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one dense vector per input text, in order."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality. Must match the vector store collection."""


class SparseEmbedder(ABC):
    """Embeds text into a sparse term-weight vector (lexical / exact match)."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[Dict[int, float]]:
        """Return one sparse vector (token_id -> weight) per input text."""


class LLMProvider(ABC):
    """Generic text-generation provider, used for hypothetical question
    generation and (optionally) answer synthesis."""

    @abstractmethod
    def generate(
        self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        ...


class VectorStore(ABC):
    """Storage + retrieval backend. Must support both dense and sparse
    vectors on the same record so hybrid search can run as a single
    query against one collection."""

    @abstractmethod
    def upsert(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        hyp_question_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist chunk vectors + metadata.

        hyp_question_records, if provided, is a flat list of:
            {"parent_id": chunk.id, "text": question, "dense_vector": [...]}
        each stored as its own retrievable record pointing back to parent_id.
        """

    @abstractmethod
    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Run dense + sparse search (fusion handled by the store or by
        the retrieval pipeline, depending on backend) and return results
        already resolved to parent chunk text."""
        pass
