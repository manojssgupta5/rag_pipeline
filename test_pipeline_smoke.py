"""
Sanity test using mock embedders -- validates chunking, metadata
enrichment, hypothetical question generation, dual embedding, storage,
and hybrid retrieval all wire together correctly without requiring
network access, API keys, or model downloads.

Run: python test_pipeline_smoke.py
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

from rag_pipeline.base import DenseEmbedder, LLMProvider, SparseEmbedder
from rag_pipeline.ingestion import (
    HypotheticalQuestionConfig,
    HypotheticalQuestionGenerator,
    IngestionPipeline,
    MetadataEnricher,
    RecursiveTextChunker,
)
from rag_pipeline.providers.vector_store import InMemoryVectorStore
from rag_pipeline.retrieval import RetrievalPipeline


class MockDenseEmbedder(DenseEmbedder):
    """Deterministic bag-of-words style embedding: hashes each token into
    a fixed-size vector. Similar text -> similar vectors, with zero
    external dependencies."""

    def __init__(self, dim: int = 32):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class MockSparseEmbedder(SparseEmbedder):
    def embed(self, texts: List[str]) -> List[Dict[int, float]]:
        results = []
        for text in texts:
            counts: Dict[int, float] = {}
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2 ** 16)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            results.append(counts)
        return results


class MockLLM(LLMProvider):
    """Returns a few canned questions regardless of input -- enough to
    exercise the hypothetical-question code path end to end."""

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        if "free delivery" in prompt.lower():
            return "Is there free delivery in Dubai Marina?\nWhat is the minimum order for free delivery?"
        if "refund" in prompt.lower():
            return "How long do refunds take?\nWhat happens if my order arrives damaged?"
        return "What does this passage describe?"


SAMPLE_DOC = """
Free delivery is available on all orders above AED 100 within the Dubai Marina zone.

Refunds for cancelled orders are processed within 3-5 business days to the original payment method.
"""


def main() -> None:
    dense = MockDenseEmbedder()
    sparse = MockSparseEmbedder()
    store = InMemoryVectorStore()

    ingestion = IngestionPipeline(
        dense_embedder=dense,
        sparse_embedder=sparse,
        vector_store=store,
        chunker=RecursiveTextChunker(chunk_size=120, chunk_overlap=20),
        metadata_enricher=MetadataEnricher(),
        hypothetical_question_generator=HypotheticalQuestionGenerator(
            llm=MockLLM(), config=HypotheticalQuestionConfig(questions_per_chunk=2)
        ),
        debug=True,
    )

    n = ingestion.ingest_document(
        doc_id="delivery-policy-v3",
        text=SAMPLE_DOC,
        document_metadata={"source": "help_center", "region": "dubai_marina"},
    )
    print(f"Ingested {n} chunks")
    assert n > 0

    retrieval = RetrievalPipeline(dense_embedder=dense, sparse_embedder=sparse, vector_store=store)

    for query in ["free delivery dubai marina", "refund timeline"]:
        results = retrieval.retrieve(query, top_k=2, filters={"region": "dubai_marina"})
        print(f"\nQuery: {query!r} -> {len(results)} result(s)")
        for r in results:
            print(f"  [{r.score:.4f}] ({r.matched_via}) {r.text.strip()[:80]}")
        assert results, f"expected at least one result for query: {query}"

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
