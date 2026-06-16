"""
Retrieval pipeline.

This is the "dumb but fast" layer described earlier: it embeds the query
(dense + sparse) and calls the vector store's hybrid search. It has no
awareness of query intent -- that belongs to an agent/planner layer built
on top of this pipeline's output.

An optional reranker hook is provided as an extension point (e.g. a
cross-encoder for precision-sensitive use cases) but is not required for
basic hybrid retrieval.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import DenseEmbedder, LLMProvider, RetrievedChunk, SparseEmbedder, VectorStore

logger = logging.getLogger(__name__)

Reranker = Callable[[str, List[RetrievedChunk]], List[RetrievedChunk]]


class RetrievalPipeline:
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        vector_store: VectorStore,
        sparse_embedder: Optional[SparseEmbedder] = None,
        reranker: Optional[Reranker] = None,
    ):
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._store = vector_store
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("query must be non-empty")

        dense_vector = self._dense.embed([query])[0]
        sparse_vector = self._sparse.embed([query])[0] if self._sparse else None

        # Over-fetch before reranking so the reranker has a meaningful
        # candidate pool to work with.
        fetch_k = top_k * 3 if self._reranker else top_k

        results = self._store.hybrid_search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=fetch_k,
            filters=filters,
        )

        print("\n=== BEFORE RERANK ===")
        for i, r in enumerate(results, 1):
            print(
                f"{i}. score={getattr(r, 'score', None)} "
                f"text={r.text[:200]}"
            )

        if self._reranker and results:
            results = self._reranker(query, results)
            print("\n=== AFTER RERANK ===")
            for i, r in enumerate(results, 1):
                print(f"\n====================")
                print(f"Rank {i}")
                print(f"Score: {r.score}")
                print(r.text)

        return results[:top_k]

class SelfQueryRetriever:
    """Wraps a RetrievalPipeline to automatically extract filters from natural language queries."""

    _PROMPT = """You are a query constructor for a vector search engine.
Given a user query, extract the core semantic search query and any applicable metadata filters.
The allowed metadata fields are: 'year' (integer), 'domain' (string), 'paper' (string), 'source' (string).

Return ONLY valid JSON in this format:
{{
    "search_query": "the core search phrase",
    "filters": {{
        "year": 2017
    }}
}}
If no filters apply, return an empty dictionary for filters.

User query:
"{query}"
"""

    def __init__(self, retriever: RetrievalPipeline, llm: LLMProvider):
        self._retriever = retriever
        self._llm = llm

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        # Only self-query if no filters are explicitly provided
        if not filters:
            try:
                prompt = self._PROMPT.format(query=query)
                raw = self._llm.generate(prompt, max_tokens=128, temperature=0.0)
                raw = raw.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
                parsed = json.loads(raw)

                search_query = parsed.get("search_query", query)
                llm_filters = parsed.get("filters", {})
                
                # Filter out empty or null values
                llm_filters = {k: v for k, v in llm_filters.items() if v is not None and v != ""}

                print(f"   → Self-Query Extracted - search_query: {search_query!r}, filters: {llm_filters}")
                query = search_query
                filters = llm_filters
            except Exception as e:
                logger.warning(f"Self-Query extraction failed: {e}. Falling back to raw query.")

        return self._retriever.retrieve(query=query, top_k=top_k, filters=filters)

class AnswerSynthesizer:
    """Wraps a retriever to synthesize a final answer using an LLM over the retrieved chunks."""

    _PROMPT = """You are a helpful and precise assistant. Use the following context chunks to answer the user's question.
If the answer is not contained in the context, say "I cannot answer this based on the provided documents."
Do NOT use outside knowledge.

Context Chunks:
{context}

User Question:
{query}

Answer:"""

    def __init__(self, retriever: Any, llm: LLMProvider):
        # We use Any for retriever because it could be RetrievalPipeline or SelfQueryRetriever
        self._retriever = retriever
        self._llm = llm

    def ask(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Returns the final answer and the source chunks used to generate it."""
        chunks = self._retriever.retrieve(query=query, top_k=top_k, filters=filters)
        
        if not chunks:
            return {"answer": "No relevant documents found.", "chunks": []}

        # Format context from the chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Chunk {i} - ID: {chunk.chunk_id}]\n{chunk.text}\n")
        
        context_str = "\n".join(context_parts)
        prompt = self._PROMPT.format(context=context_str, query=query)

        try:
            answer = self._llm.generate(prompt, max_tokens=1024, temperature=0.0)
            answer = answer.strip()
            print(f"   → Synthesized answer (used {len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            answer = "Failed to generate answer."

        return {
            "answer": answer,
            "chunks": chunks
        }

