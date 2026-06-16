"""
Vector store implementations.

QdrantVectorStore is the production backend: one collection holds both a
named "dense" vector and a named "sparse" vector (with an IDF modifier
for BM25-style scoring) per point. Hypothetical questions are stored as
separate points (type="hypothetical_q") carrying only a dense vector and
a `parent_id` pointing back to the source chunk -- this is "Pattern A"
from the dual-embedding discussion: more entry points into the same
underlying content.

InMemoryVectorStore mirrors the same behaviour without external
dependencies, for local development and unit tests.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Chunk, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
class QdrantVectorStore(VectorStore):
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        collection_name: str,
        dense_dim: int,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        recreate: bool = False,
    ):
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection_name
        self._dense_dim = dense_dim

        if recreate or not self._client.collection_exists(collection_name):
            self._client.recreate_collection(
                collection_name=collection_name,
                vectors_config={
                    self.DENSE_VECTOR_NAME: models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
            logger.info("Created Qdrant collection '%s' (dense_dim=%d)", collection_name, dense_dim)

    # -- ingestion -----------------------------------------------------
    def upsert(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        hyp_question_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        models = self._models
        points: List[Any] = []

        for i, chunk in enumerate(chunks):
            vector: Dict[str, Any] = {self.DENSE_VECTOR_NAME: dense_vectors[i]}
            if sparse_vectors is not None:
                sv = sparse_vectors[i]
                vector[self.SPARSE_VECTOR_NAME] = models.SparseVector(
                    indices=list(sv.keys()), values=list(sv.values())
                )
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.id),
                    vector=vector,
                    payload={
                        "type": "chunk",
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        **chunk.metadata,
                    },
                )
            )

        if hyp_question_records:
            for rec in hyp_question_records:
                points.append(
                    models.PointStruct(
                        id=self._point_id(f"{rec['parent_id']}::{uuid.uuid4().hex[:8]}"),
                        vector={self.DENSE_VECTOR_NAME: rec["dense_vector"]},
                        payload={
                            "type": "hypothetical_q",
                            "parent_id": rec["parent_id"],
                            "text": rec["text"],
                        },
                    )
                )

        # 🆕 NEW: print what is about to be persisted
        print(f"\n  [QdrantVectorStore.upsert] writing {len(points)} points "
              f"to collection '{self._collection}':")
        print(f"    chunks        : {len(chunks)}")
        if hyp_question_records:
            print(f"    hyp_questions : {len(hyp_question_records)}")
        print(f"    sample point ID : {self._point_id(chunks[0].id) if chunks else 'n/a'}")
        if chunks:
            first_payload_keys = list(points[0].payload.keys())
            print(f"    payload keys (first point) : {first_payload_keys}")

        if points:
            self._client.upsert(collection_name=self._collection, points=points, wait=True)
            # 🆕 NEW: confirm
            info = self._client.get_collection(collection_name=self._collection)
            print(f"  [QdrantVectorStore.upsert] DONE. "
                  f"Collection now has {info.points_count} points total.")

    # -- retrieval -------------------------------------------------------
    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        models = self._models
        qdrant_filter = self._build_filter(filters)

        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using=self.DENSE_VECTOR_NAME,
                filter=qdrant_filter,
                limit=top_k * 4,
            )
        ]
        if sparse_vector:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(sparse_vector.keys()), values=list(sparse_vector.values())
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=top_k * 4,
                )
            )

        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=qdrant_filter,
            limit=top_k * 3,  # over-fetch; hypothetical-question hits collapse to parents
            with_payload=True,
        )

        return self._resolve_hits(result.points, top_k)

    # -- helpers -----------------------------------------------------
    def _resolve_hits(self, points: List[Any], top_k: int) -> List[RetrievedChunk]:
        """Collapse hypothetical-question hits onto their parent chunk,
        keeping the best score per parent, then truncate to top_k."""
        resolved: Dict[str, RetrievedChunk] = {}
        parent_ids_to_fetch: List[str] = []

        for point in points:
            payload = point.payload or {}
            if payload.get("type") == "hypothetical_q":
                parent_id = payload["parent_id"]
                parent_ids_to_fetch.append(parent_id)
                # Score recorded provisionally; text/metadata filled in below.
                if parent_id not in resolved or point.score > resolved[parent_id].score:
                    resolved[parent_id] = RetrievedChunk(
                        chunk_id=parent_id,
                        text="",
                        metadata={},
                        score=point.score,
                        matched_via="hypothetical_question",
                    )
            else:
                chunk_id = payload["chunk_id"]
                if chunk_id not in resolved or point.score > resolved[chunk_id].score:
                    resolved[chunk_id] = RetrievedChunk(
                        chunk_id=chunk_id,
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k not in ("text", "type", "chunk_id")},
                        score=point.score,
                        matched_via="dense_or_sparse",
                    )

        if parent_ids_to_fetch:
            fetched = self._client.retrieve(
                collection_name=self._collection,
                ids=[self._point_id(pid) for pid in set(parent_ids_to_fetch)],
                with_payload=True,
            )
            for record in fetched:
                payload = record.payload or {}
                chunk_id = payload.get("chunk_id")
                if chunk_id in resolved and not resolved[chunk_id].text:
                    existing = resolved[chunk_id]
                    resolved[chunk_id] = RetrievedChunk(
                        chunk_id=chunk_id,
                        text=payload.get("text", ""),
                        metadata={k: v for k, v in payload.items() if k not in ("text", "type", "chunk_id")},
                        score=existing.score,
                        matched_via=existing.matched_via,
                    )

        ranked = sorted(resolved.values(), key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    def _build_filter(self, filters: Optional[Dict[str, Any]]):
        if not filters:
            return None
        models = self._models
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
        return models.Filter(must=conditions)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        # Qdrant point IDs must be UUIDs or unsigned ints; derive a stable
        # UUID from the chunk_id so re-ingestion overwrites cleanly.
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


# --------------------------------------------------------------------------- #
# In-memory (dev / testing)
# --------------------------------------------------------------------------- #
class InMemoryVectorStore(VectorStore):
    """Pure-Python store for local development and unit tests.

    Implements the same hybrid search contract: dense cosine similarity,
    sparse dot product, fused via Reciprocal Rank Fusion, with
    hypothetical-question hits resolved to their parent chunk.
    """

    def __init__(self, rrf_k: int = 60):
        self._chunks: Dict[str, Chunk] = {}
        self._dense: Dict[str, np.ndarray] = {}
        self._sparse: Dict[str, Dict[int, float]] = {}
        self._hyp_questions: List[Dict[str, Any]] = []  # {parent_id, text, dense_vector}
        self._rrf_k = rrf_k

    def upsert(
        self,
        chunks: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]] = None,
        hyp_question_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        print(f"\n  [InMemoryVectorStore.upsert] storing {len(chunks)} chunks"
            + (f" + {len(hyp_question_records)} hyp_questions" if hyp_question_records else ""))

        for i, chunk in enumerate(chunks):
            self._chunks[chunk.id] = chunk
            self._dense[chunk.id] = np.array(dense_vectors[i], dtype=np.float32)
            if sparse_vectors is not None:
                self._sparse[chunk.id] = sparse_vectors[i]

            # 🆕 NEW: per-chunk detail
            if i < 3:  # only print first 3 to avoid spam
                print(f"    chunk[{i}] id={chunk.id} "
                    f"dense_dim={len(dense_vectors[i])} "
                    f"sparse_terms={len(sparse_vectors[i]) if sparse_vectors else 0}")

        if hyp_question_records:
            self._hyp_questions.extend(hyp_question_records)
            print(f"    hyp_questions total: {len(self._hyp_questions)}")

        print(f"  [InMemoryVectorStore.upsert] DONE. "
            f"Store now holds {len(self._chunks)} chunks, "
            f"{len(self._hyp_questions)} hypothetical questions.")

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        candidate_ids = [
            cid for cid, chunk in self._chunks.items() if self._matches_filter(chunk, filters)
        ]
        if not candidate_ids:
            return []

        query_dense = np.array(dense_vector, dtype=np.float32)
        dense_ranked = self._rank_by_cosine(query_dense, candidate_ids)

        sparse_ranked: List[str] = []
        if sparse_vector:
            sparse_ranked = self._rank_by_sparse_dot(sparse_vector, candidate_ids)

        # Hypothetical question matches -> resolve to parent_id immediately.
        hq_ranked = self._rank_hypothetical_questions(query_dense, set(candidate_ids))

        fused_scores = self._rrf_fuse([dense_ranked, sparse_ranked, hq_ranked])
        ranked_ids = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results = []
        for chunk_id, score in ranked_ids:
            chunk = self._chunks[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    score=score,
                    matched_via="fused",
                )
            )
        return results

    # -- ranking helpers -----------------------------------------------------
    def _rank_by_cosine(self, query: np.ndarray, candidate_ids: List[str]) -> List[str]:
        scores = []
        q_norm = query / (np.linalg.norm(query) + 1e-12)
        for cid in candidate_ids:
            vec = self._dense[cid]
            v_norm = vec / (np.linalg.norm(vec) + 1e-12)
            scores.append((cid, float(np.dot(q_norm, v_norm))))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return [cid for cid, _ in scores]

    def _rank_by_sparse_dot(self, query: Dict[int, float], candidate_ids: List[str]) -> List[str]:
        scores = []
        for cid in candidate_ids:
            doc_sparse = self._sparse.get(cid, {})
            score = sum(query.get(idx, 0.0) * w for idx, w in doc_sparse.items())
            scores.append((cid, score))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return [cid for cid, _ in scores]

    def _rank_hypothetical_questions(self, query: np.ndarray, allowed_ids: set) -> List[str]:
        if not self._hyp_questions:
            return []
        q_norm = query / (np.linalg.norm(query) + 1e-12)
        scores = []
        for rec in self._hyp_questions:
            if rec["parent_id"] not in allowed_ids:
                continue
            vec = np.array(rec["dense_vector"], dtype=np.float32)
            v_norm = vec / (np.linalg.norm(vec) + 1e-12)
            scores.append((rec["parent_id"], float(np.dot(q_norm, v_norm))))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        # collapse to first occurrence per parent_id (best score, already sorted)
        seen, ordered = set(), []
        for pid, _ in scores:
            if pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return ordered

    def _rrf_fuse(self, ranked_lists: List[List[str]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, item_id in enumerate(ranked):
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        return scores

    @staticmethod
    def _matches_filter(chunk: Chunk, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        return all(chunk.metadata.get(k) == v for k, v in filters.items())
