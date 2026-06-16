"""
Multi-agent orchestrator layer.

Sits above the retrieval pipeline and coordinates four specialized agents:

    IntentAgent      – classifies the query and decomposes compound queries
                       into focused sub-queries.
    VerifierAgent    – LLM judge: given retrieved chunks, is the query
                       answerable? If not, suggests a refined search query.
    SynthesisAgent   – generates a cited final answer from verified chunks.
    Orchestrator     – drives the full flow with a verify-and-retry loop.

All agents degrade gracefully: if an LLM call fails the orchestrator
falls back to sensible defaults rather than crashing.

Relationship to existing retrieval stack
-----------------------------------------
The Orchestrator wraps a ``RetrievalPipeline`` (or any object with a
``retrieve(query, top_k, filters)`` method) directly.  It intentionally
bypasses ``SelfQueryRetriever`` — the ``IntentAgent`` handles the same
query-understanding work but with richer output (intent type, multiple
sub-queries).  The existing ``AnswerSynthesizer`` in retrieval.py is left
untouched; ``SynthesisAgent`` here adds chunk citations.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import LLMProvider, RetrievedChunk

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class OrchestratorConfig:
    """Runtime knobs for the Orchestrator."""

    max_retries: int = 2
    """Max times the verify-and-retry loop will re-retrieve if the verifier
    is not satisfied.  Each retry uses the verifier's suggested refined query."""

    top_k: int = 5
    """Number of chunks to retrieve per sub-query."""

    enable_decomposition: bool = True
    """Let the IntentAgent split compound queries into sub-queries.
    Disable to always treat the query as a single unit."""

    enable_verification: bool = True
    """Run VerifierAgent after retrieval.  Disable for lower latency when
    answer quality checking is not needed."""

    verification_confidence_threshold: float = 0.6
    """Minimum verifier confidence to accept the answer without retrying."""

    parallel_retrieval_timeout: float = 30.0
    """Seconds to wait for each parallel sub-query retrieval."""


# --------------------------------------------------------------------------- #
# Result dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class IntentResult:
    """Output of the IntentAgent."""

    intent_type: str
    """One of: factual | comparison | summarization | compound | unanswerable"""

    sub_queries: List[str]
    """Decomposed sub-queries.  For non-compound queries this is [original_query]."""

    reasoning: str = ""
    """LLM's brief reasoning (useful for debugging)."""


@dataclass
class VerificationResult:
    """Output of the VerifierAgent."""

    is_answerable: bool
    """True when the retrieved context sufficiently answers the query."""

    confidence: float
    """0.0 – 1.0. Verifier's self-reported confidence."""

    feedback: str = ""
    """What is missing, or why the context is sufficient."""

    refined_query: Optional[str] = None
    """Suggested refined search query when ``is_answerable`` is False."""


@dataclass
class OrchestratorResult:
    """Full output returned by ``Orchestrator.ask()``."""

    answer: str
    chunks: List[RetrievedChunk]
    intent: IntentResult
    verification: VerificationResult
    attempts: int
    """Number of retrieve-verify cycles that ran (1 = first attempt succeeded)."""

    sub_query_results: Dict[str, List[RetrievedChunk]] = field(default_factory=dict)
    """Per-sub-query chunk lists before merging, for inspection / debugging."""


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
class IntentAgent:
    """Classifies the query and optionally decomposes compound queries.

    Falls back silently when the LLM call fails: returns intent_type='factual'
    and sub_queries=[original_query].
    """

    _PROMPT = """\
You are a query analyzer for a RAG (Retrieval-Augmented Generation) system.

Analyze the user's query and:
1. Classify its intent type.
2. If it is a compound query (contains multiple distinct questions), \
decompose it into focused sub-queries; otherwise return the original query as-is.

Intent types:
  "factual"        – single factual question with a specific answer
  "comparison"     – comparing two or more concepts / items
  "summarization"  – request for an overview or summary of a topic
  "compound"       – multiple distinct questions packed into one query
  "unanswerable"   – too vague, out-of-scope, or clearly not document-answerable

Return ONLY valid JSON, no markdown fences:
{{
    "intent_type": "factual",
    "sub_queries": ["the focused search query"],
    "reasoning": "one-sentence explanation"
}}

For compound queries, list each sub-question as a separate entry in sub_queries.
For all other types, sub_queries must contain exactly one string (the core query).

User query: "{query}"
"""

    def __init__(self, llm: LLMProvider, enable_decomposition: bool = True):
        self._llm = llm
        self._enable_decomposition = enable_decomposition

    def analyze(self, query: str) -> IntentResult:
        try:
            raw = self._llm.generate(
                self._PROMPT.format(query=query),
                max_tokens=256,
                temperature=0.0,
            )
            parsed = self._parse_json(raw)
            intent_type = parsed.get("intent_type", "factual")
            sub_queries = parsed.get("sub_queries", [query])
            reasoning = parsed.get("reasoning", "")

            # Guard: always have at least the original query
            if not sub_queries:
                sub_queries = [query]

            # Respect the decomposition flag
            if not self._enable_decomposition and len(sub_queries) > 1:
                sub_queries = [query]

            result = IntentResult(
                intent_type=intent_type,
                sub_queries=sub_queries,
                reasoning=reasoning,
            )
            print(
                f"   🧠 IntentAgent → type={intent_type!r}  "
                f"sub_queries={sub_queries}  reasoning={reasoning!r}"
            )
            return result

        except Exception as exc:
            logger.warning("IntentAgent failed (%s) — falling back to raw query", exc)
            return IntentResult(
                intent_type="factual",
                sub_queries=[query],
                reasoning="(fallback — LLM call failed)",
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())


class VerifierAgent:
    """LLM judge: is the retrieved context sufficient to answer the query?

    Returns ``is_answerable=True`` (passing the answer through) whenever the
    LLM call fails, so a broken verifier never silences otherwise-valid results.
    """

    _PROMPT = """\
You are a strict answer verifier for a RAG (Retrieval-Augmented Generation) system.

Given a user query and retrieved context chunks, determine:
1. Whether the context is sufficient to answer the query.
2. Your confidence (0.0 = not at all, 1.0 = definitely answerable).
3. If the context is insufficient, what specific information is missing and \
suggest a refined search query that is more likely to find the missing information.

Return ONLY valid JSON, no markdown fences:
{{
    "is_answerable": true,
    "confidence": 0.85,
    "feedback": "The context explains X and Y which directly answers the question.",
    "refined_query": null
}}

Set refined_query to null when is_answerable is true.

User query: "{query}"

Retrieved context ({n_chunks} chunks):
{context}
"""

    _PREVIEW_CHARS = 400  # chars per chunk shown to the verifier

    def __init__(
        self,
        llm: LLMProvider,
        confidence_threshold: float = 0.6,
    ):
        self._llm = llm
        self._threshold = confidence_threshold

    def verify(self, query: str, chunks: List[RetrievedChunk]) -> VerificationResult:
        if not chunks:
            return VerificationResult(
                is_answerable=False,
                confidence=0.0,
                feedback="No chunks were retrieved.",
                refined_query=query,
            )

        context = self._format_context(chunks)
        try:
            raw = self._llm.generate(
                self._PROMPT.format(
                    query=query,
                    n_chunks=len(chunks),
                    context=context,
                ),
                max_tokens=256,
                temperature=0.0,
            )
            parsed = IntentAgent._parse_json(raw)

            is_answerable = bool(parsed.get("is_answerable", True))
            confidence = float(parsed.get("confidence", 1.0))
            feedback = parsed.get("feedback", "")
            refined_query = parsed.get("refined_query") or None

            # Apply threshold: even if LLM says "answerable", low confidence
            # triggers a retry.
            if confidence < self._threshold:
                is_answerable = False

            result = VerificationResult(
                is_answerable=is_answerable,
                confidence=confidence,
                feedback=feedback,
                refined_query=refined_query,
            )
            print(
                f"   🔍 VerifierAgent → answerable={is_answerable}  "
                f"confidence={confidence:.2f}  refined_query={refined_query!r}"
            )
            return result

        except Exception as exc:
            logger.warning("VerifierAgent failed (%s) — assuming answerable", exc)
            return VerificationResult(
                is_answerable=True,
                confidence=1.0,
                feedback="(verification skipped — LLM call failed)",
            )

    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            preview = chunk.text[: self._PREVIEW_CHARS].replace("\n", " ")
            parts.append(
                f"[Chunk {i} | id={chunk.chunk_id} | score={chunk.score:.3f}]\n{preview}"
            )
        return "\n\n".join(parts)


class SynthesisAgent:
    """Generates a cited final answer from the verified chunks.

    Falls back to a plain concatenation of top-chunk text if the LLM
    call fails.
    """

    _PROMPT = """\
You are a precise and helpful assistant. Answer the user's question using ONLY \
the context chunks provided below.

Rules:
- Cite the chunk ID(s) you draw information from, e.g. [chunk_id].
- If the answer cannot be found in the context, say \
"I cannot answer this based on the provided documents."
- Do NOT use outside knowledge.

Context Chunks:
{context}

User Question: {query}

Answer:"""

    def __init__(self, llm: LLMProvider, max_tokens: int = 1024):
        self._llm = llm
        self._max_tokens = max_tokens

    def synthesize(self, query: str, chunks: List[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[{c.chunk_id}]\n{c.text}" for c in chunks
        )
        try:
            answer = self._llm.generate(
                self._PROMPT.format(query=query, context=context),
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            answer = answer.strip()
            print(f"   ✍️  SynthesisAgent → generated answer ({len(answer)} chars, used {len(chunks)} chunks)")
            return answer
        except Exception as exc:
            logger.warning("SynthesisAgent LLM failed (%s) — returning top chunk text", exc)
            return chunks[0].text if chunks else "No answer could be generated."


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Top-level coordinator.

    Usage::

        result = orchestrator.ask("What architecture replaces RNNs in NLP?")
        print(result.answer)
        for chunk in result.chunks:
            print(chunk.chunk_id, chunk.score)

    The ``retrieval`` argument must expose a
    ``retrieve(query, top_k, filters) -> List[RetrievedChunk]`` method.
    Pass a bare ``RetrievalPipeline`` (not ``SelfQueryRetriever``) because
    the ``IntentAgent`` already handles query understanding.
    """

    def __init__(
        self,
        retrieval: Any,
        llm: LLMProvider,
        config: Optional[OrchestratorConfig] = None,
    ):
        cfg = config or OrchestratorConfig()
        self._retrieval = retrieval
        self._config = cfg
        self._intent_agent = IntentAgent(
            llm=llm,
            enable_decomposition=cfg.enable_decomposition,
        )
        self._verifier = VerifierAgent(
            llm=llm,
            confidence_threshold=cfg.verification_confidence_threshold,
        )
        self._synthesizer = SynthesisAgent(llm=llm)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ask(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """Deterministic sequence: Intent → Retrieve → Verify → (re-analyze intent if needed) → Synthesize."""
        top_k = top_k or self._config.top_k
        print(f"\n{'='*70}")
        print(f"🎯 Orchestrator.ask  query={query!r}")
        print(f"{'='*70}")

        # ── Step 1: intent analysis on original query ─────────────────────
        current_query = query
        intent = self._intent_agent.analyze(current_query)
        
        sub_query_results: Dict[str, List[RetrievedChunk]] = {}
        verification: VerificationResult
        attempts = 0

        # ── Step 2-3: retrieve → verify → retry loop ─────────────────────
        for retry_num in range(self._config.max_retries + 1):
            attempts += 1
            
            # Always retrieve with current intent's sub-queries
            sub_query_results = self._parallel_retrieve(
                intent.sub_queries, top_k=top_k, filters=filters
            )
            merged = self._merge_chunks(sub_query_results, max_chunks=top_k * 3)
            print(
                f"   📦 Merged {sum(len(v) for v in sub_query_results.values())} "
                f"raw chunks → {len(merged)} unique chunks  (attempt {attempts})"
            )

            # Always verify
            if self._config.enable_verification:
                verification = self._verifier.verify(current_query, merged)
                
                if verification.is_answerable:
                    current_chunks = merged
                    break
                
                if retry_num == self._config.max_retries:
                    logger.warning(
                        "Verifier not satisfied after %d attempt(s) — proceeding anyway",
                        attempts,
                    )
                    current_chunks = merged
                    break
                
                # Retry: re-analyze on refined query
                refined = verification.refined_query or current_query
                print(
                    f"   ♻️  Retry {retry_num + 1}/{self._config.max_retries} "
                    f"with refined_query={refined!r}"
                )
                current_query = refined
                intent = self._intent_agent.analyze(current_query)  # ← RE-ANALYZE
            else:
                # Verification disabled
                current_chunks = merged
                verification = VerificationResult(
                    is_answerable=True,
                    confidence=1.0,
                    feedback="(verification disabled)",
                )
                break

        # ── Step 4: synthesize answer ────────────────────────────────────
        final_chunks = current_chunks[:top_k]
        answer = self._synthesizer.synthesize(current_query, final_chunks)

        print(f"\n{'='*70}")
        print(f"✅ Orchestrator done — {attempts} attempt(s), {len(final_chunks)} chunks")
        print(f"{'='*70}\n")

        return OrchestratorResult(
            answer=answer,
            chunks=final_chunks,
            intent=intent,
            verification=verification,
            attempts=attempts,
            sub_query_results=sub_query_results,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parallel_retrieve(
        self,
        sub_queries: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, List[RetrievedChunk]]:
        """Retrieve chunks for all sub-queries in parallel.

        Falls back to sequential if only one sub-query (avoids thread overhead).
        """
        results: Dict[str, List[RetrievedChunk]] = {}

        if len(sub_queries) == 1:
            q = sub_queries[0]
            results[q] = self._retrieval.retrieve(query=q, top_k=top_k, filters=filters)
            return results

        with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as pool:
            future_to_query = {
                pool.submit(
                    self._retrieval.retrieve, q, top_k, filters
                ): q
                for q in sub_queries
            }
            for future in as_completed(
                future_to_query,
                timeout=self._config.parallel_retrieval_timeout,
            ):
                q = future_to_query[future]
                try:
                    results[q] = future.result()
                    print(f"   🔎 Sub-query {q!r} → {len(results[q])} chunks")
                except FuturesTimeout:
                    logger.warning("Retrieval timed out for sub-query: %s", q)
                    results[q] = []
                except Exception as exc:
                    logger.warning("Retrieval failed for sub-query %r: %s", q, exc)
                    results[q] = []

        return results

    @staticmethod
    def _merge_chunks(
        sub_query_results: Dict[str, List[RetrievedChunk]],
        max_chunks: int = 20,
    ) -> List[RetrievedChunk]:
        """Deduplicate by chunk_id, keep best score, sort by score descending."""
        best: Dict[str, RetrievedChunk] = {}
        for chunks in sub_query_results.values():
            for chunk in chunks:
                existing = best.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    best[chunk.chunk_id] = chunk

        ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return ranked[:max_chunks]