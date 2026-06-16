"""
Sparse embedder implementation.

Rather than maintaining a fixed vocabulary (which breaks the moment new
terms appear at query time -- product codes, new jargon, etc.), this
implementation hashes tokens into a large fixed-size space and emits
term-frequency weights. Paired with a vector store that applies an IDF
modifier at query time (e.g. Qdrant's `Modifier.IDF` on a sparse vector
field), this reproduces BM25-style scoring without precomputing corpus
statistics during ingestion.

If you need SPLADE-quality sparse vectors (learned term expansion),
implement a SpladeSparseEmbedder against the same SparseEmbedder
interface using a transformers model -- nothing else in the pipeline
needs to change.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from ..base import SparseEmbedder

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

# Common English stopwords. Kept intentionally small -- stopwords still
# carry weight in exact-match scenarios (e.g. "to be or not to be") and
# over-aggressive removal hurts recall on short queries.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "it",
        "for", "on", "with", "as", "by", "at", "from", "this", "that",
    }
)


class HashingTFSparseEmbedder(SparseEmbedder):
    """Tokenize -> lowercase -> hash into [0, vocab_size) -> term frequency.

    vocab_size should be large enough that hash collisions are rare for
    your corpus (2**18 is a reasonable default for most document sets).
    """

    def __init__(self, vocab_size: int = 2 ** 18, remove_stopwords: bool = True):
        self._vocab_size = vocab_size
        self._remove_stopwords = remove_stopwords

    def _tokenize(self, text: str) -> List[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        return tokens

    def _hash(self, token: str) -> int:
        # Stable across processes (unlike Python's built-in hash(), which
        # is salted per-process unless PYTHONHASHSEED is fixed).
        return (
            int.from_bytes(
                __import__("hashlib").md5(token.encode("utf-8")).digest()[:4],
                "big",
            )
            % self._vocab_size
        )

    def embed(self, texts: List[str]) -> List[Dict[int, float]]:
        results: List[Dict[int, float]] = []
        for text in texts:
            tokens = self._tokenize(text)
            counts = Counter(self._hash(t) for t in tokens)
            total = sum(counts.values()) or 1
            # Raw term frequency (normalized). IDF weighting is applied by
            # the vector store at query time via a sparse vector modifier.
            results.append({tok_id: count / total for tok_id, count in counts.items()})
        return results
