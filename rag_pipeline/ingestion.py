"""
Ingestion pipeline.

Order of operations (deliberate -- see module docstrings in base.py for why):

    raw document
        -> [parse with Docling, if a file path is given]   <-- NEW
        -> chunk
        -> attach metadata
        -> generate hypothetical questions
        -> embed (dense + sparse for chunks, dense for hyp questions)
        -> upsert into vector store

Everything before "embed" mutates the Chunk object. Embedding is the
final, irreversible transformation -- nothing after it can change what
was captured.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import Chunk, DenseEmbedder, LLMProvider, SparseEmbedder, VectorStore
from .providers.docling_parser import DocumentParser, ParsedDocument, PlainTextDocumentParser

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
class RecursiveTextChunker:
    """Splits text on a hierarchy of separators, falling back to smaller
    separators only when a piece still exceeds chunk_size. This keeps
    paragraph/sentence boundaries intact wherever possible.

    Chunk IDs are content-hash based: re-ingesting unchanged text produces
    the same ID, so vector store upserts are idempotent.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str, doc_id: str) -> List[Chunk]:
        pieces = self._split_recursive(text, self._separators)
        chunks: List[Chunk] = []
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = self._make_id(doc_id, idx, piece)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={"doc_id": doc_id, "chunk_index": idx},
                )
            )
        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Last resort: hard character split.
            return [
                text[i : i + self._chunk_size]
                for i in range(0, len(text), self._chunk_size - self._chunk_overlap)
            ]

        parts = text.split(separator)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self._chunk_size:
                    if remaining_separators:
                        chunks.extend(self._split_recursive(part, remaining_separators))
                        current = ""
                    else:
                        chunks.append(part)
                        current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks, separator)

    def _apply_overlap(self, chunks: List[str], separator: str) -> List[str]:
        if self._chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self._chunk_overlap :]
            overlapped.append((tail + separator + chunks[i])[: self._chunk_size + self._chunk_overlap])
        return overlapped

    @staticmethod
    def _make_id(doc_id: str, idx: int, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"{doc_id}::chunk_{idx:04d}::{digest}"


# --------------------------------------------------------------------------- #
# Markdown-header-aware chunker
# --------------------------------------------------------------------------- #
class MarkdownHeaderChunker:
    """Split text by markdown headers first, then sub-split oversized sections.

    Strategy
    --------
    1. Walk the text line-by-line. Every line that starts with one or more
       ``#`` characters begins a new section.  Text before the first header
       is treated as a preamble section with an empty header path.
    2. If a section's text is within ``chunk_size`` characters it becomes
       exactly one ``Chunk``.
    3. If a section exceeds ``chunk_size`` it is further split using the
       same recursive character-splitting logic as ``RecursiveTextChunker``
       (paragraph → sentence → word → hard-cut), with ``chunk_overlap``
       applied between the sub-chunks.
    4. Each chunk gets two extra metadata keys:
       - ``section_header``: the nearest header line (e.g. ``## Introduction``).
       - ``section_path``:   the full ancestor path joined by ``" > "``
         (e.g. ``"# Attention Is All You Need > ## Model Architecture"``).

    Chunk IDs are content-hash based (same contract as ``RecursiveTextChunker``).
    """

    # Separator hierarchy used for sub-splitting oversized sections
    _SUBSPLIT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
    _HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ #
    # Public API (same as RecursiveTextChunker)
    # ------------------------------------------------------------------ #
    def split(self, text: str, doc_id: str) -> List[Chunk]:
        sections = self._split_by_headers(text)
        chunks: List[Chunk] = []
        global_idx = 0
        for section_text, section_header, section_path in sections:
            section_text = section_text.strip()
            if not section_text:
                continue
            pieces = self._sub_split(section_text)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                chunk_id = self._make_id(doc_id, global_idx, piece)
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=piece,
                        metadata={
                            "doc_id": doc_id,
                            "chunk_index": global_idx,
                            "section_header": section_header,
                            "section_path": section_path,
                        },
                    )
                )
                global_idx += 1
        return chunks

    # ------------------------------------------------------------------ #
    # Header splitting
    # ------------------------------------------------------------------ #
    def _split_by_headers(
        self, text: str
    ) -> List[tuple]:  # (section_text, header_line, path_str)
        """Return a list of (section_text, header_line, path_str) triples."""
        lines = text.splitlines(keepends=True)

        # Stack tracks (level, header_line) pairs for path building
        header_stack: List[tuple] = []  # [(level: int, header_line: str)]
        sections: List[tuple] = []

        current_lines: List[str] = []
        current_header: str = ""
        current_path: str = ""

        def _flush():
            body = "".join(current_lines)
            sections.append((body, current_header, current_path))

        for line in lines:
            m = self._HEADER_RE.match(line)
            if m:
                # Save previous section
                _flush()
                current_lines = []

                level = len(m.group(1))
                header_text = line.rstrip()

                # Pop headers of equal or deeper level
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, header_text))

                current_header = header_text
                current_path = " > ".join(h for _, h in header_stack)
            else:
                current_lines.append(line)

        _flush()  # last section
        return sections

    # ------------------------------------------------------------------ #
    # Sub-splitting (mirrors RecursiveTextChunker logic)
    # ------------------------------------------------------------------ #
    def _sub_split(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]
        return self._split_recursive(text, self._SUBSPLIT_SEPARATORS)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]

        separator = separators[0]
        remaining = separators[1:]

        if separator == "":
            return [
                text[i : i + self._chunk_size]
                for i in range(0, len(text), self._chunk_size - self._chunk_overlap)
            ]

        parts = text.split(separator)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self._chunk_size:
                    if remaining:
                        chunks.extend(self._split_recursive(part, remaining))
                        current = ""
                    else:
                        chunks.append(part)
                        current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks, separator)

    def _apply_overlap(self, chunks: List[str], separator: str) -> List[str]:
        if self._chunk_overlap == 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-self._chunk_overlap :]
            overlapped.append(
                (tail + separator + chunks[i])[: self._chunk_size + self._chunk_overlap]
            )
        return overlapped

    @staticmethod
    def _make_id(doc_id: str, idx: int, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"{doc_id}::chunk_{idx:04d}::{digest}"


# --------------------------------------------------------------------------- #
# Auto Metadata Extraction
# --------------------------------------------------------------------------- #
class LLMMetadataExtractor:
    """Uses an LLM to extract document-level metadata from the raw text."""

    _PROMPT = """You are a metadata extractor. Extract the following fields from the document text:
- 'title' (string): Title of the document
- 'author' (string): Author(s) of the document
- 'year' (integer): Publication year
- 'domain' (string): The general subject domain (e.g. nlp, finance, healthcare)
- 'source' (string): The source or venue (e.g. arxiv, conference, book)

Return ONLY valid JSON. Do not return markdown code blocks, just raw JSON. If a field cannot be determined, set its value to null.

Text excerpt:
\"\"\"
{text}
\"\"\"
"""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def extract(self, text: str) -> Dict[str, Any]:
        # Only use the first 8000 characters to save tokens and avoid context limits
        excerpt = text[:8000]
        prompt = self._PROMPT.format(text=excerpt)
        try:
            raw = self._llm.generate(prompt, max_tokens=256, temperature=0.0)
            # Clean up potential markdown formatting (```json ... ```)
            raw = raw.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
            metadata = json.loads(raw)
            # Filter out nulls
            return {k: v for k, v in metadata.items() if v is not None}
        except Exception:
            logger.exception("Failed to extract metadata using LLM")
            return {}


# --------------------------------------------------------------------------- #
# Metadata enrichment
# --------------------------------------------------------------------------- #
class MetadataEnricher:
    """Merges document-level metadata (source, section, dates, custom
    tags) into every chunk derived from that document.

    Per-chunk metadata (doc_id, chunk_index) set by the chunker is
    preserved; document-level keys are added without overwriting
    chunk-level keys of the same name.
    """

    def enrich(self, chunks: List[Chunk], document_metadata: Dict[str, Any]) -> List[Chunk]:
        for chunk in chunks:
            for key, value in document_metadata.items():
                chunk.metadata.setdefault(key, value)
            chunk.metadata.setdefault("char_count", len(chunk.text))
        return chunks


# --------------------------------------------------------------------------- #
# Hypothetical question generation
# --------------------------------------------------------------------------- #
@dataclass
class HypotheticalQuestionConfig:
    enabled: bool = True
    questions_per_chunk: int = 3
    max_tokens: int = 256


class HypotheticalQuestionGenerator:
    """Generates N questions per chunk that the chunk content would answer.

    On any failure (LLM error, empty/unparseable response) the chunk is
    left with zero hypothetical questions rather than failing the whole
    ingestion run -- a missing HyDE entry degrades retrieval slightly, it
    does not corrupt the pipeline.
    """

    _PROMPT = """You generate questions that the following passage directly answers.

Rules:
- Output exactly {n} questions, one per line, no numbering, no extra text.
- Questions should sound like real user queries (natural, conversational).
- Each question must be answerable using only this passage.

Passage:
\"\"\"
{passage}
\"\"\"
"""

    def __init__(self, llm: LLMProvider, config: Optional[HypotheticalQuestionConfig] = None):
        self._llm = llm
        self._config = config or HypotheticalQuestionConfig()

    def generate(self, chunks: List[Chunk]) -> List[Chunk]:
        if not self._config.enabled:
            return chunks

        for idx, chunk in enumerate(chunks):
            try:
                prompt = self._PROMPT.format(n=self._config.questions_per_chunk, passage=chunk.text)
                raw = self._llm.generate(prompt, max_tokens=self._config.max_tokens, temperature=0.3)
                questions = self._parse(raw)
                chunk.hypothetical_questions = questions[: self._config.questions_per_chunk]
                # if idx % 10 == 0:  # Print every 10 chunks or if debug enabled
                #     print(f"   → Generated {len(questions)} questions for chunk {idx+1}/{len(chunks)}: {questions}")
            except Exception:
                logger.exception("Hypothetical question generation failed for chunk %s", chunk.id)
                chunk.hypothetical_questions = []
        return chunks

    @staticmethod
    def _parse(raw: str) -> List[str]:
        lines = [line.strip() for line in raw.splitlines()]
        cleaned = []
        for line in lines:
            if not line:
                continue
            line = re.sub(r"^(\d+[.)]|[-*•]|Q\d*:)\s*", "", line).strip()
            if line:
                cleaned.append(line)
        return cleaned


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class IngestionPipeline:
    def __init__(
        self,
        dense_embedder: DenseEmbedder,
        vector_store: VectorStore,
        sparse_embedder: Optional[SparseEmbedder] = None,
        chunker: Optional[RecursiveTextChunker] = None,
        metadata_enricher: Optional[MetadataEnricher] = None,
        hypothetical_question_generator: Optional[HypotheticalQuestionGenerator] = None,
        document_parser: Optional[DocumentParser] = None,
        llm_metadata_extractor: Optional[LLMMetadataExtractor] = None,
        embed_batch_size: int = 64,
        debug: bool = False,
    ):
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._store = vector_store
        self._chunker = chunker or RecursiveTextChunker()
        self._enricher = metadata_enricher or MetadataEnricher()
        self._hyp_q = hypothetical_question_generator
        self._parser = document_parser or PlainTextDocumentParser()
        self._llm_extractor = llm_metadata_extractor
        self._embed_batch_size = embed_batch_size
        self._debug = debug

    # ------------------------------------------------------------------ #
    # Original API (unchanged) — takes pre-extracted text
    # ------------------------------------------------------------------ #
    def ingest_document(
        self, doc_id: str, text: str, document_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Run the full pipeline for one document whose text is already
        extracted. Returns the number of chunks ingested."""
        return self.ingest(
            doc_id=doc_id,
            source=text,
            document_metadata=document_metadata or {},
        )

    # ------------------------------------------------------------------ #
    # NEW API — takes a file path or bytes
    # ------------------------------------------------------------------ #
    def ingest_file(
        self,
        doc_id: str,
        source: Union[str, Path, bytes],
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Parse a file (PDF/DOCX/PPTX/HTML/...) via the configured
        document_parser, then run the rest of the pipeline.

        Args:
            doc_id: Stable identifier for this document.
            source: Filesystem path (str/Path) or raw file bytes.
            document_metadata: Extra metadata merged into every chunk
                (e.g. {"source": "sharepoint", "region": "uae"}).

        Returns:
            Number of chunks produced and stored.
        """
        return self.ingest(
            doc_id=doc_id,
            source=source,
            document_metadata=document_metadata or {},
        )

    # ------------------------------------------------------------------ #
    # Unified entry point
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        doc_id: str,
        source: Union[str, bytes, ParsedDocument],
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Unified ingestion entry point — INSTRUMENTED for debugging."""
        document_metadata = dict(document_metadata or {})

        print(f"\n🔵 [ingest] START  doc_id={doc_id!r}")
        print(f"   source type: {type(source).__name__}")
        if isinstance(source, (str, Path)):
            print(f"   source preview: {str(source)[:80]!r}")

        # Step 0 — parse if needed
        try:
            if isinstance(source, ParsedDocument):
                parsed = source
                print(f"   ✅ Source is ParsedDocument (skipping parse)")
            elif isinstance(source, bytes):
                print(f"   → Calling parser.parse(bytes)...")
                parsed = self._parser.parse(source)
            elif isinstance(source, str):
                print(f"   → Calling _resolve_str_source...")
                parsed = self._resolve_str_source(source)
            elif isinstance(source, Path):
                print(f"   → Calling parser.parse(Path)...")
                parsed = self._parser.parse(source)
            else:
                raise TypeError(f"Unsupported source type: {type(source).__name__}")

            print(f"   ✅ Parsed OK: {len(parsed.text):,} chars, "
                f"format={parsed.source_format}, pages={parsed.page_count}")
        except Exception as e:
            print(f"   ❌ PARSE FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Pull parser-discovered metadata into the document metadata dict
        for k, v in parsed.metadata.items():
            document_metadata.setdefault(k, v)
        document_metadata.setdefault("source_format", parsed.source_format)
        if parsed.page_count is not None:
            document_metadata.setdefault("page_count", parsed.page_count)

        # Step 0.5 — Auto extract metadata
        if self._llm_extractor:
            print("\n\n🔵 [ingest] Auto extract metadata")
            print(f"   → Extracting auto-metadata using LLM...")
            try:
                extracted = self._llm_extractor.extract(parsed.text)
                for k, v in extracted.items():
                    document_metadata.setdefault(k, v)
                print(f"   ✅ → → → Auto-metadata extracted: {extracted}")
            except Exception as e:
                print(f"   ❌ LLM METADATA EXTRACTION FAILED: {type(e).__name__}: {e}")

        # Step 1 — chunk
        print("\n\n🔵 [ingest] Chunking")
        print(f"   → Chunking text ({len(parsed.text):,} chars)...")
        try:
            chunks = self._chunker.split(parsed.text, doc_id=doc_id)
            print(f"   ✅ → → → Chunked into {len(chunks)} chunks")
        except Exception as e:
            print(f"   ❌ CHUNKING FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

        if not chunks:
            logger.warning("Document %s produced zero chunks", doc_id)
            print(f"   ⚠️ Zero chunks produced — returning")
            return 0

        # Step 2 — enrich with metadata
        print("\n\n🔵 [ingest] Enrich with metadata")
        print(f"   → Enriching {len(chunks)} chunks with metadata...")
        try:
            chunks = self._enricher.enrich(chunks, document_metadata)
            print(f"   ✅ → → → Enriched")
        except Exception as e:
            print(f"   ❌ ENRICH FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Step 3 — generate hypothetical questions
        if self._hyp_q is not None:
            print("\n\n🔵 [ingest] Generate hypothetical questions")
            print(f"   → Generating hypothetical questions for {len(chunks)} chunks...")
            try:
                chunks = self._hyp_q.generate(chunks)
                n_questions = sum(len(c.hypothetical_questions) for c in chunks)
                print(f"   ✅ → → → Generated {n_questions} hypothetical questions total")
            except Exception as e:
                print(f"   ❌ HYP Q GEN FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            print(f"   ⏭ Skipping hypothetical questions (no generator)")

        # Step 4 & 5 — embed and store
        print(f"   → Embedding and storing {len(chunks)} chunks...")
        for c in chunks[:3]:
            print(c.metadata)
        try:
            self._embed_and_store(chunks)
            print(f"   ✅ Embedded and stored")
        except Exception as e:
            print(f"   ❌ EMBED/STORE FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

        logger.info("Ingested document %s -> %d chunks", doc_id, len(chunks))
        print(f"🟢 [ingest] DONE  doc_id={doc_id!r}  chunks={len(chunks)}\n")
        return len(chunks)


    def _resolve_str_source(self, source: str) -> ParsedDocument:
        """Safely decide whether a string is a filesystem path or raw text.

        Rules:
            1. Strings longer than 4096 chars are always treated as raw text
            (no legitimate filesystem path is that long).
            2. For shorter strings, attempt a safe stat; if the file exists,
            parse it via the document_parser.
            3. On any OSError (file name too long, permission denied, etc.)
            fall back to raw text — never crash ingestion.
        """
        MAX_PATH_LEN = 4096

        if len(source) > MAX_PATH_LEN:
            # Definitely text
            return ParsedDocument(
                text=source,
                elements=[],
                source_format="txt",
                metadata={"source_format": "txt"},
            )

        # Short string — try the filesystem, but never let it raise
        try:
            p = Path(source)
            if p.is_file():
                return self._parser.parse(p)
        except OSError:
            # Errno 63 (name too long), permission errors, etc. — fall through
            # to raw-text handling.
            pass

        # Default: treat as raw text
        return ParsedDocument(
            text=source,
            elements=[],
            source_format="txt",
            metadata={"source_format": "txt"},
        )

    def _embed_and_store(self, chunks: List[Chunk]) -> None:
        """Embed chunks + their hypothetical questions, then upsert to the
        vector store. Prints detailed diagnostics at each stage when
        debug=True on the pipeline."""
        for i in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[i : i + self._embed_batch_size]
            texts = [c.text for c in batch]

            # ============================================================ #
            # STAGE A — chunks exist, no vectors yet
            # ============================================================ #
            if self._debug:
                self._print_stage_header("A. CHUNKS (pre-embedding)", batch)
                self._print_chunks_before_embedding(batch)

            # ============================================================ #
            # STAGE B — embed chunks (dense + sparse)
            # ============================================================ #
            dense_vectors = self._dense.embed(texts)
            sparse_vectors = self._sparse.embed(texts) if self._sparse else None

            if self._debug:
                self._print_embeddings_after(batch, dense_vectors, sparse_vectors)

            # ============================================================ #
            # STAGE C — embed hypothetical questions (dense only)
            # ============================================================ #
            hyp_records: List[Dict[str, Any]] = []
            all_questions: List[str] = []
            question_owner: List[str] = []
            for chunk in batch:
                for q in chunk.hypothetical_questions:
                    all_questions.append(q)
                    question_owner.append(chunk.id)

            if all_questions:
                q_vectors = self._dense.embed(all_questions)
                for parent_id, q_text, q_vec in zip(question_owner, all_questions, q_vectors):
                    hyp_records.append(
                        {"parent_id": parent_id, "text": q_text, "dense_vector": q_vec}
                    )

            if self._debug:
                self._print_hypothetical_questions(hyp_records, q_vectors if all_questions else None)

            # ============================================================ #
            # STAGE D — upsert to vector store
            # ============================================================ #
            self._store.upsert(
                chunks=batch,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                hyp_question_records=hyp_records or None,
            )

            if self._debug:
                self._print_post_upsert_summary(batch, dense_vectors, sparse_vectors, hyp_records)

    # ------------------------------------------------------------------ #
    # Debug printing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_stage_header(title: str, batch: List[Chunk]) -> None:
        print("\n" + "=" * 78)
        print(f"  {title}  (batch of {len(batch)} chunk{'s' if len(batch) != 1 else ''})")
        print("=" * 78)

    @staticmethod
    def _print_chunks_before_embedding(batch: List[Chunk]) -> None:
        for idx, chunk in enumerate(batch):
            text_preview = chunk.text[:160].replace("\n", " ↵ ")
            more = "..." if len(chunk.text) > 160 else ""
            print(f"\n--- Chunk {idx + 1}/{len(batch)} ---")
            print(f"  id              : {chunk.id}")
            print(f"  metadata        : {chunk.metadata}")
            print(f"  char_count      : {len(chunk.text)}")
            print(f"  text            : {text_preview}{more}")
            if chunk.hypothetical_questions:
                print(f"  hyp_questions   : {len(chunk.hypothetical_questions)}")
                for j, q in enumerate(chunk.hypothetical_questions, 1):
                    print(f"    {j}. {q}")
            else:
                print(f"  hyp_questions   : (none)")

    @staticmethod
    def _print_embeddings_after(
        batch: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]],
    ) -> None:
        print("\n" + "-" * 78)
        print("  EMBEDDINGS CREATED  (what will be sent to the vector store)")
        print("-" * 78)
        for idx, chunk in enumerate(batch):
            d_vec = dense_vectors[idx]
            s_vec = sparse_vectors[idx] if sparse_vectors else None

            d_norm = (sum(v * v for v in d_vec)) ** 0.5
            print(f"\n--- Chunk {idx + 1}/{len(batch)} [{chunk.id}] ---")
            print(f"  dense vector    : dim={len(d_vec)}, L2-norm={d_norm:.4f}")
            print(f"    first 8 dims  : {[round(v, 4) for v in d_vec[:8]]}")
            print(f"    last 4 dims   : {[round(v, 4) for v in d_vec[-4:]]}")

            if s_vec is not None:
                # Show top-K terms by weight
                top_terms = sorted(s_vec.items(), key=lambda kv: kv[1], reverse=True)[:8]
                print(f"  sparse vector   : {len(s_vec)} non-zero terms")
                print(f"    top terms     : {top_terms}")
            else:
                print(f"  sparse vector   : (none — sparse embedder disabled)")

    @staticmethod
    def _print_hypothetical_questions(
        hyp_records: List[Dict[str, Any]],
        q_vectors: Optional[List[List[float]]],
    ) -> None:
        print("\n" + "-" * 78)
        print(f"  HYPOTHETICAL QUESTIONS EMBEDDED  ({len(hyp_records)} records)")
        print("-" * 78)
        if not hyp_records:
            print("  (no hypothetical questions for this batch)")
            return
        for i, rec in enumerate(hyp_records, 1):
            print(f"\n  --- HypQ {i}/{len(hyp_records)} ---")
            print(f"    parent_id : {rec['parent_id']}")
            print(f"    text      : {rec['text']}")
            if q_vectors and i - 1 < len(q_vectors):
                v = q_vectors[i - 1]
                print(f"    vector    : dim={len(v)}, "
                      f"first 4=[{', '.join(f'{x:.3f}' for x in v[:4])}]")

    @staticmethod
    def _print_post_upsert_summary(
        batch: List[Chunk],
        dense_vectors: List[List[float]],
        sparse_vectors: Optional[List[Dict[int, float]]],
        hyp_records: List[Dict[str, Any]],
    ) -> None:
        print("\n" + "-" * 78)
        print("  UPSERT COMPLETE  (what was just written to the vector store)")
        print("-" * 78)
        chunk_count = len(batch)
        hyp_count = len(hyp_records)
        total = chunk_count + hyp_count
        print(f"  chunks upserted          : {chunk_count}")
        print(f"  hyp_questions upserted   : {hyp_count}")
        print(f"  total points written     : {total}")
        print(f"  dense dim                : {len(dense_vectors[0]) if dense_vectors else 0}")
        if sparse_vectors:
            avg_terms = sum(len(s) for s in sparse_vectors) / max(1, len(sparse_vectors))
            print(f"  sparse avg terms/chunk   : {avg_terms:.1f}")
        print("=" * 78 + "\n")