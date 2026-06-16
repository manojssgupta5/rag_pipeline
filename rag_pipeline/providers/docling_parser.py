"""
Document parser built on Docling.

Docling (by IBM) handles PDF, DOCX, PPTX, XLSX, HTML, images (with
OCR), and more. It produces a structured `DoclingDocument` that we
flatten into plain text + per-element metadata before handing off to
the rest of the RAG pipeline.

Why we don't pass the structured DoclingDocument downstream as-is:
- Our Chunker (RecursiveTextChunker) operates on plain text with
  separator-based splitting. Trying to re-chunk Docling's hierarchical
  output would require duplicating Docling's tokenizer logic.
- A flat string keeps the pipeline backend-agnostic — you can swap
  Docling for Unstructured.io, PyMuPDF, etc., with no changes
  elsewhere.

Trade-off: we lose Docling's reading-order and layout signals during
chunking. If you need to preserve them, swap RecursiveTextChunker for
a Docling-aware chunker that walks `doc.body.children` directly.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ParsedDocument:
    """Output of the parser. The RAG pipeline consumes `text` and
    optionally enriches chunks with `elements` as metadata."""

    text: str
    elements: List[Dict[str, Any]]
    source_format: str
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


# --------------------------------------------------------------------------- #
# Abstract interface
# --------------------------------------------------------------------------- #
class DocumentParser(ABC):
    """Anything that can turn a file (or a string) into a ParsedDocument.

    Implemented for Docling below; you can write alternative parsers
    against this same interface and swap them in via config.
    """

    @abstractmethod
    def parse(self, source: Union[str, Path, bytes]) -> ParsedDocument:
        ...


# --------------------------------------------------------------------------- #
# Docling implementation
# --------------------------------------------------------------------------- #
class DoclingDocumentParser(DocumentParser):
    """Wraps Docling's `DocumentConverter` for multi-format parsing.

    Install:
        pip install docling

    First run downloads a few model weights (layout, table-structure).
    For CPU-only or smaller models, see `pipeline_options` below.
    """

    def __init__(
        self,
        allowed_formats: Optional[List[str]] = None,
        enable_ocr: bool = True,
        enable_table_structure: bool = True,
        artifacts_path: Optional[str] = None,
        save_markdown_to: Optional[str] = None,
        save_markdown_encoding: str = "utf-8",
    ):
        """
        Args:
            allowed_formats: Docling format names to accept. None = accept all.
            enable_ocr: Run OCR on scanned PDFs / images.
            enable_table_structure: Extract table cells as structured markdown.
            artifacts_path: If you've pre-downloaded Docling models, point this
                at the directory. Speeds up cold starts.
            save_markdown_to: If set, the converted markdown is also written
                here. Directory path -> <doc_id>.md inside it.
            save_markdown_encoding: Encoding for the saved markdown file.
        """
        # 🆕 FIX: ALL imports at the top, so they're always defined regardless
        # of which branch is taken below.
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter

        # Import PdfPipelineOptions + PdfFormatOption (wrapper docling requires).
        # format_options dict must hold PdfFormatOption, NOT PdfPipelineOptions directly.
        # docling internally does format_options[fmt].backend — that attr is on FormatOption.
        PdfPipelineOptions = None
        PdfFormatOption = None
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions as _P
            PdfPipelineOptions = _P
        except ImportError:
            pass

        try:
            from docling.document_converter import PdfFormatOption as _F
            PdfFormatOption = _F
        except ImportError:
            pass

        # Build PdfPipelineOptions
        pipeline_options = None
        if PdfPipelineOptions is not None:
            try:
                pipeline_options = PdfPipelineOptions()
                if hasattr(pipeline_options, "do_ocr"):
                    pipeline_options.do_ocr = enable_ocr
                if hasattr(pipeline_options, "do_table_structure"):
                    pipeline_options.do_table_structure = enable_table_structure
                if artifacts_path and hasattr(pipeline_options, "artifacts_path"):
                    pipeline_options.artifacts_path = artifacts_path
            except Exception as e:
                logger.warning("Could not configure PdfPipelineOptions: %s", e)
                pipeline_options = None

        # Wrap in PdfFormatOption — this is what format_options dict expects.
        format_option = None
        if PdfFormatOption is not None and pipeline_options is not None:
            try:
                format_option = PdfFormatOption(pipeline_options=pipeline_options)
            except Exception as e:
                logger.warning("Could not build PdfFormatOption: %s", e)
                format_option = None

        # Build allowed-formats filter if provided
        format_enums = []
        if allowed_formats is not None:
            for fmt_name in allowed_formats:
                fmt = getattr(InputFormat, fmt_name.upper(), None)
                if fmt is None:
                    logger.warning("Unknown Docling format %r; skipping", fmt_name)
                    continue
                format_enums.append(fmt)

        # Construct the converter. Try richest API first, fall back gracefully.
        converter = None
        if format_enums and format_option is not None:
            try:
                converter = DocumentConverter(
                    allowed_formats=format_enums,
                    format_options={InputFormat.PDF: format_option},
                )
            except Exception as e:
                logger.warning("API (allowed_formats+PdfFormatOption) failed: %s", e)
                converter = None
        if converter is None and format_enums:
            try:
                converter = DocumentConverter(allowed_formats=format_enums)
            except Exception as e:
                logger.warning("API (allowed_formats only) failed: %s", e)
                converter = None
        if converter is None and format_option is not None:
            try:
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: format_option}
                )
            except Exception as e:
                logger.warning("API (format_options only) failed: %s", e)
                converter = None
        if converter is None:
            converter = DocumentConverter()

        self._converter = converter
        self._input_format_enum = InputFormat
        self._save_markdown_to = save_markdown_to
        self._save_markdown_encoding = save_markdown_encoding

    # 🆕 NEW: helper
    def _save_markdown(self, parsed: ParsedDocument, source_id: str) -> None:
        """Optionally persist the converted markdown to disk for inspection,
        debugging, or downstream non-RAG use (e.g. human review, search index).

        - save_markdown_to is a directory  -> writes <source_id>.md inside it
        - save_markdown_to ends in .md     -> writes to that exact path
        - save_markdown_to is None         -> no-op
        """
        if not self._save_markdown_to:
            return

        target = Path(self._save_markdown_to)
        if target.is_dir() or str(self._save_markdown_to).endswith(("/", "\\")):
            target.mkdir(parents=True, exist_ok=True)
            out_path = target / f"{source_id}.md"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            out_path = target

        out_path.write_text(parsed.text, encoding=self._save_markdown_encoding)
        logger.info("Saved markdown -> %s  (%d chars)", out_path, len(parsed.text))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    # 🆕 NEW: parse method that knows the source_id so the saved filename is meaningful
    def parse(
        self,
        source: Union[str, Path, bytes],
        source_id: Optional[str] = None,
    ) -> ParsedDocument:
        print(f"\n📄 [DoclingDocumentParser.parse] called")
        print(f"   source type: {type(source).__name__}")
        if isinstance(source, (str, Path)):
            print(f"   source: {str(source)[:80]!r}")

        if isinstance(source, (str, Path)):
            try:
                print(f"   → _parse_path()...")
                parsed = self._parse_path(Path(source))
                print(f"   ✅ _parse_path OK: {len(parsed.text):,} chars")
            except Exception as e:
                print(f"   ❌ _parse_path FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise
            sid = source_id or Path(source).stem
        elif isinstance(source, bytes):
            try:
                print(f"   → _parse_bytes()...")
                parsed = self._parse_bytes(source)
                print(f"   ✅ _parse_bytes OK: {len(parsed.text):,} chars")
            except Exception as e:
                print(f"   ❌ _parse_bytes FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise
            sid = source_id or "document"
        else:
            raise TypeError(f"Unsupported source type: {type(source).__name__}")

        self._save_markdown(parsed, sid)
        print(f"   ✅ parse() returning ParsedDocument ({len(parsed.text):,} chars)\n")
        return parsed


    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _parse_path(self, path: Path) -> ParsedDocument:
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        logger.info("Parsing document: %s", path)
        result = self._converter.convert(str(path))
        return self._convert_to_parsed(result, source_format=path.suffix.lstrip(".").lower())

    def _parse_bytes(self, data: bytes, suffix: str = ".pdf") -> ParsedDocument:
        """Docling needs a real file on disk. Write to a temp file, parse,
        then clean up."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            logger.info("Parsing in-memory document (%d bytes)", len(data))
            result = self._converter.convert(str(tmp_path))
            return self._convert_to_parsed(result, source_format=suffix.lstrip(".").lower())
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _convert_to_parsed(self, result, source_format: str) -> ParsedDocument:
        """Flatten a Docling ConversionResult into our ParsedDocument.

        We use the `export_to_markdown` representation rather than plain
        text because it preserves:
            - heading hierarchy as # / ## / ###
            - table structure as markdown tables
            - list indentation
            - code block fences
        These structural markers give RecursiveTextChunker better
        natural split points than raw text would.
        """
        doc = result.document
        try:
            text = doc.export_to_markdown()
        except AttributeError:
            # Older Docling versions
            text = doc.export_to_text()

        # Build per-element metadata so downstream code can optionally
        # use page numbers / section headings for richer filtering.
        elements: List[Dict[str, Any]] = []
        page_count: Optional[int] = None
        try:
            # Docling v2 API
            for element in doc.body.children if hasattr(doc, "body") else []:
                elements.append(self._element_to_dict(element))
            page_count = len(doc.pages) if hasattr(doc, "pages") else None
        except Exception:
            logger.exception("Failed to extract Docling element metadata; continuing without it")

        return ParsedDocument(
            text=text,
            elements=elements,
            source_format=source_format,
            page_count=page_count,
            metadata={
                "source_format": source_format,
                "page_count": page_count,
            },
        )

    @staticmethod
    def _element_to_dict(element) -> Dict[str, Any]:
        """Pull a minimal dict of identifying info from a Docling element."""
        info: Dict[str, Any] = {
            "type": element.__class__.__name__,
        }
        # Most elements have a `text` attribute and a `prov` (provenance)
        # list pointing back to the source page/bbox.
        text = getattr(element, "text", None) or getattr(element, "export_to_text", lambda: "")()
        if callable(text):
            try:
                text = text()
            except Exception:
                text = ""
        if text:
            info["text_preview"] = text[:120]
        prov = getattr(element, "prov", None)
        if prov:
            try:
                first = prov[0]
                info["page"] = getattr(first, "page_no", None)
            except Exception:
                pass
        return info


# --------------------------------------------------------------------------- #
# Fallback / dev parser (no Docling required)
# --------------------------------------------------------------------------- #
class PlainTextDocumentParser(DocumentParser):
    """Used when the input is already plain text (or for unit tests where
    you don't want the Docling dependency)."""

    def parse(self, source: Union[str, Path, bytes]) -> ParsedDocument:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {path}")
            text = path.read_text(encoding="utf-8")
            fmt = path.suffix.lstrip(".").lower() or "txt"
        elif isinstance(source, bytes):
            text = source.decode("utf-8")
            fmt = "txt"
        else:
            raise TypeError(f"Unsupported source type: {type(source).__name__}")

        return ParsedDocument(
            text=text,
            elements=[],
            source_format=fmt,
            page_count=None,
            metadata={"source_format": fmt},
        )