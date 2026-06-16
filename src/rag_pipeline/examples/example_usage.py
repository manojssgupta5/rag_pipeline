"""
example_usage.py

Three configs are shown:

1. LOCAL_DEV_CONFIG — runs entirely on your machine, no external services.
2. PRODUCTION_CONFIG — Ollama + Qdrant with Docling for multi-format parsing.
3. PDF_INGEST_DEMO — minimal example showing `ingest_file()` with a PDF.

Swap between them by changing one dict — ingestion.py, retrieval.py, and
your application code never change.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from rag_pipeline import build_pipelines
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)

SAMPLE_DOC = """
Free delivery is available on all orders above AED 100 within the Dubai
Marina zone. Orders below this threshold incur a flat delivery fee of
AED 10. Delivery times during peak hours (12pm-2pm and 7pm-9pm) may be
extended by up to 20 minutes due to high demand.

Refunds for cancelled orders are processed within 3-5 business days to
the original payment method. If an order arrives damaged or incorrect,
customers can request a full refund or replacement through the app
within 24 hours of delivery.
"""


def demo_text_ingest(config: dict) -> None:
    """Ingest a plain-text document (no parser needed)."""
    ingestion, retrieval = build_pipelines(config)

    ingestion.ingest_document(
        doc_id="delivery-policy-v3",
        text=SAMPLE_DOC,
        document_metadata={
            "source": "help_center",
            "region": "dubai_marina",
            "section": "delivery_and_refunds",
        },
    )

    for query in ["is there free delivery in dubai marina", "how long do refunds take"]:
        print(f"\nQuery: {query}")
        results = retrieval.retrieve(query, top_k=3, filters={"region": "dubai_marina"})
        for r in results:
            print(f"  [{r.score:.4f}] ({r.matched_via}) {r.text[:90]}...")


def demo_pdf_ingest(pdf_path: str) -> None:
    """
    🆕 Ingest a PDF (or DOCX/PPTX/HTML) via Docling.

    Usage:
        demo_pdf_ingest("./sample.pdf")
    """
    ingestion, retrieval = build_pipelines()

    # ✅ FIX: use the parameter, validate it, and derive doc_id from it
    path = Path(pdf_path)
    if not path.exists():
        print(f"File not found: {pdf_path} — skipping PDF demo.")
        return
    if not path.is_file():
        print(f"Not a file: {pdf_path} — skipping PDF demo.")
        return

    # Derive doc_id from the file stem (e.g. "sample.pdf" -> "sample")
    # This way the saved markdown filename and the doc_id stay in sync.
    doc_id = path.stem

    print(f"Ingesting PDF: {path.name}  (doc_id={doc_id!r})")

    # ✅ FIX: pass the actual path variable, not a hardcoded string
    n_chunks = ingestion.ingest_file(
        doc_id=doc_id,
        source=path,                    # ← was hardcoded "./sample.pdf"
        document_metadata={
            "source": "test_upload",
            "region": "dubai_marina",
            "section": "operations_kb",
        },
    )
    print(f"✓ Ingested {n_chunks} chunks from {path.name}")

    # If markdown auto-save is enabled, mention where it went
    save_dir = os.environ.get("SAVE_MARKDOWN_TO")
    if save_dir:
        saved_md = Path(save_dir) / f"{doc_id}.md"
        if saved_md.exists():
            print(f"✓ Markdown saved to: {saved_md}")
        else:
            print(f"Expected markdown not found: {saved_md}")

    # And retrieval is unchanged
    print("\nRetrieval test queries:")
    for query in ["what is transformer", "why recurrent models", "hard to parallelize"]:
        print(f"\n  Query: {query!r}")
        results = retrieval.retrieve(query, top_k=3)
        if not results:
            print("    (no results)")
        for r in results:
            preview = r.text.replace("\n", " ")[:120]
            print(f"    [{r.score:.4f}] {preview}...")


if __name__ == "__main__":
    import sys

    # Use whichever config matches your environment
    # demo_text_ingest(LOCAL_DEV_CONFIG)

    # PDF demo — accepts a path as a CLI arg, falls back to ./sample.pdf
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else "./sample.pdf"
    demo_pdf_ingest(pdf_arg)
