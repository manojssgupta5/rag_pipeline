"""
run_transformer_paper_test.py

Production style ingestion test for the full pipeline.

This version:

* Uses your actual build_pipelines()
* Uses .env configuration
* Uses Docling PDF ingestion
* Uses Ollama embeddings
* Uses Qdrant
* Uses real retrieval flow
* Exercises metadata filters
* Exercises chunking
* Exercises hypothetical questions

Run:

```
python run_transformer_paper_test.py ./sample.pdf
```

"""

from __future__ import annotations

import sys
from pathlib import Path

from rag_pipeline import build_pipelines

TEST_QUERIES = [
#{"query": "what architecture replaces recurrent networks", "filters": {"section": "introduction"}},
{"query": "what is the transformer model based on", "filters": {"section": "abstract"}},
# {"query": "what bleu score did transformer achieve on english to german", "filters": {"section": "results"}},
# {"query": "how many gpus were used for training", "filters": {"section": "training"}},
# {"query": "why recurrent models are hard to parallelize", "filters": {"section": "introduction"}},
# {"query": "what is scaled dot product attention", "filters": {"section": "attention"}},
# {"query": "how attention output is computed", "filters": {"section": "attention"}},
# {"query": "what is multi head attention", "filters": {"section": "multi_head_attention"}},
# {"query": "how many attention heads are used", "filters": {"section": "multi_head_attention"}},
# {"query": "why decoder attention uses masking", "filters": {"section": "attention_applications"}},
# {"query": "what is dmodel value in transformer", "filters": {"section": "encoder_decoder"}},
# {"query": "what is the inner layer dimension dff", "filters": {"section": "feed_forward"}},
# {"query": "why positional encoding is needed", "filters": {"section": "positional_encoding"}},
# {"query": "what functions are used for positional encoding", "filters": {"section": "positional_encoding"}},
# {"query": "what optimizer was used for transformer training", "filters": {"section": "optimizer"}},
# {"query": "what are beta1 and beta2 values in adam optimizer", "filters": {"section": "optimizer"}},
# {"query": "what warmup steps were used during training", "filters": {"section": "optimizer"}},
# {"query": "what dropout rate was used in base model", "filters": {"section": "regularization"}},
# {"query": "what is label smoothing value", "filters": {"section": "regularization"}},
# {"query": "what bleu score did transformer big achieve", "filters": {"section": "results"}},
]

def main() -> None:
    print("=" * 80)
    print("Building pipelines from .env")
    print("=" * 80)

    ingestion, retrieval = build_pipelines()

    print("\n✓ Pipelines initialized")

    if len(sys.argv) >= 2:
        pdf_path = Path(sys.argv[1])
        
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            return

        print("\n" + "=" * 80)
        print(f"Ingesting document: {pdf_path}")
        print("=" * 80)

        doc_id = pdf_path.stem

        n_chunks = ingestion.ingest_file(
            doc_id=doc_id,
            source=pdf_path,
            document_metadata={},
        )

        print(f"\n✓ Ingested {n_chunks} chunks")
    else:
        print("\n" + "=" * 80)
        print("No PDF provided. Skipping ingestion phase and proceeding directly to retrieval.")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("Running retrieval tests")
    print("=" * 80)

    success = 0

    for i, case in enumerate(TEST_QUERIES, start=1):
        query = case["query"]
        filters = case["filters"]

        print(f"\n[{i}/{len(TEST_QUERIES)}]")
        print(f"Query   : {query}")
        print(f"Filters : {filters}")

        if hasattr(retrieval, "ask"):
            response = retrieval.ask(query=query, top_k=3)
            results = response.get("chunks", [])
            answer = response.get("answer", "")
            
            print(f"\n  🤖 Synthesized Answer:\n    {answer}")
            print("\n  📚 Top Source Context:")
        else:
            results = retrieval.retrieve(query=query, top_k=3)

        if not results:
            print("  ❌ No results")
            continue

        success += 1

        for rank, r in enumerate(results, start=1):
            preview = r.text.replace("\n", " ")[:180]
            print(
                f"\n  Rank #{rank}"
                f"\n    score       : {r.score:.4f}"
                f"\n    matched_via : {r.matched_via}"
                f"\n    chunk_id    : {r.chunk_id}"
                f"\n    text        : {preview}..."
        )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(
        f"\nSuccessful queries: "
        f"{success}/{len(TEST_QUERIES)}"
    )

if __name__ == "__main__":
    main()
