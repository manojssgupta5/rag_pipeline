# RAG Pipeline

Production-grade hybrid RAG system. Modular, provider-swappable, runs local or cloud.

---

## Architecture

```
Documents
   ↓
DocumentParser          (plain text | Docling → PDF/DOCX/PPTX/HTML/MD)
   ↓
RecursiveTextChunker
   ↓
MetadataEnricher  ──→  LLMMetadataExtractor (optional, LLM-assisted)
   ↓
HypotheticalQuestionGenerator (optional, HyDE)
   ↓
DenseEmbedder + SparseEmbedder
   ↓
VectorStore.upsert()           ← Qdrant
   
Query
   ↓
[SelfQueryRetriever]           (optional: LLM extracts filters from natural language)
   ↓
DenseEmbedder + SparseEmbedder
   ↓
VectorStore.hybrid_search()    (dense + sparse fusion)
   ↓
[CrossEncoderReranker]         (optional)
   ↓
[AnswerSynthesizer]            (optional: LLM generates final answer)
   ↓
RetrievedChunk[]  /  {answer, chunks}
```

All components implement ABCs in `base.py`. Swap any provider without touching pipeline code.

---

## Providers

| Component | Options |
|-----------|---------|
| Dense Embedder | `openai`, `ollama`, `sentence_transformers` |
| Sparse Embedder | `hashing_tf` (BM25-style TF + IDF at store), `none` |
| LLM | `openai`, `ollama`, `none` |
| Vector Store | `qdrant` |
| Document Parser | `plain`, `docling` |
| Reranker | `cross_encoder`, `none` |

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # edit as needed
```

**Qdrant** (Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Ollama** (local LLM + embeddings):
```bash
ollama pull qwen3-embedding:8b
ollama pull llama3:8b
```

---

## Environment Variables

### Embeddings
| Var | Default | Notes |
|-----|---------|-------|
| `EMBEDDING_PROVIDER` | `ollama` | `openai` \| `ollama` \| `sentence_transformers` |
| `EMBEDDING_MODEL` | `qwen3-embedding:8b` | Model name for chosen provider |
| `EMBEDDING_DIMENSION` | `768` | Must match model output |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OPENAI_API_KEY` | — | Required for openai provider |
| `OPENAI_BASE_URL` | — | Override for OpenAI-compatible endpoints |

### Sparse
| Var | Default | Notes |
|-----|---------|-------|
| `SPARSE_PROVIDER` | `hashing_tf` | `hashing_tf` \| `none` |
| `SPARSE_VOCAB_SIZE` | `262144` | 2^18; larger = fewer hash collisions |
| `REMOVE_STOPWORDS` | `true` | |

### LLM
| Var | Default | Notes |
|-----|---------|-------|
| `LLM_PROVIDER` | `none` | `ollama` \| `openai` \| `none` |
| `LLM_MODEL` | `llama3:8b` | |

### Vector Store
| Var | Default | Notes |
|-----|---------|-------|
| `VECTOR_STORE_PROVIDER` | `qdrant` | |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_COLLECTION` | `support_kb` | |
| `QDRANT_API_KEY` | — | Qdrant Cloud only |
| `QDRANT_RECREATE` | `false` | `true` wipes + recreates collection |

### Document Parser
| Var | Default | Notes |
|-----|---------|-------|
| `DOCUMENT_PARSER_PROVIDER` | `plain` | `plain` \| `docling` |
| `ALLOWED_FORMATS` | `pdf,docx,pptx,html,md` | Docling only |
| `ENABLE_OCR` | `true` | Docling only |
| `ENABLE_TABLE_STRUCTURE` | `true` | Docling only |
| `DOCLING_ARTIFACTS_PATH` | — | Optional model cache path |
| `SAVE_MARKDOWN_TO` | — | Dump parsed markdown for inspection |

### Ingestion
| Var | Default | Notes |
|-----|---------|-------|
| `CHUNK_SIZE` | `1000` | Chars per chunk |
| `CHUNK_OVERLAP` | `150` | |
| `EMBED_BATCH_SIZE` | `64` | |
| `ENABLE_HYPOTHETICAL_QUESTIONS` | `true` | HyDE; requires LLM |
| `QUESTIONS_PER_CHUNK` | `3` | |
| `HYPOTHETICAL_QUESTIONS_MAX_TOKENS` | `256` | |
| `ENABLE_LLM_METADATA_EXTRACTION` | `true` | Requires LLM |
| `DEBUG_INGESTION` | `true` | Verbose ingestion logs |

### Retrieval
| Var | Default | Notes |
|-----|---------|-------|
| `ENABLE_SELF_QUERY` | `true` | LLM extracts filters from query; requires LLM |
| `ENABLE_ANSWER_SYNTHESIS` | `true` | LLM generates answer from chunks; requires LLM |
| `RERANKER_PROVIDER` | `none` | `cross_encoder` \| `none` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |

---

## Usage

```python
from rag_pipeline import build_pipelines

ingestion, retrieval = build_pipelines()

# Ingest
ingestion.ingest(["path/to/doc.pdf", "path/to/doc2.md"])

# Retrieve chunks
chunks = retrieval.retrieve("attention mechanism in transformers", top_k=5)

# Or get synthesized answer (when ENABLE_ANSWER_SYNTHESIS=true)
result = retrieval.ask("What is multi-head attention?", top_k=5)
print(result["answer"])
for chunk in result["chunks"]:
    print(chunk.chunk_id, chunk.score, chunk.metadata)
```

### Manual filter override
```python
chunks = retrieval.retrieve(
    "transformer architecture",
    top_k=5,
    filters={"year": 2017, "domain": "nlp"},
)
```

Self-Query bypasses filter extraction when `filters` is explicitly passed.

---

## Recommended Production File Structure

The current repository is intentionally compact, but for production use it is easier to maintain when the code is split by responsibility:

```text
rag_pipeline/
├── pyproject.toml
├── README.md
├── .env.example
├── requirements.txt
├── src/
│   └── rag_pipeline/
│       ├── __init__.py
│       ├── config.py
│       ├── orchestrator.py
│       ├── ingestion.py
│       ├── retrieval.py
│       ├── base.py
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── llm.py
│       │   ├── reranker.py
│       │   ├── sparse.py
│       │   ├── vector_store.py
│       │   └── docling_parser.py
│       ├── models/
│       │   ├── __init__.py
│       ├── utils/
│       │   └── __init__.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_orchestrator.py
│   └── fixtures/
├── examples/
│   └── example_usage.py
└── data/
    ├── sample.pdf
    └── test_dataset.json
```

Why this works well in production:

1. `src/` prevents accidental imports from the repository root and matches standard packaging practice.
2. `ingestion/` and `retrieval/` separate pipeline concerns into smaller, testable modules.
3. `models/` keeps dataclasses and shared schema isolated from workflow logic.
4. `tests/` cleanly separates unit tests, integration tests, and smoke tests.
5. `examples/` keeps runnable demos out of the importable library surface.
6. `data/` keeps sample assets and evaluation datasets away from source code.

If you want to evolve the current layout incrementally, a good first pass is:

1. Move runnable scripts into `examples/`.
2. Add a `tests/` directory with unit and integration coverage.
3. Introduce a `pyproject.toml` for packaging, formatting, and test tooling.
4. Split `ingestion.py` and `retrieval.py` into package directories once each grows beyond a single responsibility.

---

## Self-Query Filter Fields

LLM extracts these from natural language queries automatically:

| Field | Type | Example |
|-------|------|---------|
| `year` | int | `2017` |
| `domain` | string | `"finance"` |
| `paper` | string | `"Attention Is All You Need"` |
| `source` | string | `"arxiv"` |

---

## Extending

Add a new provider:
1. Implement the relevant ABC from `base.py` (`DenseEmbedder`, `SparseEmbedder`, `LLMProvider`, `VectorStore`)
2. Add a branch in the matching `build_*()` function in `config.py`
3. Set the env var

No changes to `ingestion.py` or `retrieval.py`.

---

## Current File Map

```
README.md
requirements.txt
example_usage.py
run_test_dataset.py
test_pipeline_smoke.py
test_dataset.json
sample.pdf
parsed_documents/
rag_pipeline/
  __init__.py
  base.py
  config.py
  ingestion.py
  retrieval.py
  orchestrator.py
  providers/
    __init__.py
    docling_parser.py
    embeddings.py
    llm.py
    reranker.py
    sparse.py
    vector_store.py
providers/
```

---

## License

MIT
