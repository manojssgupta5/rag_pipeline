"""
Config-driven factory.

This is the single place that knows about concrete provider classes.
ingestion.py and retrieval.py only ever see the abstract interfaces in
base.py. To add a new provider (a new vector store, a new embedding
model), implement it against the relevant ABC in providers/ and add one
branch here -- nothing else changes.
"""
from __future__ import annotations

import os

from typing import Any, Dict, Optional, Tuple, Union
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from .base import DenseEmbedder, LLMProvider, SparseEmbedder, VectorStore
from .ingestion import (
    HypotheticalQuestionConfig,
    HypotheticalQuestionGenerator,
    IngestionPipeline,
    MarkdownHeaderChunker,
    MetadataEnricher,
    RecursiveTextChunker,
    LLMMetadataExtractor,
)
from .providers.docling_parser import DocumentParser, DoclingDocumentParser, PlainTextDocumentParser
from .retrieval import RetrievalPipeline, SelfQueryRetriever, AnswerSynthesizer

load_dotenv()

def _get_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

#dense embedder
def build_dense_embedder() -> DenseEmbedder:
    provider = os.environ.get( "EMBEDDING_PROVIDER", "ollama")
    if provider == "openai": 
        from .providers.embeddings import OpenAIDenseEmbedder
        return OpenAIDenseEmbedder( 
            model=os.environ.get( "EMBEDDING_MODEL", "qwen3-embedding:8b" ),
            api_key=os.environ.get( "OPENAI_API_KEY" ),
            base_url=os.environ.get( "OPENAI_BASE_URL" ),
            dimension=int( os.environ.get( "EMBEDDING_DIMENSION", "1536" ) ), )

    if provider == "ollama": 
        from .providers.embeddings import OllamaDenseEmbedder
        return OllamaDenseEmbedder( 
            model=os.environ.get( "EMBEDDING_MODEL", "qwen3-embedding:8b" ),
            base_url=os.environ.get( "OLLAMA_BASE_URL", "http://localhost:11434" ), 
            dimension=int( os.environ.get( "EMBEDDING_DIMENSION", "768" ) ) )

    if provider == "sentence_transformers": 
        from .providers.embeddings import ( SentenceTransformerDenseEmbedder, ) 
        return SentenceTransformerDenseEmbedder( 
            model_name=os.environ.get( "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5", ) ) 
            
    raise ValueError( f"Unknown EMBEDDING_PROVIDER: {provider}" )

#sparse embedder
def build_sparse_embedder() -> Optional[SparseEmbedder]:
    provider = os.environ.get( "SPARSE_PROVIDER", "hashing_tf", )

    if provider == "none": return None

    if provider == "hashing_tf": 
        from .providers.sparse import ( HashingTFSparseEmbedder, ) 
        return HashingTFSparseEmbedder( 
            vocab_size=int( os.environ.get( "SPARSE_VOCAB_SIZE", str(2 ** 18), ) ), 
            remove_stopwords=_get_bool( os.environ.get( "REMOVE_STOPWORDS", "true", ), True, ), ) 
    
    raise ValueError( f"Unknown SPARSE_PROVIDER: {provider}" )

#LLM
def build_llm() -> Optional[LLMProvider]:
    provider = os.environ.get( "LLM_PROVIDER", "none", )

    if provider == "none": return None

    if provider == "ollama":
        from .providers.llm import OllamaLLM
        return OllamaLLM(
            model=os.environ.get("LLM_MODEL", "llama3:8b"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "openai": 
        from .providers.llm import OpenAILLM 
        return OpenAILLM( 
            model=os.environ.get( "LLM_MODEL", "gpt-4o-mini", ), 
            api_key=os.environ.get( "OPENAI_API_KEY", ), 
            base_url=os.environ.get( "OPENAI_BASE_URL", ), )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def build_reranker():
    provider = os.environ.get("RERANKER_PROVIDER", "none")
    if provider == "none":
        return None
    
    if provider == "cross_encoder":
        from .providers.reranker import CrossEncoderReranker
        return CrossEncoderReranker(model_name=os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        
    raise ValueError(f"Unknown reranker provider: {provider!r}")


def build_vector_store(dense_dim: int) -> VectorStore:
    provider = os.environ.get( "VECTOR_STORE_PROVIDER", "qdrant", )

    if provider == "qdrant":
        from .providers.vector_store import QdrantVectorStore

        return QdrantVectorStore(
            collection_name=os.environ.get("QDRANT_COLLECTION", "support_kb"),
            dense_dim=dense_dim,
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            recreate=_get_bool(os.environ.get("QDRANT_RECREATE", "false"), False),
        )

    raise ValueError(f"Unknown vector store provider: {provider!r}")

def build_document_parser() -> DocumentParser:

    provider = os.environ.get( "DOCUMENT_PARSER_PROVIDER", "plain", )

    if provider == "plain": return PlainTextDocumentParser()

    if provider == "docling": 
        allowed_formats_raw = os.environ.get( "ALLOWED_FORMATS", "pdf,docx,pptx,html,md", ) 
        allowed_formats = [ fmt.strip() for fmt in allowed_formats_raw.split(",") if fmt.strip() ] 
        
        return DoclingDocumentParser( 
            allowed_formats=allowed_formats, 
            enable_ocr=_get_bool( os.environ.get( "ENABLE_OCR", "true", ), True, ), 
            enable_table_structure=_get_bool( os.environ.get( "ENABLE_TABLE_STRUCTURE", "true", ), True, ), 
            artifacts_path=os.environ.get( "DOCLING_ARTIFACTS_PATH", ), 
            save_markdown_to=os.environ.get( "SAVE_MARKDOWN_TO", ), )

    raise ValueError( f"Unknown DOCUMENT_PARSER_PROVIDER: {provider}" )

# --------------------------------------------------------------------------- #
# Full pipeline assembly
# --------------------------------------------------------------------------- #
def build_pipelines() -> Tuple[IngestionPipeline, Any]:
    """Build (IngestionPipeline, RetrievalPipeline) sharing one vector store
    and one set of embedders, so ingestion and retrieval are guaranteed to
    use identical embedding spaces."""

    #embedding
    dense_embedder = build_dense_embedder() 
    sparse_embedder = build_sparse_embedder()

    #vector store
    vector_store = build_vector_store( dense_dim=dense_embedder.dimension, )

    #chunker
    _chunk_size    = int(os.environ.get("CHUNK_SIZE",    "1000"))
    _chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "150"))
    _chunker_strategy = os.environ.get("CHUNKER_STRATEGY", "recursive").lower()

    if _chunker_strategy == "markdown_header":
        chunker = MarkdownHeaderChunker(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
        )
    else:
        chunker = RecursiveTextChunker(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
        )

    #hypothetical question generator and LLM components
    llm = build_llm()
    hyp_q_generator = None
    llm_extractor = None
    
    enable_hyp_q = ( os.environ.get( "ENABLE_HYPOTHETICAL_QUESTIONS", "true", ).lower() == "true" )
    enable_llm_extractor = ( os.environ.get( "ENABLE_LLM_METADATA_EXTRACTION", "true", ).lower() == "true" )
    
    if llm is not None:
        if enable_hyp_q:
            hyp_q_generator = ( 
                HypotheticalQuestionGenerator( 
                    llm=llm, 
                    config=HypotheticalQuestionConfig( 
                        enabled=True, 
                        questions_per_chunk=int( os.environ.get( "QUESTIONS_PER_CHUNK", "3", ) ), 
                        max_tokens=int( os.environ.get( "HYPOTHETICAL_QUESTIONS_MAX_TOKENS", "256", ) ), ), 
                ) 
            )
        if enable_llm_extractor:
            llm_extractor = LLMMetadataExtractor(llm=llm)

    #ingestion
    ingestion = IngestionPipeline(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        vector_store=vector_store,
        chunker=chunker,
        metadata_enricher=MetadataEnricher(),
        hypothetical_question_generator=hyp_q_generator,
        document_parser=build_document_parser(),
        llm_metadata_extractor=llm_extractor,
        embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "64"),),
        debug=_get_bool(os.environ.get("DEBUG_INGESTION","true"),True),
    )

    #retrieval
    reranker = build_reranker()
    retrieval = RetrievalPipeline(
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        vector_store=vector_store,
        reranker=reranker,
    )
    
    enable_self_query = ( os.environ.get( "ENABLE_SELF_QUERY", "true", ).lower() == "true" )
    if llm is not None and enable_self_query:
        retrieval = SelfQueryRetriever(retriever=retrieval, llm=llm)

    enable_synthesis = ( os.environ.get( "ENABLE_ANSWER_SYNTHESIS", "true", ).lower() == "true" )
    if llm is not None and enable_synthesis:
        retrieval = AnswerSynthesizer(retriever=retrieval, llm=llm)

    return ingestion, retrieval
