"""Universal RAG — pipeline de Retrieval-Augmented Generation plugable.

Núcleo:
    Document, Chunk            modelos de datos normalizados
    BaseConnector              contrato universal de conectores (text/file/url/audio/…)
    ConnectorRegistry          registro con auto-descubrimiento por convención
    Chunker                    división configurable con solapamiento
    BaseEmbedder / ...         embeddings locales (hash determinista o sentence-transformers)
    BaseVectorStore / ...      almacenes vectoriales (ChromaDB o memoria)
    Retriever                  búsqueda vectorial o híbrida (BM25 + reranker)
    BaseReranker / ...         rerankers plugables (no-op o cross-encoder local)
    evaluate / load_golden     evaluación de calidad con golden sets
    BaseGenerator / ...        generación plugable (extractiva o LLM compatible OpenAI)
    CollectionManager          colecciones múltiples aisladas (Fase 4)
    JobManager / JobInfo       ingesta asíncrona por jobs (Fase 4)
    Metrics                    métricas de runtime (Fase 4)
    RAGPipeline                orquestación completa: ingest -> query con citas
"""

from universal_rag.core.collections import CollectionManager
from universal_rag.core.config import Settings
from universal_rag.core.jobs import JobInfo, JobManager, JobStatus
from universal_rag.core.metrics import Metrics
from universal_rag.core.models import (
    Answer,
    Chunk,
    Citation,
    Document,
    IngestResult,
    RetrievedChunk,
)
from universal_rag.core.pipeline import RAGPipeline

__version__ = "0.5.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "CollectionManager",
    "Document",
    "IngestResult",
    "JobInfo",
    "JobManager",
    "JobStatus",
    "Metrics",
    "RAGPipeline",
    "RetrievedChunk",
    "Settings",
    "__version__",
]
