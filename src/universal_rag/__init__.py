"""Universal RAG — pipeline de Retrieval-Augmented Generation plugable.

Núcleo:
    Document, Chunk            modelos de datos normalizados
    BaseConnector              contrato universal de conectores (text/file/url/...)
    ConnectorRegistry          registro con auto-descubrimiento por convención
    Chunker                    división configurable con solapamiento
    BaseEmbedder / ...         embeddings locales (hash determinista o sentence-transformers)
    BaseVectorStore / ...      almacenes vectoriales (ChromaDB o memoria)
    Retriever                  búsqueda semántica con scores
    BaseGenerator / ...        generación plugable (extractiva o LLM compatible OpenAI)
    RAGPipeline                orquestación completa: ingest -> query con citas
"""

from universal_rag.core.config import Settings
from universal_rag.core.models import (
    Answer,
    Chunk,
    Citation,
    Document,
    IngestResult,
    RetrievedChunk,
)
from universal_rag.core.pipeline import RAGPipeline

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Document",
    "IngestResult",
    "RAGPipeline",
    "RetrievedChunk",
    "Settings",
    "__version__",
]
