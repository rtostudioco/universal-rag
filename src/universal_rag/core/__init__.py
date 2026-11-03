"""Núcleo de Universal RAG."""

from universal_rag.core.chunker import Chunker
from universal_rag.core.config import Settings
from universal_rag.core.connectors import (
    BaseConnector,
    ConnectorNotFoundError,
    ConnectorRegistry,
)
from universal_rag.core.embeddings import (
    BaseEmbedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
)
from universal_rag.core.generator import (
    BaseGenerator,
    EchoGenerator,
    GeneratorError,
    OpenAICompatGenerator,
)
from universal_rag.core.models import Answer, Chunk, Citation, Document, IngestResult, RetrievedChunk
from universal_rag.core.pipeline import RAGPipeline
from universal_rag.core.retriever import Retriever
from universal_rag.core.store import (
    BaseVectorStore,
    ChromaVectorStore,
    InMemoryVectorStore,
)

__all__ = [
    "Answer",
    "BaseConnector",
    "BaseEmbedder",
    "BaseGenerator",
    "BaseVectorStore",
    "ChromaVectorStore",
    "Chunk",
    "Chunker",
    "Citation",
    "ConnectorNotFoundError",
    "ConnectorRegistry",
    "Document",
    "EchoGenerator",
    "GeneratorError",
    "HashEmbedder",
    "InMemoryVectorStore",
    "IngestResult",
    "OpenAICompatGenerator",
    "RAGPipeline",
    "RetrievedChunk",
    "Retriever",
    "SentenceTransformerEmbedder",
    "Settings",
]
