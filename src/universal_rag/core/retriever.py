"""Retriever: búsqueda semántica sobre el almacén vectorial."""

from __future__ import annotations

from typing import Any

from universal_rag.core.embeddings import BaseEmbedder
from universal_rag.core.models import RetrievedChunk
from universal_rag.core.store import BaseVectorStore
from universal_rag.core.utils import get_logger

logger = get_logger("retriever")


class Retriever:
    """Vectoriza la consulta y recupera los fragmentos más relevantes."""

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore, top_k: int = 5) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = max(1, top_k)

    def retrieve(
        self,
        question: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        question = (question or "").strip()
        if not question:
            return []
        limit = max(1, k or self.top_k)
        query_vector = self.embedder.embed_one(question)
        results = self.store.query(query_vector, k=limit, where=where)
        logger.debug("Recuperados %d fragmentos para: %r", len(results), question[:80])
        return results
