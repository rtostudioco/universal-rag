"""Retriever: búsqueda vectorial, híbrida (BM25 + vector) y reranking (Fase 3).

Modos (``RAG_RETRIEVAL``):

* ``vector``  — búsqueda semántica pura sobre el almacén vectorial (Fase 1).
* ``hybrid``  — fusión del ranking vectorial con BM25 (léxico) vía RRF o suma
                ponderada. Brillante con términos exactos (códigos, nombres
                propios) que los embeddings diluyen, y rescata consultas que
                el embedder no captura semánticamente.

Etapa opcional de reranking (``RAG_RERANKER``): se recuperan ``candidate_k``
candidatos, el reranker los reordena con un cross-encoder y se devuelve el
top-k. Con ``reranker=none`` el orden del retriever se conserva.
"""

from __future__ import annotations

from typing import Any

from universal_rag.core.bm25 import BM25Index
from universal_rag.core.config import Settings
from universal_rag.core.embeddings import BaseEmbedder
from universal_rag.core.hybrid import rrf_fuse, weighted_fuse
from universal_rag.core.models import Chunk, RetrievedChunk
from universal_rag.core.rerank import BaseReranker, create_reranker
from universal_rag.core.store import BaseVectorStore
from universal_rag.core.utils import get_logger

logger = get_logger("retriever")


class Retriever:
    """Recupera fragmentos: vectorial, híbrido (BM25+vector) y reranking opcional."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        top_k: int = 5,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = max(1, top_k)
        self.settings = settings or Settings()
        self.retrieval = self.settings.retrieval
        self.fusion = self.settings.hybrid_fusion
        self.alpha = self.settings.hybrid_alpha
        self.rrf_k = self.settings.hybrid_rrf_k
        self.candidate_k = max(1, self.settings.candidate_k)
        self.reranker: BaseReranker = create_reranker(self.settings)
        self._bm25: BM25Index | None = None
        self._corpus: list[Chunk] = []
        self._dirty = True  # el store cambió; toca reconstruir el índice léxico

    # ------------------------------------------------------------------ API
    def invalidate_index(self) -> None:
        """Marca el índice BM25 como obsoleto (tras ingestar, borrar o vaciar)."""
        self._dirty = True

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
        extra_stages = self.retrieval == "hybrid" or self.reranker.name != "noop"
        candidate_k = max(limit, self.candidate_k) if extra_stages else limit
        vector_results = self._vector(question, candidate_k, where)
        candidates = vector_results
        if self.retrieval == "hybrid":
            candidates = self._hybrid(question, vector_results, candidate_k, where)
        candidates = self.reranker.rerank(question, candidates)
        results = candidates[:limit]
        logger.debug("Recuperados %d fragmentos para: %r", len(results), question[:80])
        return results

    # ------------------------------------------------------------- internals
    def _vector(self, question: str, k: int, where: dict[str, Any] | None) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_one(question)
        return self.store.query(query_vector, k=k, where=where)

    def _hybrid(
        self,
        question: str,
        vector_results: list[RetrievedChunk],
        k: int,
        where: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        """Fusiona el ranking vectorial con BM25 sobre el mismo corpus (filtrado)."""
        corpus, bm25 = self._lexical_index(where)
        if not corpus or bm25 is None:
            return vector_results
        position = {chunk.id: i for i, chunk in enumerate(corpus)}
        lex_pairs = bm25.top(question, n=k)
        vector_pairs = [
            (position[retrieved.chunk.id], retrieved.score)
            for retrieved in vector_results
            if retrieved.chunk.id in position
        ]
        if self.fusion == "weighted":
            fused = weighted_fuse(vector_pairs, lex_pairs, alpha=self.alpha, top_n=k)
        else:
            fused = rrf_fuse(
                [[idx for idx, _ in vector_pairs], [idx for idx, _ in lex_pairs]],
                k=self.rrf_k,
                top_n=k,
            )
        return [RetrievedChunk(chunk=corpus[idx], score=round(score, 6)) for idx, score in fused]

    def _lexical_index(
        self, where: dict[str, Any] | None
    ) -> tuple[list[Chunk], BM25Index | None]:
        """Corpus + índice BM25. Con filtro se reconstruye efímero; si no, cacheado."""
        if where:
            corpus = self.store.all_chunks(where=where)
            return corpus, BM25Index([chunk.text for chunk in corpus])
        if self._dirty or self._bm25 is None:
            self._corpus = self.store.all_chunks()
            self._bm25 = BM25Index([chunk.text for chunk in self._corpus])
            self._dirty = False
            logger.debug("Índice BM25 reconstruido con %d fragmento(s)", len(self._corpus))
        return self._corpus, self._bm25
