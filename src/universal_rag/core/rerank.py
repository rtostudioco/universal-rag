"""Rerankers pluggables (Fase 3).

Tras la recuperación (vector o híbrida), un reranker reordena los candidatos
con un modelo más fino (cross-encoder) para mejorar la precisión del top-k.
El pipeline recupera ``RAG_CANDIDATE_K`` candidatos y el reranker los
reordena antes de devolver ``k``.

* :class:`NoopReranker` — por defecto; conserva el orden del retriever.
* :class:`CrossEncoderReranker` — ``CrossEncoder`` de sentence-transformers
  (carga perezosa; requiere ``pip install 'universal-rag[local]'``).

Modelo por defecto: ``BAAI/bge-reranker-base`` (multilingüe, buen rendimiento
en español). El score final del cross-encoder se pasa por sigmoide para
mantener los scores en ``[0, 1]`` y monótonos con el orden.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from universal_rag.core.models import RetrievedChunk
from universal_rag.core.utils import get_logger

logger = get_logger("rerank")


class BaseReranker:
    """Contrato del reranker: reordena candidatos ya recuperados."""

    name: ClassVar[str] = "base"

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Devuelve los candidatos reordenados (por defecto, sin cambios)."""
        return chunks


class NoopReranker(BaseReranker):
    """Pasa-through: conserva el orden del retriever (comportamiento Fase 1/2)."""

    name = "noop"


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CrossEncoderReranker(BaseReranker):
    """Reordena candidatos con un cross-encoder de sentence-transformers.

    El modelo solo puntúa pares ``(consulta, texto)`` — mucho más fino que el
    bi-encoder de la recuperación — pero es O(candidatos) por consulta, por eso
    se aplica solo a ``candidate_k`` candidatos y no a todo el índice.
    """

    name = "cross-encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers no está instalado. "
                    "Instálalo con: pip install 'universal-rag[local]'"
                ) from exc
            logger.info("Cargando reranker '%s' (puede tardar la primera vez)…", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if len(chunks) <= 1:
            return list(chunks)
        model = self._ensure_model()
        pairs = [(query, retrieved.chunk.text) for retrieved in chunks]
        raw_scores = model.predict(pairs)
        scored = [
            RetrievedChunk(chunk=retrieved.chunk, score=round(_sigmoid(float(score)), 6))
            for retrieved, score in zip(chunks, raw_scores, strict=False)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)


def create_reranker(settings) -> BaseReranker:  # noqa: ANN001 — evita import circular
    """Fábrica según Settings (importación diferida para evitar ciclo)."""
    from universal_rag.core.config import Settings

    if not isinstance(settings, Settings):  # pragma: no cover
        raise TypeError("settings debe ser Settings")
    if settings.reranker == "cross-encoder":
        return CrossEncoderReranker(settings.rerank_model, settings.embedding_device)
    return NoopReranker()
