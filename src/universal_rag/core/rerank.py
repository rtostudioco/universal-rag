"""Rerankers pluggables (Fase 3).

Tras la recuperación (vector o híbrida), un reranker reordena los candidatos
con un modelo más fino (cross-encoder) para mejorar la precisión del top-k.
El pipeline recupera ``RAG_CANDIDATE_K`` candidatos y el reranker los
reordena antes de devolver ``k``.

* :class:`NoopReranker` — por defecto; conserva el orden del retriever.
* :class:`CrossEncoderReranker` — ``CrossEncoder`` de sentence-transformers
  (carga perezosa; requiere ``pip install 'universal-rag[local]'``).
* :class:`LLMReranker` — un LLM (endpoint compatible OpenAI) puntúa la
  relevancia de cada candidato y reordena (``RAG_RERANKER=llm``). Ante
  cualquier fallo (red, JSON inválido) degrada con gracia: conserva el orden
  original del retriever.

Modelo por defecto: ``BAAI/bge-reranker-base`` (multilingüe, buen rendimiento
en español). El score final del cross-encoder se pasa por sigmoide para
mantener los scores en ``[0, 1]`` y monótonos con el orden.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, ClassVar

import httpx

from universal_rag.core.models import RetrievedChunk
from universal_rag.core.utils import get_logger, truncate

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
    if settings.reranker == "llm":
        if not settings.llm_base_url:
            logger.warning(
                "RAG_RERANKER=llm requiere RAG_LLM_BASE_URL; se conserva el orden del retriever."
            )
            return NoopReranker()
        return LLMReranker(
            base_url=settings.llm_base_url,
            model=settings.rerank_llm_model or settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
        )
    return NoopReranker()


LLM_RERANK_SYSTEM = (
    "Eres un motor de reranking de documentos. Recibes una consulta y una lista "
    "de fragmentos numerados. Responde SOLO con un array JSON de objetos "
    "con las claves «i» (número de fragmento) y «score» (relevancia 0-10), "
    "un objeto por fragmento, sin texto adicional y sin marcadores de código."
)


class LLMReranker(BaseReranker):
    """Reordena candidatos pidiendo puntuaciones de relevancia a un LLM.

    A diferencia del cross-encoder (especializado y rápido), el LLM razona
    sobre la consulta; el coste es una llamada por consulta. Estrategia de
    fallo seguro: cualquier error (red, formato, cobertura parcial) devuelve
    el orden original sin interrumpir la respuesta.
    """

    name = "llm"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_chars: int = 800,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_chars = max_chars
        self._client = client

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if len(chunks) <= 1:
            return list(chunks)
        try:
            scores = self._ask(query, chunks)
        except Exception as exc:  # noqa: BLE001 — degradación con gracia
            logger.warning("LLMReranker falló (%s); se conserva el orden original", exc)
            return list(chunks)
        # Cobertura parcial: si el LLM omitió >20% de candidatos, no reordenamos.
        if len(scores) < math.ceil(0.8 * len(chunks)):
            logger.warning("LLMReranker puntuó %d/%d candidatos; se conserva el orden", len(scores), len(chunks))
            return list(chunks)
        scored: list[RetrievedChunk] = []
        for position, retrieved in enumerate(chunks, start=1):
            value = scores.get(position)
            score = round(max(0.0, min(10.0, float(value))) / 10.0, 6) if value is not None else 0.0
            scored.append(RetrievedChunk(chunk=retrieved.chunk, score=score))
        return sorted(scored, key=lambda item: item.score, reverse=True)

    # ------------------------------------------------------------- internals
    def _ask(self, query: str, chunks: list[RetrievedChunk]) -> dict[int, float]:
        blocks = [
            f"[{position}] {truncate(retrieved.chunk.text, self.max_chars)}"
            for position, retrieved in enumerate(chunks, start=1)
        ]
        user_prompt = f"Consulta: {query.strip()}\n\nFragmentos:\n" + "\n\n".join(blocks)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": LLM_RERANK_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = self._post(payload, headers)
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Respuesta del LLM inesperada: {data!r}") from exc
        return self._parse_scores(content, len(chunks))

    def _post(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        if self._client is not None:
            response = self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_scores(content: str, expected: int) -> dict[int, float]:
        """Extrae ``{"i": n, "score": x}`` del texto (tolerante a ruido)."""
        match = re.search(r"\[.*\]", content, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Sin array JSON en la respuesta: {content[:120]!r}")
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON inválido del LLM: {exc}") from exc
        if not isinstance(items, list):
            raise ValueError("La respuesta no es un array JSON")
        scores: dict[int, float] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                position = int(item["i"])
                value = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= position <= expected:
                scores[position] = value
        if not scores:
            raise ValueError("El LLM no puntuó ningún candidato válido")
        return scores
