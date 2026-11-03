"""Caché semántica de consultas (Fase 6).

Guarda las respuestas recientes indexadas por el embedding de la pregunta.
Ante una nueva consulta, si ya existe una entrada cuya pregunta es
suficientemente similar (coseno ≥ ``threshold``), se devuelve la respuesta
cacheada sin repetir recuperación + generación: ahorra latencia y coste de
LLM en preguntas repetidas o parafraseadas.

La caché vive en memoria (por proceso y por colección), es LRU por tamaño y
opcionalmente TTL, y se invalida automáticamente cuando cambia el índice
(ingesta, borrado de documento, vaciado).

Uso::

    cache = SemanticCache(embedder, threshold=0.93)
    cache.put("¿Qué es RAG?", answer.model_dump())
    payload = cache.lookup("¿Qué significa RAG?")  # → dict o None
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from universal_rag.core.embeddings import BaseEmbedder
from universal_rag.core.utils import get_logger

logger = get_logger("cache")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno pura (sin numpy): 1.0 idénticos, 0.0 ortogonales."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class SemanticCache:
    """Caché en memoria de respuestas, indexada por similitud de pregunta.

    * ``threshold`` — similitud coseno mínima (0-1) para servir un acierto.
      0.93 es conservador: solo parafraseos casi idénticos.
    * ``max_entries`` — LRU: se descarta la entrada más antigua al superar.
    * ``ttl_seconds`` — expiración por entrada (``None`` = sin TTL).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        *,
        threshold: float = 0.93,
        max_entries: int = 512,
        ttl_seconds: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.threshold = float(threshold)
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        # Orden de inserción (LRU): {"question", "vector", "payload", "created"}
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ API
    def lookup(self, question: str) -> dict[str, Any] | None:
        """Respuesta cacheada para la pregunta más similar ≥ threshold (o None)."""
        question = (question or "").strip()
        if not question:
            return None
        vector = self.embedder.embed_one(question)
        now = time.monotonic()
        with self._lock:
            alive: list[dict[str, Any]] = []
            best_score = -1.0
            best_payload: dict[str, Any] | None = None
            for entry in self._entries:
                if self._expired(entry, now):
                    continue  # expirada: se descarta de la lista
                alive.append(entry)
                score = cosine_similarity(vector, entry["vector"])
                if score > best_score:
                    best_score = score
                    best_payload = entry["payload"]
            self._entries = alive
            if best_payload is not None and best_score >= self.threshold:
                self.hits += 1
                logger.debug("Acierto de caché (%.3f ≥ %.2f)", best_score, self.threshold)
                return dict(best_payload)
            self.misses += 1
            return None

    def put(self, question: str, payload: dict[str, Any]) -> None:
        """Registra la respuesta de una pregunta (payload serializable JSON)."""
        question = (question or "").strip()
        if not question:
            return
        with self._lock:
            self._entries.append(
                {
                    "question": question,
                    "vector": self.embedder.embed_one(question),
                    "payload": payload,
                    "created": time.monotonic(),
                }
            )
            while len(self._entries) > self.max_entries:
                self._entries.pop(0)

    def clear(self) -> None:
        """Invalida toda la caché (tras ingestar, borrar o vaciar el índice)."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": True,
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
                "threshold": self.threshold,
            }

    # -------------------------------------------------------------- internos
    def _expired(self, entry: dict[str, Any], now: float) -> bool:
        return self.ttl_seconds is not None and (now - entry["created"]) > self.ttl_seconds
