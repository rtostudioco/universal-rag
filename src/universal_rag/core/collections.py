"""Colecciones múltiples: bases de conocimiento aisladas (Fase 4).

Cada colección es un :class:`RAGPipeline` independiente (su propio store,
chunker y retriever) identificado por nombre. Los recursos caros —embedder,
generator, registro de conectores— se comparten entre colecciones.

* La colección por defecto (``default``) usa la configuración tal cual
  (retro-compatibilidad con índices creados en Fases 1–3).
* Las demás colecciones aíslan su persistencia: con ChromaDB usan
  ``<chroma_path>/<nombre>``.

Uso::

    manager = CollectionManager(settings)
    manager.get("legal").ingest_text("El contrato…")
    manager.get("faq").ingest_text("¿Horario? …")
    manager.get("legal").query("¿cuánto dura el contrato?")
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from universal_rag.core.config import Settings
from universal_rag.core.embeddings import BaseEmbedder, create_embedder
from universal_rag.core.generator import BaseGenerator, create_generator
from universal_rag.core.pipeline import RAGPipeline

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


def normalize_name(name: str | None, default: str = "default") -> str:
    """Normaliza y valida un nombre de colección (evita rutas maliciosas)."""
    value = (name or "").strip() or default
    if not _NAME_RE.match(value):
        raise ValueError(
            f"Nombre de colección inválido: '{value}' "
            "(usa letras, números, punto, guion o guion bajo; máx. 64 caracteres)"
        )
    return value


class CollectionManager:
    """Gestor de pipelines aislados por colección (lazy y thread-safe)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: BaseEmbedder | None = None,
        generator: BaseGenerator | None = None,
        pipelines: dict[str, RAGPipeline] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._lock = threading.RLock()
        self._pipelines: dict[str, RAGPipeline] = dict(pipelines or {})
        # Recursos caros compartidos entre colecciones (se crean lazy una vez).
        self._embedder = embedder
        self._generator = generator

    # ------------------------------------------------------------------ API
    def get(self, name: str | None = None) -> RAGPipeline:
        """Devuelve el pipeline de la colección (lo crea la primera vez)."""
        key = normalize_name(name, default=self.settings.default_collection)
        with self._lock:
            pipeline = self._pipelines.get(key)
            if pipeline is None:
                pipeline = self._build(key)
                self._pipelines[key] = pipeline
            return pipeline

    def register(self, name: str, pipeline: RAGPipeline) -> None:
        """Registra un pipeline pre-construido (útil en tests y embedders custom)."""
        key = normalize_name(name, default=self.settings.default_collection)
        with self._lock:
            self._pipelines[key] = pipeline

    def names(self) -> list[str]:
        """Colecciones ya instanciadas en este proceso (orden alfabético)."""
        with self._lock:
            return sorted(self._pipelines)

    def list_collections(self) -> list[dict[str, Any]]:
        """Resumen por colección: documentos, fragmentos y store."""
        summary: list[dict[str, Any]] = []
        for name in self.names():
            pipeline = self.get(name)
            stats = pipeline.stats()
            summary.append(
                {
                    "name": name,
                    "documents": stats["documents"],
                    "chunks": stats["chunks"],
                    "store": stats["store"],
                    "retrieval": stats["retrieval"],
                }
            )
        return summary

    def delete(self, name: str | None = None) -> bool:
        """Vacía y descarta una colección instanciada. Devuelve True si existía."""
        key = normalize_name(name, default=self.settings.default_collection)
        with self._lock:
            pipeline = self._pipelines.pop(key, None)
        if pipeline is None:
            return False
        pipeline.clear()
        return True

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._pipelines

    # -------------------------------------------------------------- internos
    def _build(self, name: str) -> RAGPipeline:
        settings = self._settings_for(name)
        if self._embedder is None:
            self._embedder = create_embedder(settings)
        if self._generator is None:
            self._generator = create_generator(settings)
        return RAGPipeline(
            settings=settings,
            embedder=self._embedder,
            generator=self._generator,
        )

    def _settings_for(self, name: str) -> Settings:
        """Settings de una colección: la default es tal cual; el resto aísla path."""
        if name == self.settings.default_collection:
            return self.settings
        data = self.settings.model_dump()
        if self.settings.store_provider == "chroma":
            data["chroma_path"] = str(Path(self.settings.chroma_path) / name)
        return Settings(**data)
