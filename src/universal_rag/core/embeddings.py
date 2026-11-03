"""Embeddings: contrato + implementaciones locales.

* :class:`HashEmbedder` — determinista, sin dependencias (ideal para tests y
  modo offline). Usa hashing con signo sobre tokens (bag-of-words proyectado).
* :class:`SentenceTransformerEmbedder` — embeddings locales reales con
  ``sentence-transformers`` (privados, sin APIs externas). Modelo por defecto
  multilingüe, con buen rendimiento en español:
  ``paraphrase-multilingual-MiniLM-L12-v2``.

El embedder se carga perezosamente: importar universal_rag nunca descarga
modelos.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import ClassVar

from universal_rag.core.config import Settings
from universal_rag.core.utils import get_logger

logger = get_logger("embeddings")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class BaseEmbedder(ABC):
    """Contrato de vectorización de textos."""

    name: ClassVar[str] = "base"
    dimension: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza una lista de textos (L2-normalizado cuando aplique)."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(BaseEmbedder):
    """Embedding determinista sin dependencias (hashing trick con signo)."""

    name = "hash"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimension = max(32, int(dimensions))

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            tokens = _TOKEN_RE.findall((text or "").lower())
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[index] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embeddings locales reales con sentence-transformers (carga perezosa)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # se carga en el primer embed()
        self.dimension = 0

    def _ensure_model(self):  # noqa: ANN202
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers no está instalado. "
                    "Instálalo con: pip install 'universal-rag[local]'"
                ) from exc
            logger.info("Cargando modelo de embeddings '%s' (puede tardar la primera vez)…", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # compat: API nueva (get_embedding_dimension) con fallback a la antigua
            dim_fn = getattr(self._model, "get_embedding_dimension", None)
            self.dimension = int(dim_fn()) if callable(dim_fn) else int(
                self._model.get_sentence_embedding_dimension()
            )
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [row.tolist() for row in embeddings]


def create_embedder(settings: Settings) -> BaseEmbedder:
    """Fábrica que resuelve el proveedor según Settings (con fallback seguro)."""
    provider = settings.embedding_provider

    if provider == "hash":
        return HashEmbedder(settings.embedding_dimensions)

    if provider in ("auto", "sentence-transformers"):
        try:
            import sentence_transformers  # noqa: F401

            return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
        except ImportError:
            if provider == "sentence-transformers":
                raise RuntimeError(
                    "sentence-transformers no está instalado. "
                    "Instálalo con: pip install 'universal-rag[local]'"
                ) from None
            logger.warning(
                "sentence-transformers no instalado; se usa HashEmbedder (determinista). "
                "Para embeddings reales: pip install 'universal-rag[local]'"
            )
            return HashEmbedder(settings.embedding_dimensions)

    raise ValueError(f"Proveedor de embeddings desconocido: {provider}")  # pragma: no cover
