"""Almacenes vectoriales: contrato + ChromaDB (default) + memoria.

* :class:`ChromaVectorStore` — persistente, embebido, sin servidor. Intercambiable.
* :class:`InMemoryVectorStore` — volátil, cero dependencias (tests/demo rápida).

Todos los scores se reportan como similitud coseno (mayor = mejor), aunque el
store internamente use distancias.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from universal_rag.core.models import Chunk, RetrievedChunk
from universal_rag.core.utils import flatten_metadata, get_logger

logger = get_logger("store")


class BaseVectorStore(ABC):
    """Contrato del almacén vectorial."""

    name: ClassVar[str] = "base"

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        """Todos los fragmentos (opcionalmente filtrados) — índice léxico BM25 (Fase 3)."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int: ...

    @abstractmethod
    def count_chunks(self) -> int: ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def clear(self) -> None: ...

    def close(self) -> None:  # noqa: B027 — hook opcional para stores que necesiten liberar recursos
        pass


# --------------------------------------------------------------------- memoria
class InMemoryVectorStore(BaseVectorStore):
    """Almacén volátil en memoria con similitud coseno pura."""

    name = "memory"

    def __init__(self) -> None:
        # entries: (id, embedding, text, metadata)
        self._entries: dict[str, tuple[list[float], str, dict[str, Any]]] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks y embeddings deben tener la misma longitud")
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            meta = flatten_metadata({"document_id": chunk.document_id, **chunk.metadata})
            self._entries[chunk.id] = (list(embedding), chunk.text, meta)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        scored: list[tuple[float, str, str, dict[str, Any]]] = []
        for chunk_id, (vec, text, meta) in self._entries.items():
            if where and not self._matches(meta, where):
                continue
            scored.append((self._cosine(embedding, vec), chunk_id, text, meta))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk_id, text, meta in scored[: max(0, k)]:
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=chunk_id,
                        document_id=str(meta.get("document_id", "")),
                        text=text,
                        index=int(meta.get("index", 0)),
                        total=int(meta.get("total", 0)),
                        metadata=dict(meta),
                    ),
                    score=round(score, 6),
                )
            )
        return results

    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chunk_id, (_, text, meta) in self._entries.items():
            if where and not self._matches(meta, where):
                continue
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=str(meta.get("document_id", "")),
                    text=text,
                    index=int(meta.get("index", 0)),
                    total=int(meta.get("total", 0)),
                    metadata=dict(meta),
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> int:
        victims = [cid for cid, (_, _, meta) in self._entries.items() if meta.get("document_id") == document_id]
        for cid in victims:
            del self._entries[cid]
        return len(victims)

    def count_chunks(self) -> int:
        return len(self._entries)

    def list_documents(self) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for _, (_, _, meta) in self._entries.items():
            doc_id = str(meta.get("document_id", ""))
            if doc_id not in grouped:
                grouped[doc_id] = {
                    "document_id": doc_id,
                    "source": meta.get("source", ""),
                    "uri": meta.get("uri", ""),
                    "title": meta.get("title"),
                    "chunks": 0,
                }
            grouped[doc_id]["chunks"] += 1
        return sorted(grouped.values(), key=lambda d: d["document_id"])

    def clear(self) -> None:
        self._entries.clear()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    @staticmethod
    def _matches(meta: dict[str, Any], where: dict[str, Any]) -> bool:
        for key, expected in where.items():
            if meta.get(key) != expected:
                return False
        return True


# ---------------------------------------------------------------------- chroma
class ChromaVectorStore(BaseVectorStore):
    """Almacén persistente basado en ChromaDB embebido (sin servidor)."""

    name = "chroma"

    def __init__(self, path: str = "./data/chroma", collection_name: str = "universal_rag") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "chromadb no está instalado. Instálalo con: pip install universal-rag"
            ) from exc
        self.path = str(path)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB listo en %s (colección '%s')", self.path, collection_name)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks y embeddings deben tener la misma longitud")
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[self._metadata_for(c) for c in chunks],
        )

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        response = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(max(1, k), self.collection.count()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        results: list[RetrievedChunk] = []
        ids = response.get("ids") or [[]]
        docs = response.get("documents") or [[]]
        metas = response.get("metadatas") or [[]]
        dists = response.get("distances") or [[]]
        for chunk_id, text, meta, dist in zip(ids[0], docs[0], metas[0], dists[0], strict=False):
            meta = dict(meta or {})
            results.append(
                RetrievedChunk(
                    chunk=Chunk(
                        id=str(chunk_id),
                        document_id=str(meta.get("document_id", "")),
                        text=str(text or ""),
                        index=int(meta.get("index", 0)),
                        total=int(meta.get("total", 0)),
                        metadata=meta,
                    ),
                    # Chroma devuelve distancia coseno; convertimos a similitud.
                    score=round(1.0 - float(dist), 6),
                )
            )
        return results

    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        if self.collection.count() == 0:
            return []
        data = self.collection.get(include=["documents", "metadatas"], where=where or None)
        chunks: list[Chunk] = []
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for chunk_id, text, meta in zip(ids, docs, metas, strict=False):
            meta = dict(meta or {})
            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    document_id=str(meta.get("document_id", "")),
                    text=str(text or ""),
                    index=int(meta.get("index", 0)),
                    total=int(meta.get("total", 0)),
                    metadata=meta,
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> int:
        before = self.collection.count()
        self.collection.delete(where={"document_id": document_id})
        return before - self.collection.count()

    def count_chunks(self) -> int:
        return int(self.collection.count())

    def list_documents(self) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        data = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            meta = dict(meta or {})
            doc_id = str(meta.get("document_id", ""))
            if not doc_id:
                continue
            if doc_id not in grouped:
                grouped[doc_id] = {
                    "document_id": doc_id,
                    "source": meta.get("source", ""),
                    "uri": meta.get("uri", ""),
                    "title": meta.get("title"),
                    "chunks": 0,
                }
            grouped[doc_id]["chunks"] += 1
        return sorted(grouped.values(), key=lambda d: d["document_id"])

    def clear(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 — la colección puede no existir aún
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _metadata_for(chunk: Chunk) -> dict[str, Any]:
        flat = flatten_metadata({"document_id": chunk.document_id, **chunk.metadata})
        flat["index"] = chunk.index
        flat["total"] = chunk.total
        return flat


def create_store(settings) -> BaseVectorStore:  # noqa: ANN001 — evita import circular
    """Fábrica según Settings (importación diferida para evitar ciclo)."""
    from universal_rag.core.config import Settings

    if not isinstance(settings, Settings):  # pragma: no cover
        raise TypeError("settings debe ser Settings")
    if settings.store_provider == "memory":
        return InMemoryVectorStore()
    return ChromaVectorStore(
        path=str(settings.chroma_path),
        collection_name=settings.collection_name,
    )
