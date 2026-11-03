"""RAGPipeline: orquestación completa del ciclo ingest → retrieve → generate.

Es la única pieza que necesitan la API, el CLI y el servidor MCP.
"""

from __future__ import annotations

import time
from typing import Any

from universal_rag.core.chunker import Chunker
from universal_rag.core.config import Settings
from universal_rag.core.connectors import (
    BaseConnector,
    ConnectorNotFoundError,
    ConnectorRegistry,
)
from universal_rag.core.embeddings import BaseEmbedder, create_embedder
from universal_rag.core.generator import BaseGenerator, create_generator
from universal_rag.core.models import (
    Answer,
    Citation,
    Document,
    IngestResult,
    RetrievedChunk,
)
from universal_rag.core.retriever import Retriever
from universal_rag.core.store import BaseVectorStore, create_store
from universal_rag.core.utils import chunk_quote, extract_citation_indexes, get_logger

logger = get_logger("pipeline")


class RAGPipeline:
    """Pipeline universal: conectores → chunking → embeddings → store → citas."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        registry: ConnectorRegistry | None = None,
        chunker: Chunker | None = None,
        embedder: BaseEmbedder | None = None,
        store: BaseVectorStore | None = None,
        generator: BaseGenerator | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.registry = registry or ConnectorRegistry.discover(self.settings.connector_dirs)
        self.chunker = chunker or Chunker(self.settings.chunk_size, self.settings.chunk_overlap)
        self.embedder = embedder or create_embedder(self.settings)
        self.store = store or create_store(self.settings)
        self.generator = generator or create_generator(self.settings)
        self.retriever = Retriever(self.embedder, self.store, top_k=self.settings.top_k)

    # ------------------------------------------------------------------ INGEST
    def ingest(self, source: str, connector: str | None = None, **options: Any) -> IngestResult:
        """Ingesta una fuente (ruta, URL, texto…) resolviendo el conector automáticamente.

        ``connector`` fuerza un conector concreto por nombre (opcional).
        Las claves de ``options`` se pasan tal cual a ``connector.fetch()``.
        """
        try:
            if connector:
                conn = self.registry.get(connector)
            else:
                conn = self.registry.resolve(source)
        except ConnectorNotFoundError as exc:
            return IngestResult(connector=connector or "?", errors=[str(exc)])

        try:
            documents = conn.fetch(source, **options)
        except Exception as exc:  # noqa: BLE001 — el fallo de una fuente no derriba el sistema
            logger.exception("El conector '%s' falló con: %s", conn.name, source)
            return IngestResult(connector=conn.name, errors=[f"{type(exc).__name__}: {exc}"])

        return self._ingest_documents(documents, conn)

    def ingest_text(
        self,
        text: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Atajo para ingestar texto directo sin pasar por resolución de fuente."""
        text_connector = self.registry.get("text")
        document = text_connector.from_text(text, title=title, metadata=metadata)
        return self._ingest_documents([document], text_connector)

    def _ingest_documents(self, documents: list[Document], connector: BaseConnector) -> IngestResult:
        result = IngestResult(connector=connector.name)
        for document in documents:
            if not document.content.strip():
                result.skipped += 1
                continue
            try:
                # Semántica de re-ingesta: la misma URI reemplaza su versión anterior.
                self.store.delete_document(document.id)
                chunks = self.chunker.chunk_document(document)
                if not chunks:
                    result.skipped += 1
                    continue
                embeddings = self.embedder.embed([chunk.text for chunk in chunks])
                self.store.add(chunks, embeddings)
                result.documents += 1
                result.chunks += len(chunks)
                result.document_ids.append(document.id)
                logger.info(
                    "Ingestado '%s' (%s): %d fragmentos", document.title or document.uri, connector.name, len(chunks)
                )
            except Exception as exc:  # noqa: BLE001 — aislamiento por documento
                logger.exception("Fallo ingesta del documento %s", document.uri)
                result.errors.append(f"{document.uri}: {type(exc).__name__}: {exc}")
        return result

    # ------------------------------------------------------------------- QUERY
    def query(self, question: str, k: int | None = None, where: dict[str, Any] | None = None) -> Answer:
        """Responde una pregunta con citas verificables a las fuentes."""
        started = time.perf_counter()
        contexts = self.retriever.retrieve(question, k=k, where=where)
        if not contexts:
            return Answer(
                question=question,
                answer=(
                    "No encontré información relevante en la base de conocimiento para esa pregunta. "
                    "Prueba con otros términos o ingesta más contenido."
                ),
                citations=[],
                model=self.generator.name,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                context_count=0,
            )
        text, model_name = self.generator.generate(question, contexts)
        citations = self._build_citations(text, contexts)
        return Answer(
            question=question,
            answer=text,
            citations=citations,
            model=model_name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            context_count=len(contexts),
        )

    def _build_citations(self, answer_text: str, contexts: list[RetrievedChunk]) -> list[Citation]:
        """Convierte los marcadores [n] del texto en citas estructuradas."""
        indexes = extract_citation_indexes(answer_text)
        valid = [i for i in indexes if 1 <= i <= len(contexts)]
        if not valid:
            # Sin marcadores: citamos el mejor fragmento recuperado (top-1).
            valid = [1]
        citations: list[Citation] = []
        for i in valid:
            retrieved = contexts[i - 1]
            meta = retrieved.chunk.metadata
            start_time = meta.get("start_time")
            end_time = meta.get("end_time")
            has_time = start_time is not None or end_time is not None
            if has_time:
                from universal_rag.core.transcript import format_range

                quote_text = f"{format_range(start_time, end_time)} {chunk_quote(retrieved.chunk.text)}"
            else:
                quote_text = chunk_quote(retrieved.chunk.text)
            citations.append(
                Citation(
                    chunk_id=retrieved.chunk.id,
                    document_id=retrieved.chunk.document_id,
                    source=str(meta.get("source", "?")),
                    uri=str(meta.get("uri", "")),
                    title=meta.get("title"),
                    quote=quote_text,
                    score=retrieved.score,
                    start_time=float(start_time) if start_time is not None else None,
                    end_time=float(end_time) if end_time is not None else None,
                    timestamp=format_range(start_time, end_time) if has_time else None,
                )
            )
        return citations

    # ------------------------------------------------------------- administración
    def stats(self) -> dict[str, Any]:
        documents = self.store.list_documents()
        return {
            "version": _version(),
            "store": self.store.name,
            "embedder": self.embedder.name,
            "embedding_dimensions": self.embedder.dimension,
            "generator": self.generator.name,
            "chunk_size": self.chunker.size,
            "chunk_overlap": self.chunker.overlap,
            "top_k": self.settings.top_k,
            "documents": len(documents),
            "chunks": self.store.count_chunks(),
            "connectors": self.registry.names(),
            "connector_report": [
                {"origin": r.origin, "ok": r.ok, "error": r.error, "name": r.connector.name if r.connector else None}
                for r in self.registry.records
            ],
        }

    def documents(self) -> list[dict[str, Any]]:
        return self.store.list_documents()

    def delete_document(self, document_id: str) -> bool:
        removed = self.store.delete_document(document_id)
        return removed > 0

    def clear(self) -> None:
        self.store.clear()


def _version() -> str:
    from universal_rag import __version__

    return __version__
