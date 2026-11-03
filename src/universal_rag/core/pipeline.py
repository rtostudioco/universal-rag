"""RAGPipeline: orquestación completa del ciclo ingest → retrieve → generate.

Es la única pieza que necesitan la API, el CLI y el servidor MCP.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from universal_rag.core.cache import SemanticCache
from universal_rag.core.chunker import Chunker
from universal_rag.core.config import Settings
from universal_rag.core.connectors import (
    BaseConnector,
    ConnectorNotFoundError,
    ConnectorRegistry,
)
from universal_rag.core.embeddings import BaseEmbedder, create_embedder
from universal_rag.core.generator import BaseGenerator, create_generator, tokenize_stream
from universal_rag.core.models import (
    Answer,
    Citation,
    Document,
    IngestResult,
    RetrievedChunk,
)
from universal_rag.core.retriever import Retriever
from universal_rag.core.store import BaseVectorStore, create_store
from universal_rag.core.utils import (
    chunk_quote,
    content_fingerprint,
    extract_citation_indexes,
    get_logger,
)

logger = get_logger("pipeline")

_NO_CONTEXT_ANSWER = (
    "No encontré información relevante en la base de conocimiento para esa pregunta. "
    "Prueba con otros términos o ingesta más contenido."
)


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
        self.cache: SemanticCache | None = None
        if self.settings.cache_enabled:
            self.cache = SemanticCache(
                self.embedder,
                threshold=self.settings.cache_threshold,
                max_entries=self.settings.cache_max_entries,
                ttl_seconds=self.settings.cache_ttl,
            )
        self.retriever = Retriever(
            self.embedder, self.store, top_k=self.settings.top_k, settings=self.settings
        )

    # ------------------------------------------------------------------ INGEST
    def ingest(
        self, source: str, connector: str | None = None, *, force: bool = False, **options: Any
    ) -> IngestResult:
        """Ingesta una fuente (ruta, URL, texto…) resolviendo el conector automáticamente.

        ``connector`` fuerza un conector concreto por nombre (opcional).
        Con ``force=True`` re-ingesta aunque el contenido no haya cambiado.
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

        return self._ingest_documents(documents, conn, force=force)

    def ingest_text(
        self,
        text: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> IngestResult:
        """Atajo para ingestar texto directo sin pasar por resolución de fuente."""
        text_connector = self.registry.get("text")
        document = text_connector.from_text(text, title=title, metadata=metadata)
        return self._ingest_documents([document], text_connector, force=force)

    def _ingest_documents(
        self, documents: list[Document], connector: BaseConnector, *, force: bool = False
    ) -> IngestResult:
        result = IngestResult(connector=connector.name)
        for document in documents:
            if not document.content.strip():
                result.skipped += 1
                continue
            try:
                # Ingesta incremental (Fase 4): si el contenido no cambió, no se re-vectoriza.
                content_hash = content_fingerprint(document.content)
                if not force and self.store.document_hash(document.id) == content_hash:
                    result.unchanged += 1
                    logger.info("Sin cambios '%s' — omitido", document.title or document.uri)
                    continue
                # Semántica de re-ingesta: la misma URI reemplaza su versión anterior.
                self.store.delete_document(document.id)
                chunks = self.chunker.chunk_document(document)
                if not chunks:
                    result.skipped += 1
                    continue
                for chunk in chunks:
                    chunk.metadata = {**chunk.metadata, "content_hash": content_hash}
                embeddings = self.embedder.embed([chunk.text for chunk in chunks])
                self.store.add(chunks, embeddings)
                self.retriever.invalidate_index()  # el índice léxico quedó obsoleto
                result.documents += 1
                result.chunks += len(chunks)
                result.document_ids.append(document.id)
                logger.info(
                    "Ingestado '%s' (%s): %d fragmentos", document.title or document.uri, connector.name, len(chunks)
                )
            except Exception as exc:  # noqa: BLE001 — aislamiento por documento
                logger.exception("Fallo ingesta del documento %s", document.uri)
                result.errors.append(f"{document.uri}: {type(exc).__name__}: {exc}")
        if result.documents and self.cache is not None:
            self.cache.clear()  # el índice cambió: las respuestas previas quedan obsoletas
        return result

    # ------------------------------------------------------------------- QUERY
    def _cache_get(self, question: str, started: float) -> Answer | None:
        """Busca la respuesta en la caché semántica (None si fallo o desactivada)."""
        if self.cache is None:
            return None
        payload = self.cache.lookup(question)
        if payload is None:
            return None
        answer = Answer(**{**payload, "cached": True})
        answer.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return answer

    def _cache_put(self, question: str, answer: Answer) -> None:
        if self.cache is not None and not answer.cached:
            self.cache.put(question, answer.model_dump())

    def _empty_answer(self, question: str, started: float) -> Answer:
        return Answer(
            question=question,
            answer=_NO_CONTEXT_ANSWER,
            citations=[],
            model=self.generator.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            context_count=0,
            retrieval=self.retrieval_label(),
        )

    def query(self, question: str, k: int | None = None, where: dict[str, Any] | None = None) -> Answer:
        """Responde una pregunta con citas verificables a las fuentes."""
        started = time.perf_counter()
        # Caché semántica (Fase 6): solo con parámetros por defecto (k/where cambian el resultado).
        if k is None and where is None:
            cached = self._cache_get(question, started)
            if cached is not None:
                return cached
        contexts = self.retriever.retrieve(question, k=k, where=where)
        if not contexts:
            answer = self._empty_answer(question, started)
            self._cache_put(question, answer)
            return answer
        text, model_name = self.generator.generate(question, contexts)
        citations = self._build_citations(text, contexts)
        answer = Answer(
            question=question,
            answer=text,
            citations=citations,
            model=model_name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            context_count=len(contexts),
            retrieval=self.retrieval_label(),
        )
        self._cache_put(question, answer)
        return answer

    def stream_query(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Versión en streaming de ``query()`` (Fase 6).

        Emite eventos dict en orden::

            {"type": "meta",  "cached": bool, "model", "retrieval",
             "context_count", "sources": [...]}     # 1 vez
            {"type": "token", "value": "..."}       # N veces
            {"type": "done",  "answer": {…Answer…}} # 1 vez, con citas

        Con acierto de caché semántica los tokens se re-emiten desde la
        respuesta guardada (``meta.cached=true``).
        """
        started = time.perf_counter()
        use_cache = k is None and where is None
        if use_cache and self.cache is not None:
            payload = self.cache.lookup(question)
            if payload is not None:
                answer = Answer(**{**payload, "cached": True})
                answer.latency_ms = round((time.perf_counter() - started) * 1000, 2)
                yield {
                    "type": "meta",
                    "cached": True,
                    "model": answer.model,
                    "retrieval": answer.retrieval,
                    "context_count": answer.context_count,
                    "sources": [c.model_dump() for c in answer.citations],
                }
                yield from ({"type": "token", "value": piece} for piece in tokenize_stream(answer.answer))
                yield {"type": "done", "answer": answer.model_dump()}
                return
        contexts = self.retriever.retrieve(question, k=k, where=where)
        yield {
            "type": "meta",
            "cached": False,
            "model": self.generator.model_label,
            "retrieval": self.retrieval_label(),
            "context_count": len(contexts),
            "sources": [_source_preview(rc) for rc in contexts],
        }
        if not contexts:
            yield {"type": "done", "answer": self._empty_answer(question, started).model_dump()}
            return
        parts: list[str] = []
        for piece in self.generator.stream_generate(question, contexts):
            parts.append(piece)
            yield {"type": "token", "value": piece}
        text = "".join(parts)
        answer = Answer(
            question=question,
            answer=text,
            citations=self._build_citations(text, contexts),
            model=self.generator.model_label,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            context_count=len(contexts),
            retrieval=self.retrieval_label(),
        )
        if use_cache:
            self._cache_put(question, answer)
        yield {"type": "done", "answer": answer.model_dump()}

    def retrieval_label(self) -> str:
        """Etiqueta del modo de búsqueda activo, p. ej. ``hybrid+cross-encoder``."""
        label = self.settings.retrieval
        if self.retriever.reranker.name != "noop":
            label = f"{label}+{self.retriever.reranker.name}"
        return label

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
            "retrieval": self.retrieval_label(),
            "candidate_k": self.settings.candidate_k,
            "documents": len(documents),
            "chunks": self.store.count_chunks(),
            "cache": self.cache.stats() if self.cache else {"enabled": False},
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
        if removed:
            self.retriever.invalidate_index()
            if self.cache is not None:
                self.cache.clear()
        return removed > 0

    def clear(self) -> None:
        self.store.clear()
        self.retriever.invalidate_index()
        if self.cache is not None:
            self.cache.clear()


def _version() -> str:
    from universal_rag import __version__

    return __version__


def _source_preview(retrieved: RetrievedChunk) -> dict[str, Any]:
    """Resumen ligero de una fuente recuperada para el evento ``meta`` del stream."""
    meta = retrieved.chunk.metadata
    start_time = meta.get("start_time")
    end_time = meta.get("end_time")
    timestamp = None
    if start_time is not None or end_time is not None:
        from universal_rag.core.transcript import format_range

        timestamp = format_range(start_time, end_time)
    return {
        "title": meta.get("title"),
        "source": meta.get("source", "?"),
        "uri": meta.get("uri", ""),
        "score": retrieved.score,
        "timestamp": timestamp,
    }
