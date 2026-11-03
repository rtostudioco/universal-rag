"""API REST de Universal RAG (FastAPI).

Endpoints:
    GET    /                    mini chat UI (HTML)
    GET    /health              estado del servicio
    GET    /stats               estadísticas del pipeline
    POST   /ingest              ingesta por fuente (resuelve conector automático)
    POST   /ingest/text         ingesta de texto directo
    POST   /query               pregunta con respuesta + citas
    GET    /documents           lista de documentos ingestado
    DELETE /documents/{id}      elimina un documento y sus fragmentos

Uso::

    uvicorn universal_rag.api.app:app --factory --port 8100
    # o: universal-rag serve --port 8100
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from universal_rag import __version__
from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class IngestRequest(BaseModel):
    source: str = Field(..., description="Ruta, URL, globo o fuente que resuelve el conector")
    connector: str | None = Field(None, description="Nombre del conector (opcional; auto si se omite)")
    options: dict[str, Any] = Field(default_factory=dict, description="Opciones específicas del conector")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Contenido de texto a ingestar")
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(None, ge=1, le=50, description="Fragmentos a recuperar (default: RAG_TOP_K)")
    where: dict[str, Any] | None = Field(None, description="Filtro de metadatos (ej. {\"source\": \"file\"})")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Fábrica de la aplicación FastAPI con su pipeline."""
    app = FastAPI(
        title="Universal RAG",
        version=__version__,
        description=(
            "Pipeline plugable de Retrieval-Augmented Generation: ingesta universal "
            "(texto, archivos, URLs, audio, video, YouTube, Drive), búsqueda vectorial "
            "o híbrida (BM25 + reranker) y respuestas con citas verificables."
        ),
    )
    app.state.settings = settings or Settings()
    app.state.pipeline = RAGPipeline(settings=app.state.settings)

    # ------------------------------------------------------------------ basic
    @app.get("/", include_in_schema=False)
    def chat_ui() -> HTMLResponse:
        """Mini chat UI (HTML autocontenido, sin CDNs)."""
        return HTMLResponse((_STATIC_DIR / "chat.html").read_text(encoding="utf-8"))

    @app.get("/health", tags=["system"], summary="Estado del servicio")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/stats", tags=["system"], summary="Estadísticas del pipeline")
    def stats() -> dict[str, Any]:
        return app.state.pipeline.stats()

    # ----------------------------------------------------------------- ingest
    @app.post("/ingest", tags=["ingest"], summary="Ingesta una fuente (conector automático)")
    def ingest(request: IngestRequest) -> dict[str, Any]:
        result = app.state.pipeline.ingest(request.source, connector=request.connector, **request.options)
        if result.errors and result.documents == 0:
            raise HTTPException(status_code=422, detail=result.errors)
        return result.model_dump()

    @app.post("/ingest/text", tags=["ingest"], summary="Ingesta de texto directo")
    def ingest_text(request: IngestTextRequest) -> dict[str, Any]:
        result = app.state.pipeline.ingest_text(request.text, title=request.title, metadata=request.metadata)
        if result.errors and result.documents == 0:
            raise HTTPException(status_code=422, detail=result.errors)
        return result.model_dump()

    # ------------------------------------------------------------------ query
    @app.post("/query", tags=["query"], summary="Pregunta con respuesta y citas")
    def query(request: QueryRequest) -> dict[str, Any]:
        return app.state.pipeline.query(request.question, k=request.k, where=request.where).model_dump()

    # -------------------------------------------------------------- documents
    @app.get("/documents", tags=["documents"], summary="Documentos ingestado")
    def documents() -> list[dict[str, Any]]:
        return app.state.pipeline.documents()

    @app.delete("/documents/{document_id}", tags=["documents"], summary="Elimina un documento")
    def delete_document(document_id: str) -> dict[str, Any]:
        deleted = app.state.pipeline.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Documento no encontrado: {document_id}")
        return {"deleted": document_id}

    return app


app = create_app()
