"""API REST de Universal RAG (FastAPI).

Endpoints:
    GET    /                        mini chat UI (HTML)
    GET    /health                  estado del servicio (abierto)
    GET    /stats                   estadísticas + métricas runtime
    POST   /ingest                  ingesta por fuente (conector automático)
    POST   /ingest/text             ingesta de texto directo
    POST   /ingest/async            ingesta en background (job) + polling
    GET    /jobs                    lista de jobs
    GET    /jobs/{id}               estado de un job
    POST   /query                   pregunta con respuesta + citas
    GET    /documents               lista de documentos ingestado
    DELETE /documents/{id}          elimina un documento y sus fragmentos
    GET    /collections             colecciones activas
    DELETE /collections/{name}      vacía y descarta una colección

Fase 4: colecciones múltiples (parámetro ``collection``), jobs asíncronos,
autenticación opcional por API key (``RAG_API_KEY``) y métricas runtime.

Uso::

    uvicorn universal_rag.api.app:app --factory --port 8100
    # o: universal-rag serve --port 8100
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from universal_rag import __version__
from universal_rag.api.auth import create_auth_dependency
from universal_rag.core.collections import CollectionManager, normalize_name
from universal_rag.core.config import Settings
from universal_rag.core.jobs import JobInfo, JobManager, JobStatus
from universal_rag.core.metrics import Metrics
from universal_rag.core.pipeline import RAGPipeline

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class IngestRequest(BaseModel):
    source: str = Field(..., description="Ruta, URL, globo o fuente que resuelve el conector")
    connector: str | None = Field(None, description="Nombre del conector (opcional; auto si se omite)")
    options: dict[str, Any] = Field(default_factory=dict, description="Opciones específicas del conector")
    collection: str | None = Field(None, description="Colección destino (default si se omite)")
    force: bool = Field(False, description="Re-ingestar aunque el contenido no haya cambiado")


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Contenido de texto a ingestar")
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    collection: str | None = None
    force: bool = False


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(None, ge=1, le=50, description="Fragmentos a recuperar (default: RAG_TOP_K)")
    where: dict[str, Any] | None = Field(None, description="Filtro de metadatos (ej. {\"source\": \"file\"})")
    collection: str | None = Field(None, description="Colección donde consultar (default si se omite)")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Fábrica de la aplicación FastAPI: pipeline(s), jobs, métricas y auth."""
    settings = settings or Settings()
    app = FastAPI(
        title="Universal RAG",
        version=__version__,
        description=(
            "Pipeline plugable de Retrieval-Augmented Generation: ingesta universal "
            "(texto, archivos, URLs, audio, video, YouTube, Drive), búsqueda vectorial "
            "o híbrida (BM25 + reranker), colecciones múltiples, jobs asíncronos y "
            "respuestas con citas verificables."
        ),
    )
    app.state.settings = settings
    app.state.collections = CollectionManager(settings)
    app.state.jobs = JobManager(
        workers=settings.jobs_workers, max_history=settings.jobs_max_history
    )
    app.state.metrics = Metrics()

    auth = create_auth_dependency(settings)
    protected = APIRouter(dependencies=[Depends(auth)])

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        app.state.jobs.shutdown(wait=False)

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        metrics: Metrics = app.state.metrics
        metrics.inc("requests")
        response: Response = await call_next(request)
        if response.status_code >= 400:
            metrics.inc("errors")
        return response

    # ------------------------------------------------------------------ basic
    @app.get("/", include_in_schema=False)
    def chat_ui() -> HTMLResponse:
        """Mini chat UI (HTML autocontenido, sin CDNs)."""
        return HTMLResponse((_STATIC_DIR / "chat.html").read_text(encoding="utf-8"))

    @app.get("/health", tags=["system"], summary="Estado del servicio")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # ------------------------------------------------------------------ stats
    @protected.get("/stats", tags=["system"], summary="Estadísticas y métricas")
    def stats() -> dict[str, Any]:
        data: dict[str, Any] = app.state.collections.get().stats()
        data["collections"] = app.state.collections.list_collections()
        data["runtime"] = app.state.metrics.snapshot()
        return data

    # ----------------------------------------------------------------- ingest
    def _pipeline(collection: str | None) -> RAGPipeline:
        try:
            return app.state.collections.get(collection)
        except ValueError as exc:  # nombre de colección inválido
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @protected.post("/ingest", tags=["ingest"], summary="Ingesta una fuente (conector automático)")
    def ingest(request: IngestRequest) -> dict[str, Any]:
        app.state.metrics.inc("ingests")
        result = _pipeline(request.collection).ingest(
            request.source, connector=request.connector, force=request.force, **request.options
        )
        if result.errors and result.documents == 0:
            raise HTTPException(status_code=422, detail=result.errors)
        return result.model_dump()

    @protected.post("/ingest/text", tags=["ingest"], summary="Ingesta de texto directo")
    def ingest_text(request: IngestTextRequest) -> dict[str, Any]:
        app.state.metrics.inc("ingests")
        result = _pipeline(request.collection).ingest_text(
            request.text, title=request.title, metadata=request.metadata, force=request.force
        )
        if result.errors and result.documents == 0:
            raise HTTPException(status_code=422, detail=result.errors)
        return result.model_dump()

    @protected.post(
        "/ingest/async", tags=["ingest"], summary="Ingesta en background (job + polling)"
    )
    def ingest_async(request: IngestRequest) -> dict[str, Any]:
        """Encola la ingesta y devuelve un job_id para consultar en ``GET /jobs/{id}``."""
        app.state.metrics.inc("ingests_async")
        collection = normalize_name(request.collection, default=settings.default_collection)
        pipeline = _pipeline(collection)

        def work() -> Any:
            return pipeline.ingest(
                request.source, connector=request.connector, force=request.force, **request.options
            )

        job_id = app.state.jobs.submit(
            work, source=request.source, collection=collection, connector=request.connector
        )
        return {"job_id": job_id, "status": JobStatus.PENDING.value, "poll": f"/jobs/{job_id}"}

    # ------------------------------------------------------------------- jobs
    @protected.get("/jobs", tags=["jobs"], summary="Lista de jobs de ingesta")
    def jobs(limit: int = Query(20, ge=1, le=100)) -> list[dict[str, Any]]:
        return [job.model_dump(mode="json") for job in app.state.jobs.list_jobs(limit=limit)]

    @protected.get("/jobs/{job_id}", tags=["jobs"], summary="Estado de un job")
    def job_status(job_id: str) -> dict[str, Any]:
        job: JobInfo | None = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job no encontrado: {job_id}")
        return job.model_dump(mode="json")

    # ------------------------------------------------------------------ query
    @protected.post("/query", tags=["query"], summary="Pregunta con respuesta y citas")
    def query(request: QueryRequest) -> dict[str, Any]:
        app.state.metrics.inc("queries")
        answer = _pipeline(request.collection).query(
            request.question, k=request.k, where=request.where
        )
        return answer.model_dump()

    # -------------------------------------------------------------- documents
    @protected.get("/documents", tags=["documents"], summary="Documentos ingestado")
    def documents(collection: str | None = None) -> list[dict[str, Any]]:
        return _pipeline(collection).documents()

    @protected.delete("/documents/{document_id}", tags=["documents"], summary="Elimina un documento")
    def delete_document(document_id: str, collection: str | None = None) -> dict[str, Any]:
        deleted = _pipeline(collection).delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Documento no encontrado: {document_id}")
        app.state.metrics.inc("documents_deleted")
        return {"deleted": document_id}

    # ------------------------------------------------------------ collections
    @protected.get("/collections", tags=["collections"], summary="Colecciones activas")
    def collections() -> list[dict[str, Any]]:
        return app.state.collections.list_collections()

    @protected.delete("/collections/{name}", tags=["collections"], summary="Vacía una colección")
    def delete_collection(name: str) -> dict[str, Any]:
        try:
            deleted = app.state.collections.delete(name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Colección no encontrada: {name}")
        return {"deleted": name}

    app.include_router(protected)
    return app


app = create_app()
