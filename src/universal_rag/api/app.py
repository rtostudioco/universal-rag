"""API REST de Universal RAG (FastAPI).

Endpoints:
    GET    /                        workbench web (chat streaming + gestión)
    GET    /chat                    mini chat clásico (HTML)
    GET    /health                  estado del servicio (abierto)
    GET    /stats                   estadísticas + métricas runtime
    POST   /ingest                  ingesta por fuente (conector automático)
    POST   /ingest/text             ingesta de texto directo
    POST   /ingest/async            ingesta en background (job) + polling
    GET    /jobs                    lista de jobs
    GET    /jobs/{id}               estado de un job
    POST   /query                   pregunta con respuesta + citas
    POST   /query/stream            pregunta con respuesta en streaming (SSE)
    POST   /query/agent             pregunta multi-hop con plan y pasos (Fase 8)
    POST   /query/agent/stream      agente multi-hop en streaming (SSE, Fase 8)
    GET    /conversations           conversaciones vivas (Fase 7)
    GET    /conversations/{id}      historial de una conversación
    DELETE /conversations/{id}      borra una conversación
    DELETE /conversations           borra todas las conversaciones
    GET    /sync                    jobs de sincronización programada (Fase 7)
    POST   /sync                    registra una fuente con intervalo
    DELETE /sync/{id}               elimina un job de sincronización
    POST   /sync/{id}/run           fuerza una ejecución inmediata
    GET    /documents               lista de documentos ingestado
    DELETE /documents/{id}          elimina un documento y sus fragmentos
    GET    /collections             colecciones activas
    DELETE /collections/{name}      vacía y descarta una colección

Fase 4: colecciones múltiples (parámetro ``collection``), jobs asíncronos,
autenticación opcional por API key (``RAG_API_KEY``) y métricas runtime.
Fase 6: streaming SSE (``/query/stream``), caché semántica y workbench web.
Fase 7: memoria conversacional multi-turno (``conversation_id``) y
sincronización programada de fuentes (``/sync``).
Fase 8: agente multi-hop (``/query/agent`` y ``/query/agent/stream``) con
plan de sub-consultas, pasos trazables y síntesis sobre contextos fusionados.

Uso::

    uvicorn universal_rag.api.app:app --port 8100
    # o: universal-rag serve --port 8100
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from universal_rag import __version__
from universal_rag.api.auth import create_auth_dependency
from universal_rag.core.collections import CollectionManager, normalize_name
from universal_rag.core.config import Settings
from universal_rag.core.jobs import JobInfo, JobManager, JobStatus
from universal_rag.core.metrics import Metrics
from universal_rag.core.pipeline import RAGPipeline
from universal_rag.core.sync import SyncScheduler
from universal_rag.core.utils import get_logger

_STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = get_logger("api")


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
    conversation_id: str | None = Field(
        None,
        max_length=128,
        description="ID de conversación multi-turno (Fase 7): inyecta el historial y registra el turno",
    )


class QueryAgentRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(None, ge=1, le=50, description="Contextos fusionados finales (default: RAG_TOP_K)")
    where: dict[str, Any] | None = Field(None, description="Filtro de metadatos aplicado a cada sub-consulta")
    collection: str | None = Field(None, description="Colección donde consultar (default si se omite)")
    max_steps: int | None = Field(
        None, ge=1, le=10, description="Máximo de sub-consultas (default: RAG_AGENT_MAX_STEPS)"
    )
    planning: Literal["auto", "heuristic", "llm"] | None = Field(
        None, description="Modo de planificación (default: RAG_AGENT_PLANNING)"
    )


class SyncRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Fuente a re-ingestar (resuelta por conector)")
    interval: float = Field(300, ge=1, description="Segundos entre re-ingestas")
    connector: str | None = Field(None, description="Conector fijo (opcional; auto si se omite)")
    options: dict[str, Any] = Field(default_factory=dict, description="Opciones del conector")


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
    app.state.sync = SyncScheduler(
        lambda: app.state.collections.get(),  # fábrica perezosa: no instancia la colección default
        tick=settings.sync_tick,
        state_path=settings.sync_state_path,
    )

    auth = create_auth_dependency(settings)
    protected = APIRouter(dependencies=[Depends(auth)])

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.sync_enabled:
            app.state.sync.start()
        yield
        app.state.sync.stop()
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
    def workbench_ui() -> HTMLResponse:
        """Workbench web (Fase 6): chat en streaming + documentos + ingesta + estado."""
        return HTMLResponse((_STATIC_DIR / "workbench.html").read_text(encoding="utf-8"))

    @app.get("/chat", include_in_schema=False)
    def chat_ui() -> HTMLResponse:
        """Mini chat clásico (HTML autocontenido, sin CDNs)."""
        return HTMLResponse((_STATIC_DIR / "chat.html").read_text(encoding="utf-8"))

    @app.get("/health", tags=["system"], summary="Estado del servicio")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # ------------------------------------------------------------------ stats
    @protected.get("/stats", tags=["system"], summary="Estadísticas y métricas")
    def stats() -> dict[str, Any]:
        pipeline = app.state.collections.get()
        data: dict[str, Any] = pipeline.stats()
        data["collections"] = app.state.collections.list_collections()
        data["runtime"] = app.state.metrics.snapshot()
        data["sync"] = app.state.sync.stats()
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
            request.question,
            k=request.k,
            where=request.where,
            conversation_id=request.conversation_id,
        )
        return answer.model_dump()

    @protected.post(
        "/query/stream",
        tags=["query"],
        summary="Pregunta con respuesta en streaming (SSE)",
    )
    def query_stream(request: QueryRequest) -> StreamingResponse:
        """Server-Sent Events: ``meta`` (fuentes) → ``token``* → ``done`` (Answer).

        Consumible con ``fetch`` + ReadableStream o ``EventSource``-compatible.
        Con caché semántica activa, ``meta.cached=true`` indica respuesta servida
        de caché.
        """
        app.state.metrics.inc("queries_stream")
        pipeline = _pipeline(request.collection)

        def event_source():
            try:
                for event in pipeline.stream_query(
                    request.question,
                    k=request.k,
                    where=request.where,
                    conversation_id=request.conversation_id,
                ):
                    etype = event["type"]
                    if etype == "token":
                        data: Any = event.get("value", "")
                    elif etype == "done":
                        data = event.get("answer")
                    else:
                        data = {key: value for key, value in event.items() if key != "type"}
                    payload = json.dumps(data, ensure_ascii=False, default=str)
                    yield f"event: {etype}\ndata: {payload}\n\n"
            except Exception as exc:  # noqa: BLE001 — el error viaja por el stream
                logger.exception("Error en /query/stream")
                error_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {error_payload}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # desactiva buffering en proxies (nginx)
            },
        )

    @protected.post(
        "/query/agent",
        tags=["query"],
        summary="Pregunta multi-hop con plan y pasos trazables (agente)",
    )
    def query_agent(request: QueryAgentRequest) -> dict[str, Any]:
        """Agente RAG (Fase 8): descompone la pregunta en sub-consultas, busca,
        fusiona contextos y sintetiza una respuesta con citas. Devuelve el plan,
        los pasos ejecutados y las citas sobre el ranking fusionado."""
        app.state.metrics.inc("queries_agent")
        pipeline = _pipeline(request.collection)
        agent = pipeline.agent
        if request.max_steps is not None:
            agent.max_steps = request.max_steps
        if request.planning is not None:
            agent.planning = request.planning
        result = agent.ask(request.question, k=request.k, where=request.where)
        return result.model_dump()

    @protected.post(
        "/query/agent/stream",
        tags=["query"],
        summary="Pregunta multi-hop en streaming (SSE): plan → pasos → tokens",
    )
    def query_agent_stream(request: QueryAgentRequest) -> StreamingResponse:
        """Server-Sent Events del agente (Fase 8): ``plan`` (sub-consultas y
        planificador) → ``step``* (búsquedas con resultados) → ``synthesize`` →
        ``token``* → ``done`` (AgentAnswer completo)."""
        app.state.metrics.inc("queries_agent_stream")
        pipeline = _pipeline(request.collection)
        agent = pipeline.agent
        if request.max_steps is not None:
            agent.max_steps = request.max_steps
        if request.planning is not None:
            agent.planning = request.planning

        def event_source():
            try:
                for event in agent.stream_ask(request.question, k=request.k, where=request.where):
                    etype = event["type"]
                    if etype == "token":
                        data: Any = event.get("value", "")
                    elif etype == "done":
                        data = event.get("answer")
                    else:
                        data = {key: value for key, value in event.items() if key != "type"}
                    payload = json.dumps(data, ensure_ascii=False, default=str)
                    yield f"event: {etype}\ndata: {payload}\n\n"
            except Exception as exc:  # noqa: BLE001 — el error viaja por el stream
                logger.exception("Error en /query/agent/stream")
                error_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                yield f"event: error\ndata: {error_payload}\n\n"

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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

    # ----------------------------------------------------------- conversations
    def _memory() -> Any:
        return app.state.collections.get().memory

    @protected.get("/conversations", tags=["conversations"], summary="Conversaciones vivas")
    def conversations() -> list[dict[str, Any]]:
        return _memory().conversations()

    @protected.get(
        "/conversations/{conversation_id}",
        tags=["conversations"],
        summary="Historial de una conversación",
    )
    def conversation_history(conversation_id: str) -> dict[str, Any]:
        memory = _memory()
        if memory.turns(conversation_id) == 0:
            raise HTTPException(status_code=404, detail=f"Conversación no encontrada: {conversation_id}")
        return {
            "id": conversation_id,
            "turns": memory.turns(conversation_id),
            "messages": memory.history(conversation_id),
        }

    @protected.delete(
        "/conversations/{conversation_id}",
        tags=["conversations"],
        summary="Borra una conversación",
    )
    def delete_conversation(conversation_id: str) -> dict[str, Any]:
        cleared = _memory().clear(conversation_id)
        if not cleared:
            raise HTTPException(status_code=404, detail=f"Conversación no encontrada: {conversation_id}")
        return {"cleared": conversation_id}

    @protected.delete("/conversations", tags=["conversations"], summary="Borra todas las conversaciones")
    def delete_conversations() -> dict[str, Any]:
        return {"cleared": _memory().clear()}

    # -------------------------------------------------------------------- sync
    @protected.get("/sync", tags=["sync"], summary="Jobs de sincronización programada")
    def sync_list() -> dict[str, Any]:
        scheduler: SyncScheduler = app.state.sync
        return {
            "enabled": settings.sync_enabled,
            **scheduler.stats(),
            "jobs": [job.model_dump(mode="json") for job in scheduler.list()],
        }

    @protected.post("/sync", tags=["sync"], summary="Registra una fuente con intervalo")
    def sync_register(request: SyncRequest) -> dict[str, Any]:
        app.state.metrics.inc("sync_registered")
        try:
            job = app.state.sync.register(
                request.source,
                request.interval,
                connector=request.connector,
                options=request.options,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return job.model_dump(mode="json")

    @protected.delete("/sync/{job_id}", tags=["sync"], summary="Elimina un job de sincronización")
    def sync_remove(job_id: str) -> dict[str, Any]:
        if not app.state.sync.remove(job_id):
            raise HTTPException(status_code=404, detail=f"Job de sincronización no encontrado: {job_id}")
        return {"deleted": job_id}

    @protected.post(
        "/sync/{job_id}/run", tags=["sync"], summary="Fuerza una ejecución inmediata"
    )
    def sync_run(job_id: str) -> dict[str, Any]:
        try:
            outcome = app.state.sync.run_now(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        app.state.metrics.inc("sync_runs")
        return outcome

    app.include_router(protected)
    return app


app = create_app()
