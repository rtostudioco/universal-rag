"""Sincronización programada de fuentes (Fase 7).

Registra fuentes con un intervalo y las re-ingesta automáticamente cuando
vencen: mantiene la base de conocimiento fresca sin intervención manual.

* ``SyncScheduler`` — registro de jobs + hilo de fondo con granularidad
  ``tick``; cada ejecución llama a ``pipeline.ingest(source)``.
* La ingesta incremental (Fase 4) hace que las re-ingestas sin cambios sean
  baratas: se omite la re-vectorización.
* Persistencia opcional a JSON (``state_path``): los jobs sobreviven al
  reinicio del proceso. Las ejecuciones en sí no se persisten.
* Thread-safe: el hilo de fondo y la API comparten el registro con lock.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from universal_rag.core.models import IngestResult
from universal_rag.core.utils import get_logger, make_id, utcnow

logger = get_logger("sync")

MIN_INTERVAL = 1.0  # segundos; evita bucles de fuego


class SyncPipeline(Protocol):
    """Superficie mínima del pipeline que necesita el planificador."""

    def ingest(self, source: str, connector: str | None = None, **options: Any) -> IngestResult: ...


class SyncJob(BaseModel):
    """Una fuente registrada para re-ingesta periódica."""

    id: str
    source: str
    interval: float  # segundos entre ejecuciones
    connector: str | None = None  # fuerza conector (opcional)
    options: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    last_run: datetime | None = None
    next_run: datetime | None = None
    last_status: str | None = None  # "ok" | "error" | "running"
    last_error: str | None = None
    last_documents: int = 0
    last_chunks: int = 0
    runs: int = 0


class SyncScheduler:
    """Planificador de re-ingestas periódicas (hilo de fondo opcional).

    ``pipeline`` puede ser un :class:`RAGPipeline` directo o una fábrica
    ``() -> RAGPipeline`` (resuelta en cada ejecución; útil para no instanciar
    la colección por defecto hasta que hace falta).

    Uso::

        scheduler = SyncScheduler(pipeline, tick=5.0)
        scheduler.register("https://ejemplo.com/docs", interval=600)
        scheduler.start()          # hilo de fondo (opcional)
        scheduler.run_now(job.id)  # ejecución inmediata (API / CLI)
        scheduler.stop()
    """

    def __init__(
        self,
        pipeline: SyncPipeline | Callable[[], SyncPipeline],
        *,
        tick: float = 5.0,
        state_path: str | Path | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.tick_seconds = max(0.5, float(tick))
        self.state_path = Path(state_path) if state_path else None
        self._jobs: OrderedDict[str, SyncJob] = OrderedDict()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.executions = 0
        self.failures = 0
        if self.state_path is not None:
            self._load()

    # ------------------------------------------------------------------ API
    def register(
        self,
        source: str,
        interval: float,
        connector: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> SyncJob:
        """Registra (o actualiza) la sincronización de una fuente.

        El ID es estable por (fuente, conector): re-registrar actualiza el
        intervalo/opciones sin duplicar. La primera ejecución es inmediata
        (``next_run = ahora``); después cada ``interval`` segundos.
        """
        source = (source or "").strip()
        if not source:
            raise ValueError("La fuente no puede estar vacía")
        interval = float(interval)
        if interval < MIN_INTERVAL:
            raise ValueError(f"El intervalo debe ser >= {MIN_INTERVAL} segundos")
        job_id = make_id("sync", source, connector or "")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = SyncJob(id=job_id, source=source, interval=interval, connector=connector)
                job.next_run = utcnow()
            job.interval = interval
            job.connector = connector
            job.options = dict(options or {})
            self._jobs[job_id] = job
            self._jobs.move_to_end(job_id)
            snapshot = job.model_copy(deep=True)
        self._save()
        logger.info("Sincronización registrada '%s' cada %.0fs (%s)", source, interval, job_id)
        return snapshot

    def remove(self, job_id: str) -> bool:
        """Elimina un job de sincronización; True si existía."""
        with self._lock:
            removed = self._jobs.pop(job_id, None) is not None
        if removed:
            self._save()
        return removed

    def get(self, job_id: str) -> SyncJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def list(self) -> list[SyncJob]:
        with self._lock:
            return [job.model_copy(deep=True) for job in self._jobs.values()]

    def run_now(self, job_id: str) -> dict[str, Any]:
        """Ejecuta un job inmediatamente (bloquea hasta terminar la ingesta).

        Devuelve ``{"job": …, "result": …}`` o lanza ``KeyError`` si no existe.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job de sincronización no encontrado: {job_id}")
        return self._execute(job)

    def tick(self) -> list[dict[str, Any]]:
        """Ejecuta los jobs vencidos; devuelve los resultados generados."""
        now = utcnow()
        with self._lock:
            due = [
                job
                for job in self._jobs.values()
                if job.next_run is not None and job.next_run <= now and job.last_status != "running"
            ]
        return [self._execute(job) for job in due]

    def start(self) -> None:
        """Arranca el hilo de fondo (idempotente)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="rag-sync", daemon=True)
            self._thread.start()
        logger.info("Planificador de sincronización iniciado (tick=%.1fs)", self.tick)

    def stop(self) -> None:
        """Detiene el hilo de fondo (idempotente)."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.tick_seconds * 2))
        self._thread = None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "jobs": len(self._jobs),
                "executions": self.executions,
                "failures": self.failures,
                "running": running,
                "tick": self.tick_seconds,
                "state_path": str(self.state_path) if self.state_path else None,
            }

    # -------------------------------------------------------------- internals
    def _loop(self) -> None:
        while not self._stop_event.wait(self.tick_seconds):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — el hilo nunca muere por un job roto
                logger.exception("Tick del planificador de sincronización falló")

    def _resolve_pipeline(self) -> SyncPipeline:
        return self.pipeline() if callable(self.pipeline) else self.pipeline

    def _execute(self, job: SyncJob) -> dict[str, Any]:
        """Corre la ingesta de un job y actualiza su estado (sin lock durante la E/S)."""
        job.last_status = "running"
        result: IngestResult | None = None
        error: str | None = None
        try:
            pipeline = self._resolve_pipeline()
            result = pipeline.ingest(job.source, connector=job.connector, **dict(job.options))
            if result.errors and result.documents == 0:
                error = "; ".join(result.errors)
        except Exception as exc:  # noqa: BLE001 — un fallo de red no derriba el planificador
            error = f"{type(exc).__name__}: {exc}"
        now = utcnow()
        with self._lock:
            job.runs += 1
            job.last_run = now
            job.next_run = now + timedelta(seconds=job.interval)
            job.last_status = "error" if error else "ok"
            job.last_error = error
            if result is not None:
                job.last_documents = result.documents
                job.last_chunks = result.chunks
            self.executions += 1
            if error:
                self.failures += 1
                logger.warning("Sincronización falló '%s': %s", job.source, error)
        self._save()
        return {
            "job": job.model_dump(mode="json"),
            "result": result.model_dump() if result is not None else None,
        }

    # ---------------------------------------------------------- persistencia
    def _save(self) -> None:
        """Serializa los jobs a JSON (mejor esfuerzo; nunca rompe el flujo)."""
        if self.state_path is None:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "version": 1,
                    "jobs": [job.model_dump(mode="json") for job in self._jobs.values()],
                }
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            logger.warning("No se pudo persistir el estado de sincronización en %s", self.state_path, exc_info=True)

    def _load(self) -> None:
        """Carga los jobs persistidos; las ejecuciones pendientes quedan para el próximo tick."""
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            with self._lock:
                for raw in payload.get("jobs", []):
                    try:
                        job = SyncJob(**raw)
                    except Exception:  # noqa: BLE001 — entrada corrupta: se descarta
                        logger.warning("Job de sincronización corrupto descartado: %r", raw)
                        continue
                    job.last_status = None if job.last_status == "running" else job.last_status
                    job.next_run = utcnow()  # vencidos: se ejecutan al primer tick
                    self._jobs[job.id] = job
            logger.info("Sincronizaciones restauradas desde %s: %d", self.state_path, len(self._jobs))
        except (OSError, json.JSONDecodeError):
            logger.warning("Estado de sincronización ilegible en %s", self.state_path, exc_info=True)
