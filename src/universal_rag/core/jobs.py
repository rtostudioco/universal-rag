"""Ingesta asíncrona por jobs (Fase 4).

Permite ingestar fuentes lentas (audio, video, YouTube, lotes grandes) sin
bloquear al cliente: se envía el trabajo con :meth:`JobManager.submit` y se
consulta el resultado con :meth:`JobManager.get`.

* Registro **en memoria** (un solo proceso): los jobs no sobreviven reinicios.
* Pool de hilos propio (``RAG_JOBS_WORKERS``, default 2) para no bloquear el
  event loop de FastAPI.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from universal_rag.core.models import IngestResult
from universal_rag.core.utils import get_logger, utcnow

logger = get_logger("jobs")


class JobStatus(str, Enum):
    """Ciclo de vida de un job de ingesta."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class JobInfo(BaseModel):
    """Estado observable de un job de ingesta asíncrona."""

    id: str
    status: JobStatus = JobStatus.PENDING
    collection: str = "default"
    source: str = ""
    connector: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: IngestResult | None = None
    error: str | None = None


class JobManager:
    """Registro de jobs con pool de hilos (thread-safe, en memoria)."""

    def __init__(self, *, workers: int = 2, max_history: int = 200) -> None:
        self._lock = threading.Lock()
        self._jobs: OrderedDict[str, JobInfo] = OrderedDict()
        self._max_history = max(10, int(max_history))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="universal-rag-ingest"
        )

    # ------------------------------------------------------------------ API
    def submit(
        self,
        fn: Callable[[], IngestResult],
        *,
        source: str,
        collection: str = "default",
        connector: str | None = None,
    ) -> str:
        """Encola ``fn`` y devuelve el id del job inmediatamente."""
        job = JobInfo(
            id=uuid.uuid4().hex[:12],
            collection=collection,
            source=source[:500],
            connector=connector,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        self._executor.submit(self._run, job.id, fn)
        logger.info("Job %s encolado (colección '%s'): %s", job.id, collection, source[:80])
        return job.id

    def get(self, job_id: str) -> JobInfo | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int | None = None) -> list[JobInfo]:
        """Jobs más recientes primero."""
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.reverse()
        return jobs[: max(0, limit)] if limit else jobs

    def wait(self, job_id: str, timeout: float = 60.0, poll: float = 0.05) -> JobInfo | None:
        """Bloquea hasta que el job termina (o expira el timeout). Útil en tests/CLI."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if job is None or job.status in (JobStatus.DONE, JobStatus.ERROR):
                return job
            time.sleep(poll)
        return self.get(job_id)

    def shutdown(self, wait: bool = False) -> None:
        """Libera el pool de hilos (llamado al apagar la app)."""
        self._executor.shutdown(wait=wait)

    # -------------------------------------------------------------- internos
    def _run(self, job_id: str, fn: Callable[[], IngestResult]) -> None:
        job = self._update(job_id, status=JobStatus.RUNNING, started_at=utcnow())
        if job is None:  # fue purgado del historial mientras esperaba el pool
            return
        try:
            result = fn()
            self._update(job_id, status=JobStatus.DONE, finished_at=utcnow(), result=result)
            logger.info("Job %s terminado: %s", job_id, result.model_dump(exclude={"document_ids"}))
        except Exception as exc:  # noqa: BLE001 — el error del job no derriba el pool
            logger.exception("Job %s falló", job_id)
            self._update(
                job_id, status=JobStatus.ERROR, finished_at=utcnow(), error=f"{type(exc).__name__}: {exc}"
            )

    def _update(self, job_id: str, **updates: Any) -> JobInfo | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in updates.items():
                setattr(job, key, value)
            return job

    def _evict_locked(self) -> None:
        """Mantiene el historial acotado: primero purga jobs terminados."""
        if len(self._jobs) <= self._max_history:
            return
        for job_id, job in list(self._jobs.items()):
            if len(self._jobs) <= self._max_history:
                break
            if job.status in (JobStatus.DONE, JobStatus.ERROR):
                del self._jobs[job_id]
        # Si aún excede (todo pending/running), purga el más antiguo.
        while len(self._jobs) > self._max_history:
            self._jobs.popitem(last=False)
