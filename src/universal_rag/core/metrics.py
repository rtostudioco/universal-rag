"""Métricas de runtime en memoria (Fase 4).

Contadores simples por proceso: peticiones, consultas, ingestas, errores y
uptime. Se exponen en ``GET /stats`` bajo la clave ``runtime``. No reemplazan
un sistema de observabilidad (Prometheus/OTel); dan visibilidad básica sin
dependencias extra.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from universal_rag.core.utils import utcnow


class Metrics:
    """Contadores thread-safe de la vida del proceso/API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at: datetime = utcnow()
        self._counters: dict[str, int] = {}

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def snapshot(self) -> dict[str, Any]:
        """Estado actual: uptime + contadores (copia)."""
        with self._lock:
            counters = dict(self._counters)
        uptime = round((utcnow() - self.started_at).total_seconds(), 1)
        return {
            "uptime_seconds": uptime,
            "requests": counters.get("requests", 0),
            "queries": counters.get("queries", 0),
            "ingests": counters.get("ingests", 0),
            "ingests_async": counters.get("ingests_async", 0),
            "errors": counters.get("errors", 0),
            "documents_deleted": counters.get("documents_deleted", 0),
        }
