"""Conectores: contrato universal + registro con auto-descubrimiento.

Patrón "universal" (igual que Universal MCP):
    connectors/<nombre>/connector.py  ->  clase que hereda de BaseConnector

Agregar una fuente nueva = soltar un paquete con ``connector.py``; el registro
lo descubre solo. Un conector roto queda aislado (se reporta y no derriba el
sistema). También se pueden cargar conectores externos con
``RAG_CONNECTOR_DIRS=/ruta/a/conectores``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from universal_rag.core.models import Document
from universal_rag.core.utils import get_logger, make_id, normalize_whitespace

logger = get_logger("connectors")

C = TypeVar("C", bound="BaseConnector")

BUILTIN_PACKAGE = "universal_rag.connectors"
CONNECTOR_MODULE = "connector"


class ConnectorError(Exception):
    """Error genérico de conector."""


class ConnectorNotFoundError(ConnectorError):
    """Ningún conector registrado sabe manejar la fuente dada."""


class BaseConnector(ABC):
    """Contrato que implementa toda fuente de conocimiento.

    Atributos de clase:
        name: identificador único ("file", "url", "text", "audio"…).
        description: descripción corta (para docs, CLI y MCP).
        schemes: esquemas URI que sabe manejar ("file", "http", "text", "drive"…).
        extensions: extensiones de archivo que sabe extraer (".pdf", ".mp3"…).
    """

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    schemes: ClassVar[tuple[str, ...]] = ()
    extensions: ClassVar[tuple[str, ...]] = ()
    # Prioridad de routing: mayor = se consulta antes. Útil cuando varios
    # conectores pueden reclamar la misma fuente (p. ej. youtube:// vs http://).
    priority: ClassVar[int] = 0

    # ------------------------------------------------------------------ API
    def can_handle(self, source: str) -> bool:
        """True si este conector sabe ingestar ``source``."""
        source = (source or "").strip()
        if not source:
            return False
        scheme = source.split("://", 1)[0].lower() if "://" in source else ""
        if scheme and scheme in self.schemes:
            return True
        lower = source.lower()
        return any(lower.endswith(ext) for ext in self.extensions)

    @abstractmethod
    def fetch(self, source: str, **options: Any) -> list[Document]:
        """Obtiene el contenido de ``source`` y lo normaliza a Document(s)."""

    # --------------------------------------------------------------- helpers
    @staticmethod
    def build_document(
        *,
        content: str,
        source: str,
        uri: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: Any = None,
    ) -> Document:
        """Helper para crear documentos con ID estable (por URI) y texto normalizado."""
        from datetime import datetime

        normalized = normalize_whitespace(content)
        if created_at is not None and not isinstance(created_at, datetime):
            created_at = None
        return Document(
            id=make_id("doc", uri),
            content=normalized,
            source=source,
            uri=uri,
            title=title,
            metadata=metadata or {},
            created_at=created_at,
        )

    def fetch_one(self, source: str, **options: Any) -> Document:
        documents = self.fetch(source, **options)
        if not documents:
            raise ConnectorError(f"El conector '{self.name}' no produjo documentos para: {source}")
        return documents[0]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"


class ConnectorRecord:
    """Entrada del registro: conector + origen (builtin/externo) + error de carga."""

    __slots__ = ("connector", "origin", "error")

    def __init__(self, connector: BaseConnector | None, origin: str, error: str | None = None) -> None:
        self.connector = connector
        self.origin = origin
        self.error = error

    @property
    def ok(self) -> bool:
        return self.connector is not None


class ConnectorRegistry:
    """Registro de conectores con auto-descubrimiento por convención."""

    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}
        self.records: list[ConnectorRecord] = []

    # ------------------------------------------------------------------ API
    def register(self, connector: BaseConnector, origin: str = "manual") -> None:
        if connector.name in self._connectors:
            logger.warning("Conector duplicado '%s' (%s) — se mantiene el primero", connector.name, origin)
            return
        self._connectors[connector.name] = connector
        self.records.append(ConnectorRecord(connector, origin))

    def get(self, name: str) -> BaseConnector:
        try:
            return self._connectors[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._connectors)) or "(vacío)"
            raise ConnectorNotFoundError(
                f"No existe el conector '{name}'. Disponibles: {known}"
            ) from exc

    def resolve(self, source: str) -> BaseConnector:
        """Elige el conector de mayor prioridad que sepa manejar ``source``."""
        candidates = sorted(self._connectors.values(), key=lambda c: -c.priority)
        for connector in candidates:
            try:
                if connector.can_handle(source):
                    return connector
            except Exception:  # noqa: BLE001 — un conector roto no decide el routing
                logger.exception("can_handle() falló en el conector '%s'", connector.name)
        raise ConnectorNotFoundError(
            f"Ningún conector puede manejar la fuente: {source!r}. "
            f"Disponibles: {', '.join(sorted(self._connectors))}"
        )

    def names(self) -> list[str]:
        return sorted(self._connectors)

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "schemes": ", ".join(c.schemes),
                "extensions": ", ".join(c.extensions),
                "origin": next(r.origin for r in self.records if r.connector is c),
            }
            for c in self._connectors.values()
        ]

    # ----------------------------------------------------------- discovery
    @classmethod
    def discover(cls, extra_dirs: list[str] | list[Path] | None = None) -> ConnectorRegistry:
        """Descubre builtins + conectores externos; los fallos quedan aislados."""
        registry = cls()
        registry._discover_builtins()
        for raw in extra_dirs or []:
            registry._discover_dir(Path(raw))
        return registry

    def _discover_builtins(self) -> None:
        package = importlib.import_module(BUILTIN_PACKAGE)
        for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
            if not module_info.ispkg:
                continue
            self._load_module(f"{BUILTIN_PACKAGE}.{module_info.name}.{CONNECTOR_MODULE}", origin="builtin")

    def _discover_dir(self, directory: Path) -> None:
        directory = directory.expanduser().resolve()
        if not directory.is_dir():
            logger.warning("Directorio de conectores no existe: %s", directory)
            return
        candidates = sorted(directory.glob("*/connector.py")) + sorted(
            p for p in directory.glob("connector.py") if p.is_file()
        )
        if not candidates:
            logger.warning(
                "Sin conectores en %s (convención: <dir>/<nombre>/connector.py)", directory
            )
            return
        for path in candidates:
            self._load_file(path, origin=f"externo:{path.parent.name}")

    def _load_module(self, module_name: str, origin: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — aislamiento de fallos por conector
            logger.error("No se pudo importar %s: %s", module_name, exc)
            self.records.append(ConnectorRecord(None, origin, error=str(exc)))
            return
        self._register_from_module(module, origin)

    def _load_file(self, path: Path, origin: str) -> None:
        module_name = f"universal_rag_ext_{path.parent.name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:  # pragma: no cover
                raise ImportError(f"spec inválida para {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 — aislamiento de fallos por conector
            logger.error("No se pudo cargar %s: %s", path, exc)
            self.records.append(ConnectorRecord(None, origin, error=str(exc)))
            return
        self._register_from_module(module, origin)

    def _register_from_module(self, module: Any, origin: str) -> None:
        try:
            found = [
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, BaseConnector) and obj is not BaseConnector and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
            ]
        except Exception as exc:  # noqa: BLE001
            self.records.append(ConnectorRecord(None, origin, error=str(exc)))
            return
        if not found:
            self.records.append(
                ConnectorRecord(None, origin, error="sin clase que herede de BaseConnector")
            )
            return
        for connector_cls in found:
            try:
                self.register(connector_cls(), origin)
            except Exception as exc:  # noqa: BLE001
                logger.error("No se pudo instanciar el conector %s: %s", connector_cls.__name__, exc)
                self.records.append(ConnectorRecord(None, origin, error=str(exc)))
