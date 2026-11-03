"""Conector ``notion``: páginas y bases de datos de Notion (API REST oficial).

Fase 5. URIs soportadas::

    notion://<page-id>            notion://page/<page-id>
    notion://database/<db-id>
    notion://search/<texto>       (búsqueda → primer resultado)
    https://www.notion.so/…/<slug>-<32-hex>   (URL pública de una página)

Autenticación: token de integración vía opción ``token=`` por ingesta o
variable de entorno ``RAG_NOTION_TOKEN``. Recuerda compartir las páginas con
la integración en Notion ("Connections").

Opciones de ingesta:
    token: token de integración (si no, ``RAG_NOTION_TOKEN``).
    limit: máximo de filas al ingestar una base de datos (default 100).
    filter: filtro JSON de la query de base de datos (se pasa tal cual).
    throttle: pausa entre peticiones en segundos (default 0.35; 0 en tests).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any, ClassVar

import httpx

from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.models import Document
from universal_rag.core.tables import table_to_markdown
from universal_rag.core.utils import get_logger

logger = get_logger("connectors.notion")

API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
PAGE_SIZE = 100
DEFAULT_THROTTLE = 0.35  # ~3 req/s: límite práctico de la API de Notion


class NotionConnector(BaseConnector):
    """Ingesta de páginas y bases de datos de Notion vía API REST oficial."""

    name: ClassVar[str] = "notion"
    description: ClassVar[str] = (
        "Ingesta de Notion: páginas, bases de datos y búsquedas "
        "(notion://<id> o URL pública; requiere RAG_NOTION_TOKEN)."
    )
    schemes: ClassVar[tuple[str, ...]] = ("notion",)
    priority: ClassVar[int] = 10  # gana al conector `url` para URLs de notion.so

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client  # inyectable (tests / reutilización de sesión)

    # ------------------------------------------------------------------ API
    def can_handle(self, source: str) -> bool:
        lower = (source or "").strip().lower()
        return lower.startswith("notion://") or ("notion.so" in lower and lower.startswith("http"))

    def fetch(self, source: str, **options: Any) -> list[Document]:
        token = options.get("token") or os.getenv("RAG_NOTION_TOKEN")
        if not token:
            raise ConnectorError(
                "Falta el token de Notion: pasa token=… en la ingesta o define RAG_NOTION_TOKEN."
            )
        throttle = max(0.0, float(options.get("throttle", DEFAULT_THROTTLE)))
        limit = max(1, int(options.get("limit", 100)))
        client = self._client or httpx.Client(timeout=30.0)
        close = self._client is None
        try:
            kind, object_id = self._resolve(client, token, source, options)
            if kind == "database":
                return self._database_documents(client, token, object_id, limit, throttle, options)
            return [self._page_document(client, token, object_id, throttle)]
        finally:
            if close:
                client.close()

    # -------------------------------------------------------------- routing
    def _resolve(
        self, client: httpx.Client, token: str, source: str, options: dict[str, Any]
    ) -> tuple[str, str]:
        """Determina (tipo, id): page | database | search → primer resultado."""
        kind = options.get("kind")  # page | database | search | None (auto)
        raw = source.strip()
        if raw.lower().startswith("notion://"):
            rest = raw[len("notion://") :].strip("/")
            head, _, tail = rest.partition("/")
            if head.lower() in ("page", "database", "search") and tail:
                kind = kind or head.lower()
                object_id = tail
            else:
                object_id = rest
            if kind == "search":
                return self._search_first(client, token, tail if head.lower() == "search" else rest)
            object_id = self._normalize_id(object_id)
            return (kind, object_id) if kind in ("page", "database") else ("page", object_id)

        # URL pública https://www.notion.so/...
        object_id = self._id_from_url(raw)
        if not object_id:
            raise ConnectorError(f"No se pudo extraer un ID de Notion de: {source}")
        return "page", object_id

    def _search_first(self, client: httpx.Client, token: str, query: str) -> tuple[str, str]:
        data = self._request(client, token, "POST", "/search", json={"query": query, "page_size": 1})
        results = data.get("results") or []
        if not results:
            raise ConnectorError(f"La búsqueda de Notion no devolvió resultados: {query!r}")
        first = results[0]
        object_id = str(first.get("id", ""))
        if not object_id:
            raise ConnectorError("Resultado de búsqueda sin ID.")
        kind = "database" if first.get("object") == "database" else "page"
        return kind, self._normalize_id(object_id)

    # ------------------------------------------------------------- documents
    def _page_document(
        self, client: httpx.Client, token: str, object_id: str, throttle: float
    ) -> Document:
        page = self._request(client, token, "GET", f"/pages/{object_id}", error_for="página")
        title = self._page_title(page) or f"Notion {object_id[:8]}"
        content = self._blocks_text(client, token, object_id, throttle)
        return self.build_document(
            content=content or title,
            source=self.name,
            uri=f"notion://page/{object_id}",
            title=title,
            metadata={
                "notion_type": "page",
                "notion_url": page.get("url"),
                "last_edited_time": page.get("last_edited_time"),
            },
            created_at=self._parse_time(page.get("created_time")),
        )

    def _database_documents(
        self,
        client: httpx.Client,
        token: str,
        db_id: str,
        limit: int,
        throttle: float,
        options: dict[str, Any],
    ) -> list[Document]:
        database = self._request(
            client, token, "GET", f"/databases/{db_id}", error_for="base de datos"
        )
        db_title = self._rich_text(database.get("title")) or f"Base {db_id[:8]}"
        documents: list[Document] = []
        for page in self._query_database(client, token, db_id, options):
            if len(documents) >= limit:
                break
            row_id = str(page.get("id", ""))
            if not row_id:
                continue
            title = self._page_title(page) or f"Fila {row_id[:8]}"
            content = self._blocks_text(client, token, row_id, throttle)
            if not content.strip():
                logger.debug("Fila de base de datos vacía, se omite: %s", row_id)
                continue
            documents.append(
                self.build_document(
                    content=content,
                    source=self.name,
                    uri=f"notion://database/{db_id}/{row_id}",
                    title=f"{db_title} — {title}",
                    metadata={
                        "notion_type": "database_row",
                        "database": db_title,
                        "database_id": db_id,
                        "notion_url": page.get("url"),
                        "last_edited_time": page.get("last_edited_time"),
                    },
                    created_at=self._parse_time(page.get("created_time")),
                )
            )
            if throttle:
                time.sleep(throttle)
        if not documents:
            raise ConnectorError(f"La base de datos '{db_title}' no produjo documentos.")
        return documents

    def _query_database(
        self, client: httpx.Client, token: str, db_id: str, options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        body: dict[str, Any] = {"page_size": PAGE_SIZE}
        if options.get("filter"):
            body["filter"] = options["filter"]
        cursor: str | None = None
        while True:
            if cursor:
                body["start_cursor"] = cursor
            data = self._request(
                client, token, "POST", f"/databases/{db_id}/query", json=body, error_for="query"
            )
            yield from data.get("results", [])
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break

    # ---------------------------------------------------------------- blocks
    def _blocks_text(
        self, client: httpx.Client, token: str, block_id: str, throttle: float
    ) -> str:
        parts: list[str] = []
        for block in self._iter_blocks(client, token, block_id, throttle):
            rendered = self._block_text(client, token, block, throttle)
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts).strip()

    def _iter_blocks(
        self, client: httpx.Client, token: str, block_id: str, throttle: float
    ) -> Iterator[dict[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": PAGE_SIZE}
            if cursor:
                params["start_cursor"] = cursor
            data = self._request(
                client,
                token,
                "GET",
                f"/blocks/{block_id}/children",
                params=params,
                error_for="bloques",
            )
            yield from data.get("results", [])
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break
            if throttle:
                time.sleep(throttle)

    def _block_text(
        self, client: httpx.Client, token: str, block: dict[str, Any], throttle: float
    ) -> str:
        btype = block.get("type", "")
        payload = block.get(btype) or {}
        if btype == "table":
            return self._table_markdown(client, token, block.get("id", ""), throttle)
        if btype == "divider":
            return ""
        text = self._rich_text(payload.get("rich_text"))
        if btype.startswith("heading_"):
            level = {"heading_1": "#", "heading_2": "##", "heading_3": "###"}.get(btype, "##")
            return f"{level} {text}" if text else ""
        if btype == "bulleted_list_item":
            return f"- {text}" if text else ""
        if btype == "numbered_list_item":
            return f"1. {text}" if text else ""
        if btype == "to_do":
            checked = "x" if payload.get("checked") else " "
            return f"- [{checked}] {text}" if text else ""
        if btype == "code":
            language = payload.get("language") or ""
            return f"```{language}\n{text}\n```" if text else ""
        return text

    def _table_markdown(
        self, client: httpx.Client, token: str, table_block_id: str, throttle: float
    ) -> str:
        rows: list[list[str]] = []
        for row_block in self._iter_blocks(client, token, table_block_id, throttle):
            if row_block.get("type") != "table_row":
                continue
            cells = row_block.get("table_row", {}).get("cells") or []
            rows.append([self._rich_text(cell) for cell in cells])
        return table_to_markdown(rows)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _request(
        client: httpx.Client,
        token: str,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        error_for: str = "objeto",
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }
        try:
            response = client.request(
                method, f"{API_BASE}{path}", headers=headers, params=params, json=json
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise ConnectorError(
                f"La API de Notion respondió {exc.response.status_code} al leer {error_for}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Error de red llamando a Notion ({path}): {exc}") from exc

    @staticmethod
    def _rich_text(items: Any) -> str:
        if not isinstance(items, list):
            return ""
        return "".join(str(item.get("plain_text", "")) for item in items if isinstance(item, dict))

    @staticmethod
    def _page_title(page: dict[str, Any]) -> str | None:
        for prop in (page.get("properties") or {}).values():
            if prop.get("type") == "title":
                title = NotionConnector._rich_text(prop.get("title"))
                return title or None
        return None

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _normalize_id(raw: str) -> str:
        """Acepta IDs con o sin guiones; devuelve 32 hex en minúsculas."""
        candidate = re.sub(r"[^0-9a-fA-F]", "", raw)
        if len(candidate) != 32:
            raise ConnectorError(f"ID de Notion inválido: {raw!r} (se esperaban 32 hex)")
        return candidate.lower()

    @staticmethod
    def _id_from_url(url: str) -> str | None:
        """Extrae el ID (32 hex, con o sin guiones) de una URL pública de Notion."""
        pattern = (
            r"([0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
            r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12})"
        )
        candidates = re.findall(pattern, url)
        if candidates:
            return NotionConnector._normalize_id(candidates[-1])
        return None
