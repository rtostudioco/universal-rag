"""Conector ``confluence``: espacios y páginas de Confluence Cloud/Server (API REST).

Fase 7. URIs soportadas::

    confluence://space/<KEY>            todas las páginas del espacio
    confluence://page/<id>              una página concreta
    confluence://cql/<consulta CQL>     búsqueda CQL → páginas resultado
    confluence://<123456>               atajo: dígitos → página
    https://<tenant>.atlassian.net/wiki/spaces/<KEY>/…           espacio
    https://<tenant>.atlassian.net/wiki/spaces/<KEY>/pages/<id>/… página
    https://<servidor>/display/<KEY>/…                           (Server/DC)

Autenticación:
    Cloud:      email + token de API → Basic
                (``RAG_CONFLUENCE_EMAIL`` + ``RAG_CONFLUENCE_TOKEN``)
    Server/DC:  token personal → Bearer (define solo ``RAG_CONFLUENCE_TOKEN``)
    Base URL:   ``RAG_CONFLUENCE_URL`` (obligatoria para fuentes confluence://;
                las URLs https la deducen de su origen)

Opciones de ingesta:
    base_url:  URL base de la instancia (si no, ``RAG_CONFLUENCE_URL`` u origen)
    email:     email de la cuenta Cloud (si no, ``RAG_CONFLUENCE_EMAIL``)
    token:     token de API / PAT (si no, ``RAG_CONFLUENCE_TOKEN``)
    kind:      fuerza el tipo ("space" | "page" | "cql"; auto si se omite)
    cql:       consulta CQL a usar con ``confluence://cql`` o kind="cql"
    limit:     máximo de páginas por espacio/búsqueda (default 50)
    throttle:  pausa entre peticiones en segundos (default 0.2; 0 en tests)
"""

from __future__ import annotations

import base64
import os
import re
import time
from datetime import datetime
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString

from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.models import Document
from universal_rag.core.utils import get_logger

logger = get_logger("connectors.confluence")

PAGE_SIZE = 25
DEFAULT_LIMIT = 50
DEFAULT_THROTTLE = 0.2
EXPAND = "body.storage,version,space"

_PREFIXES: ClassVar[dict[str, str]] = {
    "p": "\n",
    "div": "\n",
    "pre": "\n",
    "h1": "\n# ",
    "h2": "\n## ",
    "h3": "\n### ",
    "h4": "\n#### ",
    "h5": "\n##### ",
    "h6": "\n###### ",
    "li": "\n- ",
    "tr": "\n",
}


def storage_to_text(html: str) -> str:
    """Convierte el formato de almacenamiento (XHTML) de Confluence a texto plano.

    Renderiza encabezados, listas, tablas (``a | b``) y macros de código a
    bloques `` ``` ``; el resto de macros se despliegan por su contenido.
    """
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for macro in soup.find_all("ac:structured-macro"):
        name = (macro.get("ac:name") or "").lower()
        if name in ("code", "noformat"):
            body = macro.find("ac:plain-text-body")
            code = body.get_text() if body else ""
            macro.replace_with(f"\n```\n{code.strip()}\n```\n")
        else:
            macro.unwrap()  # conservar el contenido interno de otras macros
    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"], recursive=False)]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    for tag_name, prefix in _PREFIXES.items():
        for element in soup.find_all(tag_name):
            element.insert(0, NavigableString(prefix))
    text = soup.get_text()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ConfluenceConnector(BaseConnector):
    """Ingesta de páginas de Confluence vía API REST (v1, compatible Cloud/Server)."""

    name: ClassVar[str] = "confluence"
    description: ClassVar[str] = (
        "Ingesta de Confluence: espacios, páginas y búsquedas CQL "
        "(confluence://space/KEY, confluence://page/id o URL de atlassian.net; "
        "requiere RAG_CONFLUENCE_URL + RAG_CONFLUENCE_TOKEN)."
    )
    schemes: ClassVar[tuple[str, ...]] = ("confluence",)
    priority: ClassVar[int] = 10  # gana al conector `url` para URLs de atlassian.net

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client  # inyectable (tests / reutilización de sesión)

    # ------------------------------------------------------------------ API
    def can_handle(self, source: str) -> bool:
        lower = (source or "").strip().lower()
        if lower.startswith("confluence://"):
            return True
        if lower.startswith("http"):
            return "atlassian.net" in lower or "/display/" in lower
        return False

    def fetch(self, source: str, **options: Any) -> list[Document]:
        base_url = (
            options.get("base_url")
            or os.getenv("RAG_CONFLUENCE_URL")
            or self._origin_from_url(source)
        )
        token = options.get("token") or os.getenv("RAG_CONFLUENCE_TOKEN")
        email = options.get("email") or os.getenv("RAG_CONFLUENCE_EMAIL")
        if not token:
            raise ConnectorError(
                "Falta el token de Confluence: pasa token=… en la ingesta o define "
                "RAG_CONFLUENCE_TOKEN (Cloud: email + token de API; Server: PAT)."
            )
        if not base_url:
            raise ConnectorError(
                "Falta la base URL de Confluence: define RAG_CONFLUENCE_URL "
                "(ej. https://mi-empresa.atlassian.net) o ingesta la URL completa https://…"
            )
        base_url = str(base_url).rstrip("/")
        limit = max(1, int(options.get("limit", DEFAULT_LIMIT)))
        throttle = max(0.0, float(options.get("throttle", DEFAULT_THROTTLE)))
        timeout = max(1.0, float(options.get("timeout", 30.0)))
        headers = self._auth_headers(str(token), email)
        kind, key = self._resolve(source, options)

        client = self._client or httpx.Client(timeout=timeout)
        close = self._client is None
        try:
            if kind == "page":
                pages = [
                    self._request(
                        client,
                        headers,
                        "GET",
                        f"{base_url}/rest/api/content/{key}",
                        params={"expand": EXPAND},
                        error_for="página",
                    )
                ]
            elif kind == "cql":
                pages = self._search_pages(client, headers, base_url, str(key), limit, throttle)
            else:
                pages = self._space_pages(client, headers, base_url, str(key), limit, throttle)
            documents = [self._page_document(page, base_url) for page in pages]
            if not documents:
                raise ConnectorError(f"Confluence no devolvió páginas para: {source}")
            return documents
        finally:
            if close:
                client.close()

    # -------------------------------------------------------------- routing
    def _resolve(self, source: str, options: dict[str, Any]) -> tuple[str, str]:
        """Determina (kind, clave): space | page | cql."""
        kind = options.get("kind")  # "space" | "page" | "cql" | None (auto)
        cql = options.get("cql")
        raw = source.strip()
        if raw.lower().startswith("confluence://"):
            rest = raw[len("confluence://") :].strip("/")
            head, _, tail = rest.partition("/")
            if head.lower() in ("space", "page", "cql") and tail:
                return self._final_kind(kind, cql, head.lower(), unquote(tail))
            if cql:
                return "cql", str(cql)
            # atajo: dígitos → página; si no → espacio
            return ("page", rest) if rest.isdigit() else ("space", unquote(rest))
        # URLs https de Atlassian
        page_match = re.search(r"/spaces/([^/]+)/pages/(\d+)", raw)
        if page_match:
            return "page", page_match.group(2)
        space_match = re.search(r"/(?:spaces|display)/([^/?#]+)", raw)
        if space_match:
            return self._final_kind(kind, cql, "space", unquote(space_match.group(1)))
        if cql:
            return "cql", str(cql)
        raise ConnectorError(
            f"No se pudo deducir espacio/página de Confluence desde: {source} "
            "(usa confluence://space/KEY, confluence://page/ID o la URL del espacio)"
        )

    @staticmethod
    def _final_kind(
        kind: str | None, cql: Any, default_kind: str, key: str
    ) -> tuple[str, str]:
        if cql:
            return "cql", str(cql)
        if kind in ("space", "page", "cql"):
            return kind, key
        return default_kind, key

    @staticmethod
    def _origin_from_url(source: str) -> str | None:
        candidate = (source or "").strip()
        if candidate.lower().startswith("http"):
            parsed = urlparse(candidate)
            if parsed.hostname:
                return f"{parsed.scheme}://{parsed.netloc}"
        return None

    @staticmethod
    def _auth_headers(token: str, email: str | None) -> dict[str, str]:
        if email:
            encoded = base64.b64encode(f"{email}:{token}".encode()).decode("ascii")
            return {"Authorization": f"Basic {encoded}", "Accept": "application/json"}
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # ------------------------------------------------------------- listing
    def _space_pages(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        base_url: str,
        space_key: str,
        limit: int,
        throttle: float,
    ) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        start = 0
        while len(pages) < limit:
            batch = min(PAGE_SIZE, limit - len(pages))
            data = self._request(
                client,
                headers,
                "GET",
                f"{base_url}/rest/api/content",
                params={
                    "type": "page",
                    "spaceKey": space_key,
                    "expand": EXPAND,
                    "limit": batch,
                    "start": start,
                },
                error_for=f"espacio {space_key!r}",
            )
            results = data.get("results") or []
            pages.extend(results)
            if len(results) < batch:
                break
            start += len(results)
            if throttle:
                time.sleep(throttle)
        return pages

    def _search_pages(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        base_url: str,
        cql: str,
        limit: int,
        throttle: float,
    ) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        start = 0
        while len(pages) < limit:
            batch = min(PAGE_SIZE, limit - len(pages))
            data = self._request(
                client,
                headers,
                "GET",
                f"{base_url}/rest/api/content/search",
                params={"cql": cql, "expand": EXPAND, "limit": batch, "start": start},
                error_for="búsqueda CQL",
            )
            results = data.get("results") or []
            pages.extend(results)
            if len(results) < batch:
                break
            start += len(results)
            if throttle:
                time.sleep(throttle)
        return pages

    # ----------------------------------------------------------- documents
    def _page_document(self, page: dict[str, Any], base_url: str) -> Document:
        page_id = str(page.get("id", ""))
        if not page_id:
            raise ConnectorError("Página de Confluence sin ID")
        title = page.get("title") or f"Confluence {page_id}"
        links = page.get("_links") or {}
        webui = links.get("webui") or ""
        url = f"{links.get('base') or base_url}{webui}" if webui else None
        space = (page.get("space") or {}).get("key")
        version = page.get("version") or {}
        content = storage_to_text((page.get("body") or {}).get("storage", {}).get("value", ""))
        return self.build_document(
            content=content or title,
            source=self.name,
            uri=f"confluence://page/{page_id}",
            title=title,
            metadata={
                "confluence_type": "page",
                "space": space,
                "page_id": page_id,
                "version": version.get("number"),
                "url": url,
                "last_updated": version.get("when"),
            },
            created_at=self._parse_time(version.get("when")),
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _request(
        client: httpx.Client,
        headers: dict[str, str],
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        error_for: str = "objeto",
    ) -> dict[str, Any]:
        try:
            response = client.request(method, path, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            hint = ""
            if exc.response.status_code in (401, 403):
                hint = " (revisa RAG_CONFLUENCE_EMAIL/RAG_CONFLUENCE_TOKEN y los permisos del espacio)"
            raise ConnectorError(
                f"La API de Confluence respondió {exc.response.status_code} al leer {error_for}{hint}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Error de red llamando a Confluence ({path}): {exc}") from exc

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
