"""Conector ``sharepoint``: documentos de Microsoft SharePoint vía Microsoft Graph.

Fase 7. URIs soportadas::

    sharepoint://<tenant>.sharepoint.com/sites/<Sitio>[/subruta]
    https://<tenant>.sharepoint.com/sites/<Sitio>[/…]

Ingesta los archivos de la librería de documentos por defecto del sitio
(Carpetas recursivas opcionales con ``folder=``).

Autenticación (client credentials — aplicación registrada en Microsoft Entra ID
con permiso de aplicación ``Sites.Read.All`` y consentimiento de administrador):

    RAG_SHAREPOINT_TENANT        id del tenant o dominio (contoso.onmicrosoft.com)
    RAG_SHAREPOINT_CLIENT_ID     id de cliente de la app
    RAG_SHAREPOINT_CLIENT_SECRET secreto de la app

Opciones de ingesta:
    folder:     carpeta de la librería (ej. "/General/2024"; vacío = raíz)
    drive_name: nombre de la librería (default: la primera del sitio)
    limit:      máximo de archivos a ingestar (default 50)
    extensions: extensiones permitidas (default: .txt .md .markdown .csv .pdf .docx)
    throttle:   pausa entre descargas en segundos (default 0.1; 0 en tests)
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import quote, urlparse

import httpx

from universal_rag.connectors.file.connector import FileConnector
from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.models import Document
from universal_rag.core.utils import get_logger

logger = get_logger("connectors.sharepoint")

GRAPH = "https://graph.microsoft.com/v1.0"
LOGIN = "https://login.microsoftonline.com"
DEFAULT_LIMIT = 50
DEFAULT_THROTTLE = 0.1
SUPPORTED_EXTENSIONS: ClassVar[tuple[str, ...]] = FileConnector.extensions
_TOKEN_SAFETY = 60.0  # renovar el token un minuto antes de expirar
_TOKEN_MAX_AGE = 3000.0  # cachear como máximo 50 minutos


class SharePointConnector(BaseConnector):
    """Ingesta de documentos de SharePoint vía Microsoft Graph (client credentials)."""

    name: ClassVar[str] = "sharepoint"
    description: ClassVar[str] = (
        "Ingesta de SharePoint: archivos de la librería de documentos de un sitio "
        "(sharepoint://<tenant>.sharepoint.com/sites/Sitio o URL https; requiere "
        "RAG_SHAREPOINT_TENANT + RAG_SHAREPOINT_CLIENT_ID + RAG_SHAREPOINT_CLIENT_SECRET)."
    )
    schemes: ClassVar[tuple[str, ...]] = ("sharepoint",)
    priority: ClassVar[int] = 10  # gana al conector `url` para URLs de sharepoint.com

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client  # inyectable (tests / reutilización de sesión)
        self._token: tuple[str, float] | None = None  # (access_token, expira_monotonic)

    # ------------------------------------------------------------------ API
    def can_handle(self, source: str) -> bool:
        lower = (source or "").strip().lower()
        if lower.startswith("sharepoint://"):
            return True
        return lower.startswith("http") and ".sharepoint.com/" in lower

    def fetch(self, source: str, **options: Any) -> list[Document]:
        tenant = options.get("tenant") or os.getenv("RAG_SHAREPOINT_TENANT")
        client_id = options.get("client_id") or os.getenv("RAG_SHAREPOINT_CLIENT_ID")
        client_secret = options.get("client_secret") or os.getenv("RAG_SHAREPOINT_CLIENT_SECRET")
        if not (tenant and client_id and client_secret):
            raise ConnectorError(
                "Faltan credenciales de SharePoint: define RAG_SHAREPOINT_TENANT, "
                "RAG_SHAREPOINT_CLIENT_ID y RAG_SHAREPOINT_CLIENT_SECRET (app Entra ID "
                "con permiso de aplicación Sites.Read.All) o pásalas como opciones "
                "tenant=/client_id=/client_secret=."
            )
        hostname, site_path = self._parse(source, options)
        limit = max(1, int(options.get("limit", DEFAULT_LIMIT)))
        throttle = max(0.0, float(options.get("throttle", DEFAULT_THROTTLE)))
        timeout = max(1.0, float(options.get("timeout", 30.0)))
        allowed = tuple(options["extensions"]) if options.get("extensions") else SUPPORTED_EXTENSIONS

        client = self._client or httpx.Client(timeout=timeout)
        close = self._client is None
        try:
            token = self._get_token(client, str(tenant), str(client_id), str(client_secret))
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            site = self._site(client, headers, hostname, site_path)
            drive = self._default_drive(client, headers, str(site.get("id", "")), options.get("drive_name"))
            drive_id = str(drive.get("id", ""))
            folder = str(options.get("folder") or "")
            documents: list[Document] = []
            for item in self._iter_files(client, headers, drive_id, folder):
                if len(documents) >= limit:
                    break
                name = str(item.get("name", ""))
                extension = Path(name).suffix.lower()
                if extension not in allowed:
                    continue
                document = self._file_document(client, headers, drive_id, item, extension)
                if document is not None:
                    documents.append(document)
                if throttle:
                    time.sleep(throttle)
            if not documents:
                raise ConnectorError(
                    f"No se extrajo ningún documento compatible de SharePoint "
                    f"({hostname}/{site_path}{' carpeta ' + folder if folder else ''}); "
                    f"extensiones soportadas: {', '.join(allowed)}"
                )
            return documents
        finally:
            if close:
                client.close()

    # -------------------------------------------------------------- routing
    @staticmethod
    def _parse(source: str, options: dict[str, Any]) -> tuple[str, str]:
        """Extrae (hostname, ruta_del_sitio), ej. ("contoso.sharepoint.com", "sites/Marketing")."""
        raw = source.strip()
        if raw.lower().startswith("sharepoint://"):
            rest = raw[len("sharepoint://") :].strip("/")
            parts = [p for p in rest.split("/") if p]
            if parts and "." in parts[0]:
                hostname, site_path = parts[0], "/".join(parts[1:])
            else:
                hostname = str(options.get("hostname") or "")
                site_path = "/".join(parts)
            if not hostname:
                raise ConnectorError(
                    "Falta el host de SharePoint: usa sharepoint://<tenant>.sharepoint.com/sites/<Sitio> "
                    "o ingesta la URL https://… del sitio"
                )
            if not site_path:
                raise ConnectorError(
                    "Falta la ruta del sitio: sharepoint://<host>/sites/<Sitio>"
                )
            return hostname, site_path.strip("/")
        if raw.lower().startswith("http"):
            parsed = urlparse(raw)
            hostname = parsed.hostname or ""
            site_path = parsed.path.strip("/")
            if not hostname or not site_path:
                raise ConnectorError(f"URL de SharePoint inválida: {source}")
            return hostname, site_path
        raise ConnectorError(
            f"Fuente no reconocida por el conector sharepoint: {source} "
            "(usa sharepoint://<host>/sites/<Sitio> o la URL https del sitio)"
        )

    # ---------------------------------------------------------------- auth
    def _get_token(self, client: httpx.Client, tenant: str, client_id: str, client_secret: str) -> str:
        """Token de acceso vía client credentials (cacheado en memoria)."""
        now = time.monotonic()
        if self._token and self._token[1] > now:
            return self._token[0]
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "https://graph.microsoft.com/.default",
        }
        url = f"{LOGIN}/{tenant}/oauth2/v2.0/token"
        try:
            response = client.post(url, data=data)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            raise ConnectorError(
                f"Microsoft Entra ID rechazó las credenciales ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Error de red pidiendo el token a Microsoft Entra: {exc}") from exc
        token = payload.get("access_token")
        if not token:
            raise ConnectorError(f"Respuesta de Entra ID sin access_token: {payload!r}")
        expires_in = float(payload.get("expires_in", 3600))
        self._token = (str(token), now + min(expires_in - _TOKEN_SAFETY, _TOKEN_MAX_AGE))
        return str(token)

    # ---------------------------------------------------------------- graph
    def _site(
        self, client: httpx.Client, headers: dict[str, str], hostname: str, site_path: str
    ) -> dict[str, Any]:
        url = f"{GRAPH}/sites/{hostname}:/{site_path}"
        return self._get_json(client, headers, url, error_for=f"sitio {hostname}/{site_path}")

    def _default_drive(
        self, client: httpx.Client, headers: dict[str, str], site_id: str, drive_name: str | None
    ) -> dict[str, Any]:
        data = self._get_json(client, headers, f"{GRAPH}/sites/{site_id}/drives", error_for="librerías del sitio")
        drives = data.get("value") or []
        if not drives:
            raise ConnectorError(f"El sitio {site_id} no tiene librerías de documentos")
        if drive_name:
            for drive in drives:
                if str(drive.get("name", "")).lower() == drive_name.lower():
                    return drive
            raise ConnectorError(
                f"Librería {drive_name!r} no encontrada; disponibles: "
                f"{', '.join(str(d.get('name')) for d in drives)}"
            )
        return drives[0]

    def _iter_files(self, client: httpx.Client, headers: dict[str, str], drive_id: str, folder: str):
        """Itera los archivos de una carpeta (paginado; solo items con ``file``)."""
        if folder:
            encoded = quote(folder.strip("/"), safe="/")
            url: str | None = f"{GRAPH}/drives/{drive_id}/root:/{encoded}:/children"
        else:
            url = f"{GRAPH}/drives/{drive_id}/root/children"
        params: dict[str, Any] | None = {"top": 200}
        pages = 0
        while url and pages < 50:  # tope duro de páginas: evita bucles infinitos
            data = self._get_json(client, headers, url, params=params, error_for="listado de archivos")
            pages += 1
            for item in data.get("value") or []:
                if isinstance(item, dict) and item.get("file") is not None:
                    yield item
            url = data.get("@odata.nextLink")
            params = None  # el nextLink ya trae los parámetros

    def _file_document(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        drive_id: str,
        item: dict[str, Any],
        extension: str,
    ) -> Document | None:
        """Descarga un archivo y lo convierte en Document (None si queda vacío)."""
        item_id = str(item.get("id", ""))
        name = str(item.get("name", "archivo"))
        url = f"{GRAPH}/drives/{drive_id}/items/{item_id}/content"
        try:
            response = client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("Descarga fallida de %s: %s", name, exc)
            return None
        if response.status_code >= 400:
            logger.warning("Descarga %d para %s (item %s)", response.status_code, name, item_id)
            return None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(response.content)
            text = extract_text(path)
        if not text.strip():
            return None
        parent_ref = (item.get("parentReference") or {}).get("path")
        return self.build_document(
            content=text,
            source=self.name,
            uri=f"sharepoint://item/{item_id}",
            title=Path(name).stem or name,
            metadata={
                "sharepoint_type": "file",
                "file_name": name,
                "extension": extension,
                "size": item.get("size"),
                "web_url": item.get("webUrl"),
                "folder": parent_ref,
                "last_modified": item.get("lastModifiedDateTime"),
                "e_tag": item.get("eTag"),
            },
            created_at=self._parse_time(item.get("lastModifiedDateTime")),
        )

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _get_json(
        client: httpx.Client,
        headers: dict[str, str],
        url: str,
        *,
        params: dict[str, Any] | None = None,
        error_for: str = "objeto",
    ) -> dict[str, Any]:
        try:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            hint = ""
            if exc.response.status_code in (401, 403):
                hint = " (revisa los permisos de aplicación Sites.Read.All y el consentimiento de admin)"
            elif exc.response.status_code == 404:
                hint = " (¿sitio o carpeta correctos?)"
            raise ConnectorError(
                f"Microsoft Graph respondió {exc.response.status_code} al leer {error_for}{hint}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectorError(f"Error de red llamando a Microsoft Graph ({url}): {exc}") from exc

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None


def extract_text(path: Path) -> str:
    """Extrae texto de un archivo descargado (reutiliza las rutinas del conector file)."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".csv":
        return FileConnector._read_csv(path, "utf-8")
    if suffix == ".pdf":
        text, _, _ = FileConnector._read_pdf(path)
        return text
    if suffix == ".docx":
        text, _, _ = FileConnector._read_docx(path)
        return text
    raise ConnectorError(f"Formato no soportado por el conector sharepoint: {suffix}")
