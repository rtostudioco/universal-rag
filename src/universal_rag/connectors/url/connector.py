"""Conector ``url``: extracción legible de páginas web (HTTP/HTTPS)."""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from universal_rag.core.connectors import BaseConnector
from universal_rag.core.models import Document
from universal_rag.core.utils import get_logger, utcnow

logger = get_logger("connectors.url")

_USER_AGENT = "UniversalRAG/0.1 (+https://github.com/rtostudioco/universal-rag)"


class URLConnector(BaseConnector):
    """Descarga una página web y extrae su contenido principal legible.

    Usa ``trafilatura`` si está disponible (extracción de alta calidad) y hace
    fallback a heurísticas con BeautifulSoup (main/article/limpieza).
    """

    name: ClassVar[str] = "url"
    description: ClassVar[str] = (
        "Ingesta de páginas web (http/https) con extracción legible del contenido principal."
    )
    schemes: ClassVar[tuple[str, ...]] = ("http", "https")

    def can_handle(self, source: str) -> bool:
        lower = (source or "").strip().lower()
        return lower.startswith("http://") or lower.startswith("https://")

    def fetch(self, source: str, **options: Any) -> list[Document]:
        url = source.strip()
        timeout = float(options.get("timeout", 20.0))
        extra_metadata = options.get("metadata") or {}

        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT, "Accept-Language": "es,en;q=0.8"},
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type or "xml" in content_type or not content_type:
            title, text = extract_html(response.text)
        else:
            title, text = None, response.text

        if not text.strip():
            from universal_rag.core.connectors import ConnectorError

            raise ConnectorError(f"No se pudo extraer contenido legible de: {url}")

        metadata = {
            **extra_metadata,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "content_type": content_type.split(";")[0].strip(),
            "fetched_at": utcnow().isoformat(),
        }
        document = self.build_document(
            content=text,
            source=self.name,
            uri=str(response.url),
            title=title or str(response.url),
            metadata=metadata,
        )
        return [document]


def extract_html(html: str) -> tuple[str | None, str]:
    """Extrae (título, texto) del HTML: trafilatura sobre HTML limpiado + fallback bs4.

    Primero se eliminan nodos de ruido (script, nav, footer…); después se
    intenta trafilatura (alta calidad sobre el contenido restante) y, si no
    produce texto, se usa la heurística bs4 (main/article/párrafos).
    """
    soup = _clean_soup(html)
    title = _soup_title(soup)

    text = ""
    if soup.find(["p", "h1", "article", "main"]):  # hay contenido estructurado que intentar
        try:
            import trafilatura

            text = trafilatura.extract(str(soup), include_comments=False, include_tables=True) or ""
        except Exception:  # noqa: BLE001 — fallback limpio
            logger.debug("trafilatura falló; usando bs4", exc_info=True)

    if not text.strip():
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n") if main else ""
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

    return title, text.strip()


def _soup_title(soup) -> str | None:  # noqa: ANN001
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return str(og["content"]).strip()
    return None


def _clean_soup(html: str):  # noqa: ANN001
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "form", "iframe"]):
        tag.decompose()
    return soup
