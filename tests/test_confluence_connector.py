"""Tests del conector confluence (Fase 7) — API simulada con httpx.MockTransport."""

from __future__ import annotations

import base64
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from universal_rag.connectors.confluence.connector import ConfluenceConnector, storage_to_text
from universal_rag.core.connectors import ConnectorError

BASE = "https://miempresa.atlassian.net"


def make_page(pid: int = 101, title: str = "Guía de seguridad", storage: str = "<p>Hola mundo</p>") -> dict:
    return {
        "id": str(pid),
        "type": "page",
        "title": title,
        "_links": {"webui": f"/spaces/DEV/pages/{pid}", "base": BASE},
        "space": {"key": "DEV"},
        "version": {"number": 7, "when": "2024-05-01T10:00:00.000Z"},
        "body": {"storage": {"value": storage, "representation": "storage"}},
    }


def make_client(pages: list[dict], capture: dict | None = None, status: int = 200) -> httpx.Client:
    """Servidor falso de la API REST de Confluence."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.setdefault("requests", []).append(request)
        if status != 200:
            return httpx.Response(status, text="denegado")
        path = urlparse(str(request.url)).path
        params = {k: v[0] for k, v in parse_qs(str(request.url.query)).items()}
        if path == "/rest/api/content" and params.get("spaceKey"):
            limit = min(int(params.get("limit", 25)), len(pages))
            return httpx.Response(200, json={"results": pages[:limit], "size": limit})
        if path == "/rest/api/content/search":
            limit = min(int(params.get("limit", 25)), len(pages))
            return httpx.Response(200, json={"results": pages[:limit], "size": limit})
        page_id = path.rsplit("/", 1)[-1]
        for page in pages:
            if page["id"] == page_id:
                return httpx.Response(200, json=page)
        return httpx.Response(404, text="no encontrado")

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture()
def no_env(monkeypatch):
    for var in ("RAG_CONFLUENCE_URL", "RAG_CONFLUENCE_EMAIL", "RAG_CONFLUENCE_TOKEN"):
        monkeypatch.delenv(var, raising=False)


class TestCanHandleYResolve:
    def test_can_handle(self):
        conn = ConfluenceConnector()
        assert conn.can_handle("confluence://space/DEV")
        assert conn.can_handle("confluence://page/12345")
        assert conn.can_handle("https://acme.atlassian.net/wiki/spaces/DEV")
        assert conn.can_handle("https://servidor.corp/display/DEV/Nota")
        assert not conn.can_handle("https://ejemplo.com/otra/cosa")
        assert not conn.can_handle("/ruta/local.pdf")

    def test_resolve_variants(self):
        conn = ConfluenceConnector()
        assert conn._resolve("confluence://space/DEV", {}) == ("space", "DEV")
        assert conn._resolve("confluence://page/123", {}) == ("page", "123")
        assert conn._resolve("confluence://123", {}) == ("page", "123")
        assert conn._resolve("confluence://DEV", {}) == ("space", "DEV")
        assert conn._resolve("confluence://cql/text ~ %22rag%22", {}) == ("cql", "text ~ \"rag\"")
        assert conn._resolve("https://acme.atlassian.net/wiki/spaces/DEV/pages/77/Titulo", {}) == ("page", "77")
        assert conn._resolve("https://acme.atlassian.net/wiki/spaces/DEV/overview", {}) == ("space", "DEV")
        assert conn._resolve("https://servidor.corp/display/DEV/Nota", {}) == ("space", "DEV")
        assert conn._resolve("confluence://space/DEV", {"cql": 'text ~ "x"'}) == ("cql", 'text ~ "x"')

    def test_resolve_sin_pista(self):
        with pytest.raises(ConnectorError, match="Confluence"):
            ConfluenceConnector()._resolve("https://ejemplo.com/x/y", {})


class TestStorageToText:
    def test_bloques_basicos(self):
        html = "<p>Hola <br/>mundo</p><h2>Sección</h2><ul><li>Uno</li><li>Dos</li></ul>"
        text = storage_to_text(html)
        assert "Hola\nmundo" in text
        assert "## Sección" in text
        assert "- Uno" in text and "- Dos" in text

    def test_tabla(self):
        html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        assert "A | B" in storage_to_text(html)
        assert "1 | 2" in storage_to_text(html)

    def test_macro_codigo(self):
        html = (
            '<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">py</ac:parameter>'
            "<ac:plain-text-body><![CDATA[print('hola')]]></ac:plain-text-body></ac:structured-macro>"
        )
        text = storage_to_text(html)
        assert "```" in text
        assert "print('hola')" in text

    def test_macro_desconocido_conserva_contenido(self):
        html = '<ac:structured-macro ac:name="info"><ac:rich-text-body><p>Atención</p></ac:rich-text-body></ac:structured-macro>'
        assert "Atención" in storage_to_text(html)

    def test_vacio(self):
        assert storage_to_text("") == ""
        assert storage_to_text("   ") == ""


class TestFetch:
    def test_espacio_completo(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_EMAIL", "svc@acme.com")
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok-123")
        pages = [make_page(101, "Guía", "<p>Contenido uno</p>"), make_page(102, "Manual", "<p>Contenido dos</p>")]
        conn = ConfluenceConnector(client=make_client(pages))
        documents = conn.fetch("confluence://space/DEV", throttle=0)
        assert [d.title for d in documents] == ["Guía", "Manual"]
        assert documents[0].uri == "confluence://page/101"
        assert documents[0].source == "confluence"
        assert "Contenido uno" in documents[0].content
        assert documents[0].metadata["space"] == "DEV"
        assert documents[0].metadata["version"] == 7
        assert documents[0].metadata["url"] == f"{BASE}/spaces/DEV/pages/101"
        assert documents[0].created_at is not None

    def test_pagina_unica(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok")
        conn = ConfluenceConnector(client=make_client([make_page(101, "Guía", "<p>Uno</p>")]))
        documents = conn.fetch("confluence://page/101")
        assert len(documents) == 1
        assert documents[0].title == "Guía"

    def test_busqueda_cql(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok")
        pages = [make_page(101, "Encontrada", "<p>Resultado</p>")]
        conn = ConfluenceConnector(client=make_client(pages))
        documents = conn.fetch("confluence://cql/text ~ %22seguridad%22", throttle=0)
        assert documents[0].title == "Encontrada"

    def test_url_https_deduce_base_y_pagina(self, no_env):
        conn = ConfluenceConnector(client=make_client([make_page(101, "Por URL", "<p>x</p>")]))
        documents = conn.fetch(f"{BASE}/wiki/spaces/DEV/pages/101/Titulo", token="tok", throttle=0)
        assert documents[0].title == "Por URL"

    def test_limit_respeta_parametro(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok")
        pages = [make_page(100 + i, f"P{i}") for i in range(10)]
        conn = ConfluenceConnector(client=make_client(pages))
        documents = conn.fetch("confluence://space/DEV", limit=3, throttle=0)
        assert len(documents) == 3

    def test_falta_token(self, no_env):
        conn = ConfluenceConnector(client=make_client([make_page()]))
        with pytest.raises(ConnectorError, match="token"):
            conn.fetch("confluence://space/DEV")

    def test_falta_base_url(self, no_env):
        conn = ConfluenceConnector(client=make_client([make_page()]))
        with pytest.raises(ConnectorError, match="base URL|RAG_CONFLUENCE_URL"):
            conn.fetch("confluence://space/DEV", token="tok")

    def test_auth_basic_con_email(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_EMAIL", "svc@acme.com")
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok-123")
        capture: dict = {}
        conn = ConfluenceConnector(client=make_client([make_page()], capture))
        conn.fetch("confluence://space/DEV", throttle=0)
        header = capture["requests"][0].headers["authorization"]
        assert header.startswith("Basic ")
        expected = base64.b64encode(b"svc@acme.com:tok-123").decode()
        assert header == f"Basic {expected}"

    def test_auth_bearer_sin_email(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "pat-server")
        capture: dict = {}
        conn = ConfluenceConnector(client=make_client([make_page()], capture))
        conn.fetch("confluence://space/DEV", throttle=0)
        assert capture["requests"][0].headers["authorization"] == "Bearer pat-server"

    def test_credenciales_invalidas(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok")
        conn = ConfluenceConnector(client=make_client([make_page()], status=401))
        with pytest.raises(ConnectorError, match="401"):
            conn.fetch("confluence://space/DEV")

    def test_espacio_inexistente(self, no_env, monkeypatch):
        monkeypatch.setenv("RAG_CONFLUENCE_URL", BASE)
        monkeypatch.setenv("RAG_CONFLUENCE_TOKEN", "tok")
        conn = ConfluenceConnector(client=make_client([]))
        with pytest.raises(ConnectorError, match="no devolvió páginas"):
            conn.fetch("confluence://space/NOPE", throttle=0)
