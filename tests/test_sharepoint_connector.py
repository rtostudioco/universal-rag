"""Tests del conector sharepoint (Fase 7) — Microsoft Graph simulado con MockTransport."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
import pytest
from test_connectors import build_minimal_pdf

from universal_rag.connectors.sharepoint.connector import SharePointConnector
from universal_rag.core.connectors import ConnectorError

TENANT = "contoso.onmicrosoft.com"
HOST = "contoso.sharepoint.com"

NOTAS = b"Notas compartidas desde el sitio de Marketing."
GUÍA = b"# Guia\n\nContenido markdown desde SharePoint."


def graph_responses(capture: dict | None = None, *, files: list[dict] | None = None):
    """Servidor falso de Microsoft Graph + Entra ID (client credentials)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.setdefault("requests", []).append(request)
        url = str(request.url)
        parsed = urlparse(url)
        path = parsed.path
        if parsed.hostname == "login.microsoftonline.com":
            return httpx.Response(200, json={"access_token": "tok-graph", "expires_in": 3600})
        if path.endswith("/content"):
            item_id = path.split("/items/")[1].split("/")[0]
            payloads = {item["id"]: item.get("_bytes", b"") for item in files or []}
            return httpx.Response(200, content=payloads.get(item_id, b""))
        if path.endswith("/children"):
            public = [{k: v for k, v in item.items() if not k.startswith("_")} for item in files or []]
            return httpx.Response(200, json={"value": public})
        if path.endswith("/drives"):
            return httpx.Response(200, json={"value": [{"id": "drive1", "name": "Documentos"}]})
        if path.startswith("/v1.0/sites/"):
            return httpx.Response(200, json={"id": "contoso,guid1,guid2", "displayName": "Marketing"})
        return httpx.Response(404, text="no encontrado")

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_items() -> list[dict]:
    return [
        {
            "id": "i1",
            "name": "notas.txt",
            "file": {"mimeType": "text/plain"},
            "size": 41,
            "webUrl": f"https://{HOST}/sites/Marketing/Documentos/notas.txt",
            "lastModifiedDateTime": "2024-06-01T12:00:00Z",
            "_bytes": NOTAS,
        },
        {
            "id": "i2",
            "name": "informe.pdf",
            "file": {"mimeType": "application/pdf"},
            "size": 900,
            "webUrl": f"https://{HOST}/sites/Marketing/Documentos/informe.pdf",
            "_bytes": build_minimal_pdf("Informe desde SharePoint"),
        },
        {"id": "i3", "name": "foto.png", "file": {"mimeType": "image/png"}, "size": 10, "_bytes": b"\x89PNG"},
        {"id": "i4", "name": "Subcarpeta", "folder": {"childCount": 2}, "size": 0},
    ]


@pytest.fixture()
def creds(monkeypatch):
    monkeypatch.setenv("RAG_SHAREPOINT_TENANT", TENANT)
    monkeypatch.setenv("RAG_SHAREPOINT_CLIENT_ID", "client-123")
    monkeypatch.setenv("RAG_SHAREPOINT_CLIENT_SECRET", "secret-456")


@pytest.fixture()
def no_creds(monkeypatch):
    for var in ("RAG_SHAREPOINT_TENANT", "RAG_SHAREPOINT_CLIENT_ID", "RAG_SHAREPOINT_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)


class TestCanHandleYParse:
    def test_can_handle(self):
        conn = SharePointConnector()
        assert conn.can_handle(f"sharepoint://{HOST}/sites/Mkt")
        assert conn.can_handle(f"https://{HOST}/sites/Mkt")
        assert not conn.can_handle("https://google.com/search")
        assert not conn.can_handle("/ruta/local.pdf")

    def test_parse_variantes(self):
        assert SharePointConnector._parse(f"sharepoint://{HOST}/sites/Marketing", {}) == (HOST, "sites/Marketing")
        assert SharePointConnector._parse(f"https://{HOST}/sites/Marketing", {}) == (HOST, "sites/Marketing")
        host, path = SharePointConnector._parse("sharepoint://sites/Mkt", {"hostname": HOST})
        assert (host, path) == (HOST, "sites/Mkt")

    def test_parse_sin_host(self, no_creds):
        with pytest.raises(ConnectorError, match="host"):
            SharePointConnector._parse("sharepoint://sites/Mkt", {})

    def test_parse_sin_ruta(self, no_creds):
        with pytest.raises(ConnectorError, match="sitio"):
            SharePointConnector._parse(f"sharepoint://{HOST}", {})


class TestCredenciales:
    def test_faltan_credenciales(self, no_creds):
        conn = SharePointConnector(client=graph_responses())
        with pytest.raises(ConnectorError, match="RAG_SHAREPOINT_TENANT"):
            conn.fetch(f"sharepoint://{HOST}/sites/Marketing")

    def test_token_rechazado(self, no_creds, monkeypatch):
        monkeypatch.setenv("RAG_SHAREPOINT_TENANT", TENANT)
        monkeypatch.setenv("RAG_SHAREPOINT_CLIENT_ID", "bad")
        monkeypatch.setenv("RAG_SHAREPOINT_CLIENT_SECRET", "bad")

        def deny(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="invalid_client")

        conn = SharePointConnector(client=httpx.Client(transport=httpx.MockTransport(deny)))
        with pytest.raises(ConnectorError, match="Entra ID"):
            conn.fetch(f"sharepoint://{HOST}/sites/Marketing")


class TestFetch:
    def test_ingesta_completa(self, creds):
        capture: dict = {}
        conn = SharePointConnector(client=graph_responses(capture, files=make_items()))
        documents = conn.fetch(f"sharepoint://{HOST}/sites/Marketing", throttle=0)
        assert [d.title for d in documents] == ["notas", "informe"]
        assert documents[0].uri == "sharepoint://item/i1"
        assert documents[0].source == "sharepoint"
        assert "Notas compartidas" in documents[0].content
        assert documents[0].metadata["extension"] == ".txt"
        assert documents[0].metadata["web_url"].endswith("notas.txt")
        assert documents[0].created_at is not None
        # PDF real extraído con pypdf
        assert "Informe desde SharePoint" in documents[1].content
        # primero el token, luego el sitio
        first, second = capture["requests"][:2]
        assert "login.microsoftonline.com" in str(first.url)
        assert "graph.microsoft.com" in str(second.url)
        assert "/v1.0/sites/" in str(second.url)

    def test_url_https(self, creds):
        conn = SharePointConnector(client=graph_responses(files=make_items()))
        documents = conn.fetch(f"https://{HOST}/sites/Marketing", throttle=0)
        assert len(documents) == 2

    def test_extensiones_filtradas(self, creds):
        conn = SharePointConnector(client=graph_responses(files=make_items()))
        documents = conn.fetch(f"sharepoint://{HOST}/sites/Mkt", throttle=0)
        extensions = {d.metadata["extension"] for d in documents}
        assert ".png" not in extensions

    def test_carpeta_opcional(self, creds):
        capture: dict = {}
        conn = SharePointConnector(client=graph_responses(capture, files=make_items()))
        conn.fetch(f"sharepoint://{HOST}/sites/Mkt", folder="/General/2024", throttle=0)
        listing = next(str(r.url) for r in capture["requests"] if "/children" in str(r.url))
        assert "root:/General/2024:/children" in listing

    def test_drive_name_inexistente(self, creds):
        conn = SharePointConnector(client=graph_responses(files=make_items()))
        with pytest.raises(ConnectorError, match="Librería"):
            conn.fetch(f"sharepoint://{HOST}/sites/Mkt", drive_name="NoExiste", throttle=0)

    def test_sitio_inexistente(self, creds, monkeypatch):
        def not_found(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="sitio no existe")

        conn = SharePointConnector(client=httpx.Client(transport=httpx.MockTransport(not_found)))
        with pytest.raises(ConnectorError, match="404"):
            conn.fetch(f"sharepoint://{HOST}/sites/Nope")

    def test_extensiones_custom(self, creds):
        conn = SharePointConnector(client=graph_responses(files=make_items()))
        documents = conn.fetch(
            f"sharepoint://{HOST}/sites/Mkt", extensions=[".txt"], throttle=0
        )
        assert [d.title for d in documents] == ["notas"]


def test_extract_text_csv(tmp_path):
    from universal_rag.connectors.sharepoint.connector import extract_text

    csv_path = tmp_path / "datos.csv"
    csv_path.write_text("nombre,valor\nsolar,450\n", encoding="utf-8")
    text = extract_text(csv_path)
    assert "nombre: solar" in text
