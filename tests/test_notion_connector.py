"""Tests del conector notion (Fase 5) — API REST mockeada, sin red real."""

from __future__ import annotations

import httpx
import pytest

from universal_rag.connectors.notion.connector import NotionConnector
from universal_rag.core.connectors import ConnectorError, ConnectorRegistry

PAGE_ID = "a" * 32
DB_ID = "b" * 32
ROW1 = "c" * 32
ROW2 = "d" * 32
TABLE_ID = "e" * 32
TOKEN = "secret-token"


# ----------------------------------------------------------------- fábrica HTTP
def _page(page_id: str, title: str) -> dict:
    return {
        "object": "page",
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "created_time": "2025-01-15T10:00:00.000Z",
        "last_edited_time": "2025-02-01T09:00:00.000Z",
        "properties": {
            "Nombre": {"type": "title", "title": [{"plain_text": title}]},
        },
    }


def _list_response(results: list[dict], has_more: bool = False, cursor: str | None = None) -> dict:
    return {"object": "list", "results": results, "has_more": has_more, "next_cursor": cursor}


def _blocks(results: list[dict]) -> dict:
    return _list_response(results)


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture()
def captured() -> dict:
    return {"auth": None, "version": None}


@pytest.fixture()
def handler(captured):
    def route(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["version"] = request.headers.get("notion-version")
        path = request.url.path
        if path.startswith("/v1"):
            path = path[len("/v1") :]

        if request.method == "GET" and path == f"/pages/{PAGE_ID}":
            return httpx.Response(200, json=_page(PAGE_ID, "Guía interna"))
        if request.method == "GET" and path == f"/blocks/{PAGE_ID}/children":
            return httpx.Response(
                200,
                json=_blocks(
                    [
                        {"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "Objetivos"}]}},
                        {
                            "type": "paragraph",
                            "paragraph": {"rich_text": [{"plain_text": "Contenido clave del documento."}]},
                        },
                        {"type": "to_do", "to_do": {"rich_text": [{"plain_text": "Revisar KPIs"}], "checked": False}},
                        {"type": "divider", "divider": {}},
                        {"type": "table", "id": TABLE_ID, "table": {}},
                    ]
                ),
            )
        if request.method == "GET" and path == f"/blocks/{TABLE_ID}/children":
            return httpx.Response(
                200,
                json=_blocks(
                    [
                        {
                            "type": "table_row",
                            "table_row": {"cells": [[{"plain_text": "KPI"}], [{"plain_text": "Valor"}]]},
                        },
                        {
                            "type": "table_row",
                            "table_row": {"cells": [[{"plain_text": "LTV"}], [{"plain_text": "3.2x"}]]},
                        },
                    ]
                ),
            )
        if request.method == "GET" and path == f"/databases/{DB_ID}":
            return httpx.Response(
                200,
                json={
                    "object": "database",
                    "id": DB_ID,
                    "title": [{"plain_text": "CRM"}],
                    "properties": {},
                },
            )
        if request.method == "POST" and path == f"/databases/{DB_ID}/query":
            body = request.read()
            if b"cur2" in body:
                return httpx.Response(200, json=_list_response([_page(ROW2, "Luis")]))
            if b"cur1" in body:
                return httpx.Response(200, json=_list_response([_page(ROW2, "Luis")], has_more=False))
            return httpx.Response(
                200, json=_list_response([_page(ROW1, "Ana")], has_more=True, cursor="cur1")
            )
        if request.method == "GET" and path == f"/blocks/{ROW1}/children":
            return httpx.Response(
                200,
                json=_blocks(
                    [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Cliente premium desde 2024."}]}}]
                ),
            )
        if request.method == "GET" and path == f"/blocks/{ROW2}/children":
            return httpx.Response(200, json=_blocks([]))
        if request.method == "POST" and path == "/search":
            return httpx.Response(
                200, json=_list_response([_page(PAGE_ID, "Guía interna")])
            )
        return httpx.Response(404, json={"message": f"no mock: {request.method} {path}"})

    return route


# --------------------------------------------------------------------- tests
class TestNotionCanHandle:
    def test_esquemas(self):
        conn = NotionConnector()
        assert conn.can_handle(f"notion://{PAGE_ID}")
        assert conn.can_handle(f"notion://page/{PAGE_ID}")
        assert conn.can_handle("https://www.notion.so/miworkspace/Gu-a-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        assert not conn.can_handle("https://ejemplo.com")
        assert not conn.can_handle("/ruta/archivo.pdf")


class TestNotionPage:
    def test_pagina_a_documento(self, handler):
        docs = NotionConnector(client=make_client(handler)).fetch(
            f"notion://page/{PAGE_ID}", token=TOKEN, throttle=0
        )
        (doc,) = docs
        assert doc.source == "notion"
        assert doc.uri == f"notion://page/{PAGE_ID}"
        assert doc.title == "Guía interna"
        assert "## Objetivos" in doc.content
        assert "Contenido clave del documento." in doc.content
        assert "- [ ] Revisar KPIs" in doc.content
        # tabla de Notion → markdown
        assert "| KPI | Valor |" in doc.content
        assert "| LTV | 3.2x |" in doc.content
        assert doc.metadata["notion_type"] == "page"
        assert doc.metadata["notion_url"].endswith(PAGE_ID)
        assert doc.created_at is not None and doc.created_at.year == 2025

    def test_cabeceras_auth(self, handler, captured):
        assert captured["auth"] is None  # aún sin llamar
        NotionConnector(client=make_client(handler)).fetch(
            f"notion://page/{PAGE_ID}", token=TOKEN, throttle=0
        )
        assert captured["auth"] == f"Bearer {TOKEN}"
        assert captured["version"] == "2022-06-28"

    def test_falta_token(self, handler, monkeypatch):
        monkeypatch.delenv("RAG_NOTION_TOKEN", raising=False)
        with pytest.raises(ConnectorError, match="RAG_NOTION_TOKEN"):
            NotionConnector(client=make_client(handler)).fetch(f"notion://page/{PAGE_ID}")

    def test_token_por_entorno(self, handler, captured, monkeypatch):
        monkeypatch.setenv("RAG_NOTION_TOKEN", TOKEN)
        docs = NotionConnector(client=make_client(handler)).fetch(
            f"notion://page/{PAGE_ID}", throttle=0
        )
        assert docs and captured["auth"] == f"Bearer {TOKEN}"

    def test_url_publica(self, handler):
        url = f"https://www.notion.so/miws/Gu-a-interna-{PAGE_ID[:8]}-{PAGE_ID[8:12]}-{PAGE_ID[12:16]}-{PAGE_ID[16:20]}-{PAGE_ID[20:]}"
        docs = NotionConnector(client=make_client(handler)).fetch(url, token=TOKEN, throttle=0)
        assert docs[0].uri == f"notion://page/{PAGE_ID}"

    def test_url_publica_sin_id(self, handler):
        with pytest.raises(ConnectorError, match="ID"):
            NotionConnector(client=make_client(handler)).fetch(
                "https://www.notion.so/sin-identificador", token=TOKEN
            )

    def test_id_invalido(self, handler):
        with pytest.raises(ConnectorError, match="inválido"):
            NotionConnector(client=make_client(handler)).fetch(
                "notion://page/xyz", token=TOKEN, throttle=0
            )

    def test_error_api(self):
        def handler_401(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "unauthorized"})

        with pytest.raises(ConnectorError, match="401"):
            NotionConnector(client=make_client(handler_401)).fetch(
                f"notion://page/{PAGE_ID}", token="malo", throttle=0
            )


class TestNotionDatabase:
    def test_base_de_datos_varias_filas(self, handler):
        docs = NotionConnector(client=make_client(handler)).fetch(
            f"notion://database/{DB_ID}", token=TOKEN, throttle=0
        )
        assert len(docs) == 1  # la fila vacía (ROW2/Luis) se omite
        ana = docs[0]
        assert ana.title == "CRM — Ana"
        assert ana.uri == f"notion://database/{DB_ID}/{ROW1}"
        assert "Cliente premium desde 2024." in ana.content
        assert ana.metadata["notion_type"] == "database_row"
        assert ana.metadata["database"] == "CRM"
        assert all("Luis" not in (doc.title or "") for doc in docs)

    def test_limit(self, handler):
        docs = NotionConnector(client=make_client(handler)).fetch(
            f"notion://database/{DB_ID}", token=TOKEN, throttle=0, limit=1
        )
        assert len(docs) == 1

    def test_search(self, handler):
        docs = NotionConnector(client=make_client(handler)).fetch(
            "notion://search/guía interna", token=TOKEN, throttle=0
        )
        assert docs[0].title == "Guía interna"
        assert docs[0].uri == f"notion://page/{PAGE_ID}"


class TestNotionRouting:
    def test_prioridad_sobre_url_conector(self):
        registry = ConnectorRegistry.discover()
        url = f"https://www.notion.so/ws/Pagina-{PAGE_ID}"
        assert registry.resolve(url).name == "notion"

    def test_describe(self):
        registry = ConnectorRegistry.discover()
        entry = next(e for e in registry.describe() if e["name"] == "notion")
        assert "notion" in entry["schemes"]
