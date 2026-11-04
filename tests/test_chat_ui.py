"""Tests de la mini chat UI servida en ``GET /`` (Fase 3)."""

import pytest
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app


@pytest.fixture()
def client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_raiz_sirve_chat_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "Universal RAG" in body
    assert "/query" in body  # el chat consume la API existente
    assert "/health" in body  # indicador de estado


def test_chat_ui_no_rompe_endpoints_existentes(client):
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    response = client.post("/query", json={"question": "pregunta en índice vacío"})
    assert response.status_code == 200
    assert "retrieval" in response.json()


def test_workbench_expone_modo_agente_fase8(client):
    """El workbench sirve el toggle del modo agente y consume /query/agent/stream (Fase 8)."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert 'id="agent-mode"' in body
    assert "/query/agent/stream" in body
    # eventos SSE del agente manejados por el JS del workbench
    for event_type in ("plan", "step", "synthesize"):
        assert f'"{event_type}"' in body or f"event === \"{event_type}\"" in body
