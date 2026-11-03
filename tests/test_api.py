"""Tests de la API REST (FastAPI + TestClient)."""

import pytest
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app


@pytest.fixture()
def client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_health(client):
    import universal_rag

    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["version"] == universal_rag.__version__


def test_stats_vacio(client):
    data = client.get("/stats").json()
    assert data["documents"] == 0
    assert "connectors" in data


def test_ingest_text_y_query(client):
    response = client.post(
        "/ingest/text",
        json={"text": "La energía solar fotovoltaica convierte luz en electricidad.", "title": "Solar"},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["documents"] == 1

    response = client.post("/query", json={"question": "¿qué hace la energía solar fotovoltaica?"})
    assert response.status_code == 200
    answer = response.json()
    assert answer["citations"]
    assert answer["citations"][0]["uri"] == "text://Solar"


def test_ingest_fuente_invalida_422(client):
    response = client.post("/ingest", json={"source": "ftp://no/soportado"})
    assert response.status_code == 422


def test_documents_y_delete(client):
    client.post("/ingest/text", json={"text": "Documento para borrar.", "title": "Borrable"})
    docs = client.get("/documents").json()
    assert len(docs) == 1
    doc_id = docs[0]["document_id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/documents").json() == []


def test_ingest_archivo_via_api(client, tmp_path):
    target = tmp_path / "api.txt"
    target.write_text("Contenido vía API REST.", encoding="utf-8")
    response = client.post("/ingest", json={"source": str(target)})
    assert response.status_code == 200
    assert response.json()["connector"] == "file"
