"""Tests de autenticación por API key y métricas runtime (Fase 4)."""

import pytest
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app
from universal_rag.core.metrics import Metrics


@pytest.fixture()
def client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def secure_settings(settings):
    return settings.model_copy(update={"api_key": "secreto-123"})


@pytest.fixture()
def secure_client(secure_settings) -> TestClient:
    app = create_app(secure_settings)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------- auth
def test_sin_key_401(secure_client):
    assert secure_client.get("/stats").status_code == 401
    assert secure_client.post("/query", json={"question": "hola?"}).status_code == 401
    assert secure_client.get("/documents").status_code == 401
    assert secure_client.get("/collections").status_code == 401


def test_con_x_api_key(secure_client):
    assert secure_client.get("/stats", headers={"X-API-Key": "secreto-123"}).status_code == 200


def test_con_bearer(secure_client):
    assert (
        secure_client.get("/stats", headers={"Authorization": "Bearer secreto-123"}).status_code
        == 200
    )


def test_key_incorrecta_401(secure_client):
    assert secure_client.get("/stats", headers={"X-API-Key": "mala"}).status_code == 401


def test_rutas_publicas_abiertas(secure_client):
    assert secure_client.get("/health").status_code == 200
    assert secure_client.get("/").status_code == 200  # chat UI


def test_sin_configurar_auth_todo_abierto(client):
    """Sin RAG_API_KEY definida, no se exige autenticación."""
    assert client.get("/stats").status_code == 200
    assert client.get("/documents").status_code == 200


# ------------------------------------------------------------------- métricas
def test_metrics_snapshot():
    metrics = Metrics()
    before = metrics.snapshot()
    assert before["requests"] == 0 and before["uptime_seconds"] >= 0
    metrics.inc("queries", 3)
    metrics.inc("queries")
    metrics.inc("errors")
    snapshot = metrics.snapshot()
    assert snapshot["queries"] == 4
    assert snapshot["errors"] == 1


def test_api_registra_metricas(client):
    client.get("/health")
    client.post("/query", json={"question": "¿solar?"})
    client.post("/ingest/text", json={"text": "Texto para métricas.", "title": "M"})
    stats = client.get("/stats").json()
    runtime = stats["runtime"]
    assert runtime["requests"] >= 4  # health + query + ingest + stats
    assert runtime["queries"] >= 1
    assert runtime["ingests"] >= 1


def test_stats_expone_colecciones(client):
    client.post("/ingest/text", json={"text": "Dato en colección.", "title": "C", "collection": "kpi"})
    stats = client.get("/stats").json()
    assert any(row["name"] == "kpi" for row in stats["collections"])
