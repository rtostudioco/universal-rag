"""Tests de jobs de ingesta asíncrona (Fase 4)."""

import time

import pytest
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app
from universal_rag.core.jobs import JobManager, JobStatus
from universal_rag.core.models import IngestResult


# --------------------------------------------------------------- JobManager
def test_submit_exitoso():
    manager = JobManager()
    job_id = manager.submit(
        lambda: IngestResult(connector="text", documents=2, chunks=7),
        source="demo",
        collection="legal",
    )
    job = manager.wait(job_id, timeout=10)
    assert job is not None
    assert job.status == JobStatus.DONE
    assert job.result is not None and job.result.chunks == 7
    assert job.collection == "legal"
    assert job.started_at is not None and job.finished_at is not None


def test_submit_con_error():
    def boom() -> IngestResult:
        raise RuntimeError("fallo simulado")

    manager = JobManager()
    job_id = manager.submit(boom, source="malo")
    job = manager.wait(job_id, timeout=10)
    assert job is not None and job.status == JobStatus.ERROR
    assert job.error and "RuntimeError" in job.error
    assert job.result is None


def test_get_desconocido_y_listado():
    manager = JobManager()
    assert manager.get("no-existe") is None
    id1 = manager.submit(lambda: IngestResult(connector="text"), source="a")
    id2 = manager.submit(lambda: IngestResult(connector="text"), source="b")
    manager.wait(id1, timeout=10)
    jobs = manager.list_jobs()
    assert [j.id for j in jobs][:2] == [id2, id1]  # más reciente primero
    assert manager.list_jobs(limit=1)[0].id == id2


def test_purguea_historial():
    manager = JobManager(max_history=10)
    for i in range(15):
        job_id = manager.submit(lambda i=i: IngestResult(connector="text"), source=str(i))
        manager.wait(job_id, timeout=10)
    assert len(manager.list_jobs()) <= 10


def test_wait_timeout_devuelve_estado_actual():
    manager = JobManager()

    def lento() -> IngestResult:
        time.sleep(0.3)
        return IngestResult(connector="text")

    job_id = manager.submit(lento, source="lento")
    job = manager.wait(job_id, timeout=0.01)
    assert job is not None and job.status in (JobStatus.RUNNING, JobStatus.DONE, JobStatus.PENDING)
    job = manager.wait(job_id, timeout=10)
    assert job is not None and job.status == JobStatus.DONE


# ------------------------------------------------------------------ API REST
@pytest.fixture()
def client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_ingest_async_flow(client):
    response = client.post(
        "/ingest/async",
        json={"source": "text://La energía solar fotovoltaica convierte luz en electricidad.", "connector": "text"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["poll"] == f"/jobs/{data['job_id']}"

    job = client.get(f"/jobs/{data['job_id']}").json()
    assert job["id"] == data["job_id"]

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{data['job_id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done"
    assert job["result"]["documents"] == 1

    listed = client.get("/jobs").json()
    assert any(j["id"] == data["job_id"] for j in listed)


def test_ingest_async_coleccion_y_documentos(client):
    data = client.post(
        "/ingest/async",
        json={"source": "text://Texto para colección legal.", "connector": "text", "collection": "legal"},
    ).json()
    job = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{data['job_id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done"
    assert job["collection"] == "legal"
    docs = client.get("/documents?collection=legal").json()
    assert len(docs) == 1


def test_job_desconocido_404(client):
    assert client.get("/jobs/zzz").status_code == 404
