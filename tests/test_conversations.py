"""Tests de memoria conversacional (pipeline + API) y sincronización vía API (Fase 7)."""

from __future__ import annotations

import json

import pytest
from conftest import SOLAR_TEXT
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app
from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline


class RecordingGenerator:
    """Generador falso que registra el historial recibido."""

    name = "rec"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @property
    def model_label(self) -> str:
        return "rec"

    def generate(self, question: str, contexts, history=None):
        self.calls.append({"question": question, "history": history})
        return f"Respuesta a «{question}» [1]", self.name


@pytest.fixture()
def recording_pipeline(pipeline_con_datos) -> tuple[RAGPipeline, RecordingGenerator]:
    generator = RecordingGenerator()
    pipeline_con_datos.generator = generator
    return pipeline_con_datos, generator


# ----------------------------------------------------------------- pipeline
class TestPipelineConversacional:
    def test_registra_turno(self, recording_pipeline):
        pipeline, _ = recording_pipeline
        answer = pipeline.query("¿qué es la energía solar?", conversation_id="conv1")
        assert answer.conversation_id == "conv1"
        assert pipeline.memory.turns("conv1") == 1

    def test_segundo_turno_recibe_historial(self, recording_pipeline):
        pipeline, generator = recording_pipeline
        pipeline.query("¿qué es la energía solar?", conversation_id="conv1")
        pipeline.query("¿y la eólica?", conversation_id="conv1")
        history = generator.calls[1]["history"]
        assert history is not None
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "¿qué es la energía solar?"
        assert "Respuesta a «¿qué es la energía solar?»" in history[1]["content"]

    def test_sin_conversacion_no_hay_historial(self, recording_pipeline):
        pipeline, generator = recording_pipeline
        pipeline.query("¿qué es la energía solar?")
        assert generator.calls[0]["history"] is None

    def test_cache_omitida_en_conversaciones(self, recording_pipeline):
        """Misma pregunta dos veces con conversation_id: el generador se llama dos veces."""
        pipeline, generator = recording_pipeline
        pipeline.query("¿qué es la energía solar?", conversation_id="conv1")
        pipeline.query("¿qué es la energía solar?", conversation_id="conv1")
        assert len(generator.calls) == 2

    def test_respuesta_sin_contexto_registra_turno(self, settings):
        pipeline = RAGPipeline(settings=settings)
        answer = pipeline.query("pregunta huérfana", conversation_id="conv9")
        assert "No encontré información" in answer.answer
        assert pipeline.memory.turns("conv9") == 1

    def test_historial_limitado_por_configuracion(self, recording_pipeline):
        pipeline, generator = recording_pipeline
        pipeline.settings = Settings(
            store_provider="memory",
            embedding_provider="hash",
            embedding_dimensions=128,
            llm_provider="echo",
            connector_dirs=[],
            memory_max_turns=1,
        )
        pipeline.memory = pipeline.memory.__class__(max_turns=1)
        pipeline.query("¿uno?", conversation_id="c")
        pipeline.query("¿dos?", conversation_id="c")
        pipeline.query("¿tres?", conversation_id="c")
        history = generator.calls[2]["history"]
        assert history is not None
        assert len(history) == 2  # solo el turno anterior (max_turns=1)
        assert history[0]["content"] == "¿dos?"

    def test_stats_incluyen_memoria(self, recording_pipeline):
        pipeline, _ = recording_pipeline
        pipeline.query("¿qué es la energía solar?", conversation_id="conv1")
        assert pipeline.stats()["memory"]["conversations"] == 1


# ---------------------------------------------------------------------- API
@pytest.fixture()
def client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        etype, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                etype = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if etype:
            events.append((etype, json.loads(data) if data else {}))
    return events


class TestAPIConversaciones:
    def test_query_conversacional(self, client):
        client.post("/ingest/text", json={"text": SOLAR_TEXT, "title": "Solar"})
        response = client.post(
            "/query", json={"question": "¿cuánto produce un panel?", "conversation_id": "web-1"}
        )
        assert response.status_code == 200
        assert response.json()["conversation_id"] == "web-1"

    def test_lista_y_historial(self, client):
        client.post("/ingest/text", json={"text": SOLAR_TEXT, "title": "Solar"})
        client.post("/query", json={"question": "pregunta A", "conversation_id": "web-1"})
        client.post("/query", json={"question": "pregunta B", "conversation_id": "web-1"})
        rows = client.get("/conversations").json()
        assert [row["id"] for row in rows] == ["web-1"]
        assert rows[0]["turns"] == 2
        detail = client.get("/conversations/web-1").json()
        assert detail["turns"] == 2
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user", "assistant"]

    def test_historial_404(self, client):
        assert client.get("/conversations/fantasma").status_code == 404

    def test_borrar_conversacion(self, client):
        client.post("/query", json={"question": "hola", "conversation_id": "web-1"})
        assert client.delete("/conversations/web-1").json() == {"cleared": "web-1"}
        assert client.get("/conversations/web-1").status_code == 404

    def test_borrar_todas(self, client):
        client.post("/query", json={"question": "hola", "conversation_id": "a"})
        client.post("/query", json={"question": "hola", "conversation_id": "b"})
        assert client.delete("/conversations").json() == {"cleared": 2}
        assert client.get("/conversations").json() == []

    def test_stream_conversacional(self, client):
        client.post("/ingest/text", json={"text": SOLAR_TEXT, "title": "Solar"})
        response = client.post(
            "/query/stream", json={"question": "¿cuánto produce?", "conversation_id": "web-2"}
        )
        assert response.status_code == 200
        events = parse_sse(response.text)
        types = [etype for etype, _ in events]
        assert types[0] == "meta" and types[-1] == "done"
        meta = events[0][1]
        assert meta["conversation_id"] == "web-2"
        done = events[-1][1]
        assert done["conversation_id"] == "web-2"
        # el turno quedó registrado y visible en el historial
        assert client.get("/conversations/web-2").json()["turns"] == 1


class TestAPISync:
    def test_registro_y_listado(self, client):
        job = client.post("/sync", json={"source": "text://demo", "interval": 120}).json()
        assert job["source"] == "text://demo"
        assert job["interval"] == 120
        assert job["next_run"] is not None
        listing = client.get("/sync").json()
        assert listing["jobs"][0]["id"] == job["id"]
        assert listing["enabled"] is False  # settings por defecto

    def test_intervalo_invalido_422(self, client):
        response = client.post("/sync", json={"source": "text://demo", "interval": 0.2})
        assert response.status_code == 422

    def test_run_inmediato(self, client):
        job = client.post("/sync", json={"source": "text://demo", "interval": 120}).json()
        outcome = client.post(f"/sync/{job['id']}/run").json()
        assert outcome["job"]["last_status"] == "ok"
        assert outcome["result"]["documents"] == 1
        updated = client.get("/sync").json()["jobs"][0]
        assert updated["runs"] == 1

    def test_run_desconocido_404(self, client):
        assert client.post("/sync/fantasma/run").status_code == 404

    def test_eliminar_job(self, client):
        job = client.post("/sync", json={"source": "text://demo", "interval": 120}).json()
        assert client.delete(f"/sync/{job['id']}").json() == {"deleted": job["id"]}
        assert client.get("/sync").json()["jobs"] == []

    def test_eliminar_desconocido_404(self, client):
        assert client.delete("/sync/fantasma").status_code == 404
