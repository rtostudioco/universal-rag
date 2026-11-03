"""Tests del streaming (Fase 6): pipeline.stream_query, generadores y API SSE."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import SOLAR_TEXT
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app
from universal_rag.core.config import Settings
from universal_rag.core.generator import OpenAICompatGenerator, tokenize_stream
from universal_rag.core.models import Chunk, RetrievedChunk
from universal_rag.core.pipeline import RAGPipeline

PREGUNTA = "¿Cuánto produce un panel solar al día?"


# ------------------------------------------------------------------ helpers
def make_contexts(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                id=f"c-{i}-{text[:6]}", document_id=f"d{i}", text=text, index=i, total=len(texts)
            ),
            score=0.9 - i * 0.1,
        )
        for i, text in enumerate(texts)
    ]


def make_stream_client(sse_body: bytes, capture: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["payload"] = json.loads(request.read())
            capture["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, content=sse_body, headers={"content-type": "text/event-stream"}
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- tokenizer
class TestTokenizeStream:
    def test_concatenacion_exacta(self):
        text = "Hola mundo.\n\nSegunda línea  con  espacios."
        pieces = list(tokenize_stream(text, words_per_piece=2))
        assert "".join(pieces) == text
        assert len(pieces) > 1

    def test_texto_vacio(self):
        assert list(tokenize_stream("")) == []

    def test_una_palabra(self):
        assert list(tokenize_stream("hola")) == ["hola"]


# ---------------------------------------------------------------- generador
class TestOpenAICompatStreaming:
    SSE = (
        b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" mundo"}}]}\n\n'
        b": keepalive\n\n"
        b'data: {"choices":[{"delta":{}}]}\n\n'
        b'data: [DONE]\n\n'
    )

    def test_stream_sse_parsea_deltas(self):
        capture: dict = {}
        generator = OpenAICompatGenerator(
            base_url="http://llm.test/v1",
            model="gpt-x",
            api_key="sk-test",
            client=make_stream_client(self.SSE, capture),
        )
        pieces = list(generator.stream_generate("q", make_contexts("fragmento uno")))
        assert pieces == ["Hola", " mundo"]
        assert "".join(pieces) == "Hola mundo"
        assert capture["payload"]["stream"] is True
        assert capture["auth"] == "Bearer sk-test"

    def test_stream_sin_datos_cae_a_generate(self):
        # endpoint que ignora stream=true: primero SSE vacío, luego respuesta JSON
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            if body.get("stream"):
                return httpx.Response(200, content=b"event: ping\n\n", headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "respuesta completa"}}]})

        generator = OpenAICompatGenerator(
            base_url="http://llm.test/v1",
            model="gpt-x",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert list(generator.stream_generate("q", make_contexts("ctx"))) == ["respuesta completa"]

    def test_json_ruido_en_lineas_sse_se_ignora(self):
        sse = (
            b'data: no-soy-json\n\n'
            b'data: {"choices": null}\n\n'
            b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        generator = OpenAICompatGenerator(
            base_url="http://llm.test/v1",
            model="m",
            client=make_stream_client(sse),
        )
        assert list(generator.stream_generate("q", make_contexts("ctx"))) == ["OK"]

    def test_model_label_es_el_modelo(self):
        generator = OpenAICompatGenerator(base_url="http://llm.test/v1", model="mi-modelo")
        assert generator.model_label == "mi-modelo"


class TestEchoStreaming:
    def test_stream_en_trozos_concatena_generate(self):
        from universal_rag.core.generator import EchoGenerator

        generator = EchoGenerator()
        contexts = make_contexts("fragmento con contenido", "otro fragmento")
        completo, _ = generator.generate("pregunta", contexts)
        streamed = "".join(generator.stream_generate("pregunta", contexts))
        assert streamed == completo


# ----------------------------------------------------------------- pipeline
class TestStreamQuery:
    @pytest.fixture()
    def pipeline(self, tmp_path) -> RAGPipeline:
        settings = Settings(
            store_provider="memory",
            embedding_provider="hash",
            embedding_dimensions=128,
            llm_provider="echo",
            chroma_path=tmp_path / "chroma",
            connector_dirs=[],
            top_k=4,
        )
        pipeline = RAGPipeline(settings=settings)
        pipeline.ingest_text(SOLAR_TEXT, title="Guía solar")
        return pipeline

    def test_secuencia_eventos_meta_tokens_done(self, pipeline):
        events = list(pipeline.stream_query(PREGUNTA))
        types = [event["type"] for event in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert types.count("done") == 1
        assert types.count("meta") == 1
        assert types[1:-1] == ["token"] * (len(types) - 2)
        assert len(types) > 3  # hay tokens de verdad

    def test_tokens_concatenados_son_la_respuesta_final(self, pipeline):
        events = list(pipeline.stream_query(PREGUNTA))
        tokens = "".join(e["value"] for e in events if e["type"] == "token")
        done = events[-1]["answer"]
        assert tokens == done["answer"]

    def test_meta_trae_fuentes_y_done_citas(self, pipeline):
        events = list(pipeline.stream_query(PREGUNTA))
        meta = events[0]
        assert meta["cached"] is False
        assert meta["context_count"] > 0
        assert meta["sources"], "el evento meta debe traer preview de fuentes"
        assert all("title" in source or "uri" in source for source in meta["sources"])
        done = events[-1]["answer"]
        assert done["citations"], "echo cita [1] → el Answer final lleva citas"
        assert done["retrieval"] == meta["retrieval"]

    def test_segunda_consulta_es_cache_hit(self, pipeline):
        list(pipeline.stream_query(PREGUNTA))
        events = list(pipeline.stream_query(PREGUNTA))
        assert events[0]["cached"] is True
        done = events[-1]["answer"]
        assert done["cached"] is True

    def test_sin_contexto_no_emite_tokens(self, tmp_path):
        settings = Settings(
            store_provider="memory",
            embedding_provider="hash",
            embedding_dimensions=128,
            llm_provider="echo",
            chroma_path=tmp_path / "chroma",
            connector_dirs=[],
        )
        pipeline = RAGPipeline(settings=settings)  # sin documentos
        events = list(pipeline.stream_query("¿qué es lo que sea?"))
        types = [event["type"] for event in events]
        assert "token" not in types
        assert events[0]["type"] == "meta"
        assert events[-1]["type"] == "done"
        assert "No encontré" in events[-1]["answer"]["answer"]

    def test_generador_custom_palabra_a_palabra(self, pipeline):
        class WordGenerator:
            name = "word"

            @property
            def model_label(self):
                return "word"

            def generate(self, question, contexts):
                return "uno dos tres", self.name

            def stream_generate(self, question, contexts):
                yield from ["uno ", "dos ", "tres"]

        pipeline.generator = WordGenerator()
        events = list(pipeline.stream_query(PREGUNTA))
        tokens = [e["value"] for e in events if e["type"] == "token"]
        assert tokens == ["uno ", "dos ", "tres"]
        assert events[-1]["answer"]["model"] == "word"


# --------------------------------------------------------------------- API
class TestQueryStreamAPI:
    @pytest.fixture()
    def client(self, settings) -> TestClient:
        app = create_app(settings)
        with TestClient(app) as test_client:
            test_client.post("/ingest/text", json={"text": SOLAR_TEXT, "title": "Solar"})
            yield test_client

    def _parse_sse(self, body: str) -> list[tuple[str, object]]:
        events: list[tuple[str, object]] = []
        for block in body.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            event, data = "message", ""
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data += line[5:].strip()
            events.append((event, json.loads(data) if data else None))
        return events

    def test_sse_formato_y_secuencia(self, client):
        response = client.post("/query/stream", json={"question": PREGUNTA})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        events = self._parse_sse(response.text)
        types = [event for event, _ in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert "token" in types
        # done es un Answer serializable completo
        done = dict(events[-1][1])
        assert done["question"] == PREGUNTA
        assert isinstance(done["citations"], list)

    def test_sse_requiere_payload_valido(self, client):
        response = client.post("/query/stream", json={"question": ""})
        assert response.status_code == 422

    def test_sse_nombre_coleccion_invalido_es_422(self, client):
        # las colecciones válidas se auto-crean; un nombre inválido sí es error
        response = client.post("/query/stream", json={"question": PREGUNTA, "collection": "../evil"})
        assert response.status_code == 422

    def test_stats_cuenta_queries_stream(self, client):
        client.post("/query/stream", json={"question": PREGUNTA})
        runtime = client.get("/stats").json()["runtime"]
        assert runtime.get("queries_stream", 0) >= 1

    def test_raiz_sirve_workbench(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        body = response.text
        assert "Universal RAG" in body
        assert "/query/stream" in body
        assert "/documents" in body

    def test_chat_clasico_sigue_disponible(self, client):
        response = client.get("/chat")
        assert response.status_code == 200
        assert "/query" in response.text
