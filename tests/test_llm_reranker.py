"""Tests del LLMReranker (Fase 5) — endpoint OpenAI-compat mockeado."""

from __future__ import annotations

import json

import httpx

from universal_rag.core.config import Settings
from universal_rag.core.models import Chunk, RetrievedChunk
from universal_rag.core.rerank import LLMReranker, NoopReranker, create_reranker

BASE_URL = "http://llm.test/v1"


def make_chunk(text: str, score: float = 0.1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"c-{text[:8]}", document_id="d1", text=text, index=0, total=1),
        score=score,
    )


def make_client(content: str | None = None, status: int = 200, capture: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["payload"] = json.loads(request.read())
            capture["auth"] = request.headers.get("authorization")
        if status != 200:
            return httpx.Response(status, text="boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_reranker(content: str | None = None, status: int = 200, capture: dict | None = None) -> LLMReranker:
    return LLMReranker(
        base_url=BASE_URL,
        model="rerank-model",
        api_key="sk-test",
        client=make_client(content, status, capture),
    )


# --------------------------------------------------------------------- tests
class TestLLMReranker:
    def test_reordena_por_scores(self):
        chunks = [make_chunk("primero"), make_chunk("segundo"), make_chunk("tercero")]
        content = '[{"i":1,"score":2},{"i":2,"score":9},{"i":3,"score":5}]'
        reordered = make_reranker(content).rerank("consulta", chunks)
        assert [r.chunk.text for r in reordered] == ["segundo", "tercero", "primero"]
        assert [r.score for r in reordered] == [0.9, 0.5, 0.2]

    def test_scores_fuera_de_rango_se_acotan(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        content = '[{"i":1,"score":42},{"i":2,"score":-5}]'
        reordered = make_reranker(content).rerank("q", chunks)
        assert [r.score for r in reordered] == [1.0, 0.0]
        assert [r.chunk.text for r in reordered] == ["a", "b"]

    def test_json_invalido_conserva_orden(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        original = list(chunks)
        reordered = make_reranker("No soy JSON, lo siento").rerank("q", chunks)
        assert [r.chunk.id for r in reordered] == [r.chunk.id for r in original]
        # los scores originales se conservan (no se reescriben)
        assert [r.score for r in reordered] == [r.score for r in original]

    def test_error_http_conserva_orden(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        reordered = make_reranker(status=500).rerank("q", chunks)
        assert [r.chunk.text for r in reordered] == ["a", "b"]

    def test_respuesta_inalida_conserva_orden(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        reordered = make_reranker(None).rerank("q", chunks)  # content None → formato raro
        assert [r.chunk.text for r in reordered] == ["a", "b"]

    def test_cobertura_parcial_conserva_orden(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c"), make_chunk("d")]
        content = '[{"i":2,"score":9}]'  # puntuó solo 1 de 4 (<80%)
        reordered = make_reranker(content).rerank("q", chunks)
        assert [r.chunk.text for r in reordered] == ["a", "b", "c", "d"]

    def test_un_solo_candidato_no_llama_llm(self):
        chunks = [make_chunk("único")]
        capture: dict = {}
        reranker = make_reranker('[{"i":1,"score":1}]', capture=capture)
        reordered = reranker.rerank("q", chunks)
        assert [r.chunk.text for r in reordered] == ["único"]
        assert "payload" not in capture

    def test_prompt_incluye_consulta_y_fragmentos(self):
        capture: dict = {}
        chunks = [make_chunk("fragmento uno"), make_chunk("fragmento dos")]
        make_reranker('[{"i":1,"score":5},{"i":2,"score":6}]', capture=capture).rerank(
            "mi consulta", chunks
        )
        user = capture["payload"]["messages"][1]["content"]
        assert "mi consulta" in user
        assert "[1] fragmento uno" in user
        assert "[2] fragmento dos" in user
        assert capture["payload"]["model"] == "rerank-model"
        assert capture["auth"] == "Bearer sk-test"

    def test_parse_tolerante_a_ruido(self):
        content = 'Claro, aquí va:\n```json\n[{"i": 1, "score": 7}, {"i": 2, "score": 3}]\n```\nSaludos.'
        chunks = [make_chunk("a"), make_chunk("b")]
        reordered = make_reranker(content).rerank("q", chunks)
        assert [r.chunk.text for r in reordered] == ["a", "b"]  # 0.7 > 0.3
        assert [r.score for r in reordered] == [0.7, 0.3]

    def test_items_invalidos_se_ignoran(self):
        chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
        content = (
            '[{"i":"x","score":1},{"i":1,"score":2},{"i":2,"score":9},{"i":9,"score":5},{"i":3,"score":4}]'
        )
        reordered = make_reranker(content).rerank("q", chunks)
        assert [r.chunk.text for r in reordered] == ["b", "c", "a"]


class TestFabricaLLMReranker:
    def test_settings_llm(self):
        settings = Settings(
            store_provider="memory",
            connector_dirs=[],
            reranker="llm",
            llm_base_url="http://llm.test/v1",
            llm_model="gpt-x",
        )
        reranker = create_reranker(settings)
        assert isinstance(reranker, LLMReranker)
        assert reranker.model == "gpt-x"

    def test_settings_llm_modelo_dedicado(self):
        settings = Settings(
            store_provider="memory",
            connector_dirs=[],
            reranker="llm",
            llm_base_url="http://llm.test/v1",
            llm_model="gpt-x",
            rerank_llm_model="rerank-barato",
        )
        assert create_reranker(settings).model == "rerank-barato"

    def test_sin_base_url_cae_a_noop(self):
        settings = Settings(store_provider="memory", connector_dirs=[], reranker="llm")
        assert isinstance(create_reranker(settings), NoopReranker)

    def test_none_sigue_siendo_noop(self):
        settings = Settings(store_provider="memory", connector_dirs=[], reranker="none")
        assert isinstance(create_reranker(settings), NoopReranker)


def test_defaults_settings_fase5():
    settings = Settings(store_provider="memory", connector_dirs=[])
    assert settings.reranker == "none"
    assert settings.rerank_llm_model is None
