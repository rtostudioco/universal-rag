"""Tests del agente multi-hop (Fase 8)."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from universal_rag.api.app import create_app
from universal_rag.cli import app as cli_app
from universal_rag.core.agent import (
    RAGAgent,
    content_words_query,
    heuristic_plan,
)
from universal_rag.core.generator import OpenAICompatGenerator
from universal_rag.core.models import AgentAnswer
from universal_rag.core.pipeline import RAGPipeline
from universal_rag.mcp_server import create_mcp_server

runner = CliRunner()


# ------------------------------------------------------------------ planner puro
def test_heuristic_plan_split_por_interrogaciones():
    plan = heuristic_plan("¿Qué es la energía solar? ¿Y el bombeo? ¿Y el mantenimiento?", 3)
    assert len(plan) == 3
    assert all("solar" in p or "bombeo" in p or "mantenimiento" in p for p in plan)


def test_heuristic_plan_conjuncion_y():
    plan = heuristic_plan("Mantenimiento de paneles solares y vida útil del inversor", 3)
    assert len(plan) == 2
    assert any("paneles" in p for p in plan)
    assert any("inversor" in p for p in plan)


def test_heuristic_plan_pregunta_simple_sin_partir():
    question = "¿Qué es la energía solar fotovoltaica?"
    assert heuristic_plan(question, 3) == [question]


def test_heuristic_plan_fusiona_fragmentos_cortos():
    plan = heuristic_plan("Guía de energía solar\nmás\nManual de energía eólica", 3)
    # "más" (< 3 palabras) se fusiona con el fragmento anterior
    assert len(plan) == 2
    assert plan[0].endswith("más")


def test_heuristic_plan_respeta_max_steps():
    question = "A uno? B dos? C tres? D cuatro? E cinco?"
    plan = heuristic_plan(question, 2)
    assert len(plan) <= 2


def test_heuristic_plan_vacio():
    assert heuristic_plan("", 3) == [""]


def test_content_words_query_quita_stopwords():
    assert content_words_query("¿Qué es la energía solar?") == "energía solar"
    assert content_words_query("How does the wind turbine work?") == "wind turbine work"


def test_content_words_query_sin_contenido_devuelve_none():
    assert content_words_query("¿Qué es?") is None


def test_parse_json_list_tolerante():
    from universal_rag.core.agent import _parse_json_list

    assert _parse_json_list('["a b", "c d"]') == ["a b", "c d"]
    assert _parse_json_list('Texto previo ["a b"] texto final') == ["a b"]
    assert _parse_json_list("no hay json") is None
    assert _parse_json_list('{"no": "es lista"}') is None
    assert _parse_json_list('["", "  "]') is None


# ----------------------------------------------------------------- agente + echo
def test_plan_heuristico_con_pipeline_echo(pipeline):
    agent = RAGAgent(pipeline, planning="heuristic")
    sub_queries, planner = agent.plan("¿Qué es la energía solar?")
    assert planner == "heuristic"
    assert sub_queries == ["¿Qué es la energía solar?"]


def test_auto_sin_llm_cae_en_heuristico(pipeline):
    agent = RAGAgent(pipeline)  # planning="auto", generador echo sin chat()
    _, planner = agent.plan("Pregunta cualquiera con suficiente contenido")
    assert planner == "heuristic"


def test_ask_end_to_end_con_datos(pipeline_con_datos):
    agent = pipeline_con_datos.agent
    result = agent.ask("Mantenimiento de paneles solares y vida útil del inversor")
    assert isinstance(result, AgentAnswer)
    assert result.sub_queries and len(result.sub_queries) == 2
    actions = [step.action for step in result.steps]
    assert actions[0] == "plan"
    assert actions.count("search") == 2
    assert actions[-1] == "synthesize"
    assert result.context_count > 0
    assert result.citations
    assert result.model == "echo"
    assert result.retrieval == "vector"
    assert result.latency_ms >= 0


def test_ask_dedupe_contextos_repetidos(pipeline_con_datos, monkeypatch):
    # Dos sub-consultas distintas que caen en el mismo fragmento → 1 contexto único.
    same = pipeline_con_datos.retriever.retrieve("energía solar", k=1)
    assert same, "el corpus debería devolver el top-1"
    seen: list[str] = []

    def fake_retrieve(question, k=None, where=None):
        seen.append(question)
        return list(same)

    monkeypatch.setattr(pipeline_con_datos.retriever, "retrieve", fake_retrieve)
    result = pipeline_con_datos.agent.ask("Primera sub-consulta sobre solar y segunda sub-consulta solar")
    assert len(seen) == 2
    assert result.context_count == 1


def test_ask_refina_subconsulta_sin_resultados(pipeline_con_datos, monkeypatch):
    raw = "consulta sin resultados garantizados"

    original = pipeline_con_datos.retriever.retrieve

    def fake_retrieve(question, k=None, where=None):
        if question == raw:
            return []
        return original(question, k=k, where=where)

    monkeypatch.setattr(pipeline_con_datos.retriever, "retrieve", fake_retrieve)
    agent = RAGAgent(pipeline_con_datos, planning="heuristic")
    result = agent.ask(raw)
    search = next(step for step in result.steps if step.action == "search")
    assert search.note and "reformulada" in search.note  # raw devolvía 0, reformulada recuperó
    assert search.found > 0
    assert result.context_count > 0


def test_ask_refino_global_cuando_todo_vacio(pipeline, monkeypatch):
    raw = "nada que ver con el indice vacio"
    calls: list[str] = []

    def fake_retrieve(question, k=None, where=None):
        calls.append(question)
        return []

    monkeypatch.setattr(pipeline.retriever, "retrieve", fake_retrieve)
    result = pipeline.agent.ask(raw)
    assert result.context_count == 0
    assert "No encontré información" in result.answer
    assert result.citations == []
    assert any(step.action == "refine" for step in result.steps)


def test_ask_respeta_k_y_dedupe_por_score(pipeline_con_datos):
    result = pipeline_con_datos.agent.ask("energía solar", k=2)
    assert result.context_count <= 2


def test_stream_ask_orden_de_eventos(pipeline_con_datos):
    events = list(
        pipeline_con_datos.stream_agent_ask(
            "Mantenimiento de paneles solares y vida útil del inversor"
        )
    )
    types = [event["type"] for event in events]
    assert types[0] == "plan"
    assert types.count("step") >= 2
    assert "synthesize" in types
    assert "token" in types
    assert types[-1] == "done"
    plan_event = events[0]
    assert plan_event["planner"] == "heuristic"
    assert len(plan_event["sub_queries"]) == 2
    done = events[-1]["answer"]
    assert done["context_count"] > 0
    assert done["citations"]
    # los tokens reproducen la respuesta final
    tokens = "".join(e["value"] for e in events if e["type"] == "token")
    assert tokens == done["answer"]


# ------------------------------------------------------------- planificación LLM
def _mock_llm(transport: httpx.MockTransport, model: str = "planner-x") -> OpenAICompatGenerator:
    return OpenAICompatGenerator(
        base_url="http://fake.local/v1",
        model=model,
        client=httpx.Client(transport=transport),
    )


def _llm_pipeline(settings, generator) -> RAGPipeline:
    return RAGPipeline(settings=settings, generator=generator)


def test_plan_llm_devuelve_subconsultas(settings, pipeline, monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '["definicion de energia solar", "mantenimiento de paneles"]'}}
                ]
            },
        )

    llm = _mock_llm(httpx.MockTransport(handler))
    pipeline_llm = _llm_pipeline(settings, llm)
    agent = RAGAgent(pipeline_llm)  # planning="auto" → detecta chat()
    sub_queries, planner = agent.plan("energia solar")
    assert sub_queries == ["definicion de energia solar", "mantenimiento de paneles"]
    assert planner == "planner-x"


def test_plan_llm_falla_y_cae_en_heuristico(settings, pipeline):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    llm = _mock_llm(httpx.MockTransport(handler))
    pipeline_llm = _llm_pipeline(settings, llm)
    agent = RAGAgent(pipeline_llm)
    sub_queries, planner = agent.plan("¿Qué es la energía solar fotovoltaica?")
    assert planner == "heuristic"
    assert sub_queries == ["¿Qué es la energía solar fotovoltaica?"]


def test_plan_llm_respuesta_no_json_cae_en_heuristico(settings, pipeline):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "lo siento, no sé"}}]})

    llm = _mock_llm(httpx.MockTransport(handler))
    pipeline_llm = _llm_pipeline(settings, llm)
    agent = RAGAgent(pipeline_llm)
    _, planner = agent.plan("¿Qué es la energía solar fotovoltaica?")
    assert planner == "heuristic"


def test_plan_llm_dedupe_y_cap(settings, pipeline):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '["una consulta", "Una Consulta", "dos consulta", "tres consulta", "cuatro"]'
                        }
                    }
                ]
            },
        )

    llm = _mock_llm(httpx.MockTransport(handler))
    pipeline_llm = _llm_pipeline(settings, llm)
    agent = RAGAgent(pipeline_llm, max_steps=2)
    sub_queries, _ = agent.plan("pregunta")
    assert sub_queries == ["una consulta", "dos consulta"]


def test_ask_llm_planes_y_sintesis_llm(settings, pipeline_con_datos):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        if "planificador" in system:
            content = '["energia solar definicion", "vida util paneles"]'
        else:
            content = "Síntesis final con [1] y [2]."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(httpx.MockTransport(handler))
    pipeline_llm = _llm_pipeline(settings, llm)
    pipeline_llm.ingest_text("La energía solar convierte luz en electricidad.", title="Solar")
    pipeline_llm.ingest_text("El mantenimiento de paneles solares revisa el inversor cada año.", title="Mantenimiento")
    result = pipeline_llm.agent.ask("energia solar y vida util")
    assert calls["n"] == 2  # plan (1) + síntesis (1); las búsquedas no llaman al LLM
    assert result.planner == "planner-x"
    assert result.answer == "Síntesis final con [1] y [2]."
    assert result.context_count == 2
    assert len(result.citations) == 2


def test_heuristico_forzado_con_llm_disponible(settings, pipeline_con_datos, monkeypatch):
    def broken(*args, **kwargs):  # pragma: no cover — no debe llamarse
        raise AssertionError("planning=heuristic no debe llamar al LLM")

    monkeypatch.setattr("universal_rag.core.agent.RAGAgent._llm_plan", broken)
    agent = RAGAgent(pipeline_con_datos, planning="heuristic")
    sub_queries, planner = agent.plan("¿Qué es la energía solar?")
    assert planner == "heuristic"


# ----------------------------------------------------------------------- API
@pytest.fixture()
def client(pipeline_con_datos, settings) -> TestClient:
    app = create_app(settings)
    app.state.collections.register(settings.default_collection, pipeline_con_datos)
    with TestClient(app) as test_client:
        yield test_client


def test_api_query_agent(client):
    response = client.post(
        "/query/agent",
        json={"question": "Mantenimiento de paneles solares y vida útil del inversor"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sub_queries"]
    assert data["steps"][0]["action"] == "plan"
    assert data["steps"][-1]["action"] == "synthesize"
    assert data["citations"]
    assert data["context_count"] > 0


def test_api_query_agent_parametros(client):
    response = client.post(
        "/query/agent",
        json={"question": "energía solar", "max_steps": 1, "planning": "heuristic", "k": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["sub_queries"]) == 1
    assert data["context_count"] <= 2


def test_api_query_agent_validacion(client):
    response = client.post("/query/agent", json={"question": "x", "max_steps": 99})
    assert response.status_code == 422


def test_api_query_agent_stream_sse(client):
    with client.stream(
        "POST",
        "/query/agent/stream",
        json={"question": "Mantenimiento solar y vida útil del inversor"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in response.iter_text())
    events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
    assert events[0] == "plan"
    assert "step" in events
    assert "synthesize" in events
    assert "token" in events
    assert events[-1] == "done"


def test_api_metricas_agente(client):
    client.post("/query/agent", json={"question": "energía solar"})
    client.post("/query/agent/stream", json={"question": "energía eólica"})
    snapshot = client.get("/stats").json()["runtime"]
    assert snapshot["queries_agent"] == 1
    assert snapshot["queries_agent_stream"] == 1


# ----------------------------------------------------------------------- MCP
def test_mcp_rag_ask_tool(settings, pipeline_con_datos):
    from universal_rag.core.collections import CollectionManager

    manager = CollectionManager(settings)
    manager.register(settings.default_collection, pipeline_con_datos)
    mcp = create_mcp_server(manager)
    import asyncio

    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "rag_ask" in tools
    result = asyncio.run(
        mcp.call_tool(
            "rag_ask",
            {"question": "Mantenimiento solar y vida útil del inversor", "max_steps": 2},
        )
    )
    content = result[0][0].text
    data = json.loads(content)
    assert data["sub_queries"]
    assert data["citations"]


# ----------------------------------------------------------------------- CLI
def test_cli_ask_muestra_plan_y_respuesta(settings, pipeline_con_datos, monkeypatch):
    from universal_rag.core.collections import CollectionManager

    manager = CollectionManager(settings)
    manager.register(settings.default_collection, pipeline_con_datos)
    monkeypatch.setattr(
        "universal_rag.cli._pipeline",
        lambda collection=None: manager.get(collection),
    )
    result = runner.invoke(
        cli_app,
        ["ask", "Mantenimiento de paneles solares y vida útil del inversor"],
    )
    assert result.exit_code == 0
    assert "Plan del agente" in result.output
    assert "heuristic" in result.output
    assert "Agente ·" in result.output


def test_cli_ask_quiet(settings, pipeline_con_datos, monkeypatch):
    from universal_rag.core.collections import CollectionManager

    manager = CollectionManager(settings)
    manager.register(settings.default_collection, pipeline_con_datos)
    monkeypatch.setattr(
        "universal_rag.cli._pipeline",
        lambda collection=None: manager.get(collection),
    )
    result = runner.invoke(cli_app, ["ask", "energía solar", "--quiet"])
    assert result.exit_code == 0
    assert "Plan del agente" not in result.output


def test_cli_ask_planning_invalido(settings, pipeline_con_datos, monkeypatch):
    from universal_rag.core.collections import CollectionManager

    manager = CollectionManager(settings)
    manager.register(settings.default_collection, pipeline_con_datos)
    monkeypatch.setattr(
        "universal_rag.cli._pipeline",
        lambda collection=None: manager.get(collection),
    )
    result = runner.invoke(cli_app, ["ask", "pregunta", "--planning", "magia"])
    assert result.exit_code == 1


# ------------------------------------------------------------------- pipeline
def test_pipeline_agent_ask_y_build_citations_publico(pipeline_con_datos):
    result = pipeline_con_datos.agent_ask("energía solar")
    assert isinstance(result, AgentAnswer)
    contexts = pipeline_con_datos.retriever.retrieve("energía solar", k=1)
    citations = pipeline_con_datos.build_citations("texto con [1]", contexts)
    assert citations and citations[0].chunk_id == contexts[0].chunk.id


def test_agente_es_perezoso_y_estable(pipeline):
    assert pipeline._agent is None
    first = pipeline.agent
    assert pipeline.agent is first
