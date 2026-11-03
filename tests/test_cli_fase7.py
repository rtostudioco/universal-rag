"""Tests de los comandos CLI de la Fase 7 (sync + conversations + query --conversation)."""

from __future__ import annotations

import pytest
from conftest import SOLAR_TEXT
from typer.testing import CliRunner

from universal_rag.cli import app

runner = CliRunner()


@pytest.fixture()
def pipeline_con_datos(settings, tmp_path, monkeypatch):
    from universal_rag.core.pipeline import RAGPipeline

    monkeypatch.setenv("RAG_SYNC_STATE_PATH", str(tmp_path / "sync_state.json"))
    pipeline = RAGPipeline(settings=settings)
    pipeline.ingest_text(SOLAR_TEXT, title="Solar")
    return pipeline


class TestSync:
    def test_add_y_list(self, pipeline_con_datos, monkeypatch):
        monkeypatch.setattr("universal_rag.cli._pipeline", lambda collection=None: pipeline_con_datos)
        result = runner.invoke(app, ["sync", "add", "text://demo", "--interval", "120"])
        assert result.exit_code == 0
        assert "Sincronización" in result.output
        listing = runner.invoke(app, ["sync", "list"])
        assert listing.exit_code == 0
        assert "text://demo" in listing.output
        assert "120s" in listing.output

    def test_add_sin_intervalo_valido(self, pipeline_con_datos, monkeypatch):
        monkeypatch.setattr("universal_rag.cli._pipeline", lambda collection=None: pipeline_con_datos)
        result = runner.invoke(app, ["sync", "add", "text://demo", "--interval", "0"])
        assert result.exit_code != 0

    def test_run_y_remove(self, pipeline_con_datos, monkeypatch):
        monkeypatch.setattr("universal_rag.cli._pipeline", lambda collection=None: pipeline_con_datos)
        runner.invoke(app, ["sync", "add", "text://demo", "--interval", "120"])
        listing = runner.invoke(app, ["sync", "list"]).output
        job_id = listing.split("\n")[4].split("│")[1].strip()  # primera fila de datos
        run = runner.invoke(app, ["sync", "run", job_id])
        assert run.exit_code == 0
        assert "ok" in run.output
        removed = runner.invoke(app, ["sync", "remove", job_id])
        assert removed.exit_code == 0
        missing = runner.invoke(app, ["sync", "remove", job_id])
        assert missing.exit_code == 1


class TestConversaciones:
    def test_query_con_conversacion_y_listado(self, pipeline_con_datos, monkeypatch):
        monkeypatch.setattr("universal_rag.cli._pipeline", lambda collection=None: pipeline_con_datos)
        result = runner.invoke(app, ["query", "¿qué es la energía solar?", "--conversation", "cli-test"])
        assert result.exit_code == 0
        listing = runner.invoke(app, ["conversations"])
        assert listing.exit_code == 0
        assert "cli-test" in listing.output

    def test_show_historial(self, pipeline_con_datos, monkeypatch):
        monkeypatch.setattr("universal_rag.cli._pipeline", lambda collection=None: pipeline_con_datos)
        runner.invoke(app, ["query", "¿qué es la energía solar?", "--conversation", "cli-test"])
        shown = runner.invoke(app, ["conversations", "--show", "cli-test"])
        assert shown.exit_code == 0
        assert "Tú" in shown.output and "RAG" in shown.output

    def test_clear(self, pipeline_con_datos, monkeypatch):
        monkeypatch.setattr("universal_rag.cli._pipeline", lambda collection=None: pipeline_con_datos)
        runner.invoke(app, ["query", "pregunta", "--conversation", "cli-test"])
        cleared = runner.invoke(app, ["conversations", "--clear", "all"])
        assert cleared.exit_code == 0
        assert "1" in cleared.output
        assert "Sin conversaciones" in runner.invoke(app, ["conversations"]).output
