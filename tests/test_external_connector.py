"""Test del conector externo de ejemplo (examples/external_connector)."""

from pathlib import Path

from universal_rag.core.connectors import ConnectorRegistry
from universal_rag.core.pipeline import RAGPipeline


def test_conector_externo_end_to_end(tmp_path):
    examples_dir = Path(__file__).parent.parent / "examples" / "external_connector"
    registry = ConnectorRegistry.discover([str(examples_dir)])
    assert "memo" in registry.names()

    from universal_rag.core.config import Settings

    pipeline = RAGPipeline(
        settings=Settings(store_provider="memory", embedding_provider="hash", llm_provider="echo"),
        registry=registry,
    )
    result = pipeline.ingest("memo://reunion-semanal")
    assert result.documents == 1
    assert result.connector == "memo"

    answer = pipeline.query("¿qué decidió el equipo sobre contenedores?")
    assert answer.citations
    assert answer.citations[0].source == "memo"
    assert answer.citations[0].uri == "memo://reunion-semanal"

    # memo inexistente → error aislado en IngestResult
    failed = pipeline.ingest("memo://no-existe")
    assert failed.documents == 0
    assert failed.errors
