"""Fixtures compartidas de los tests de Universal RAG."""

from __future__ import annotations

import pytest

from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """Settings rápidas y deterministas: memoria + hash + echo."""
    return Settings(
        store_provider="memory",
        embedding_provider="hash",
        embedding_dimensions=128,
        llm_provider="echo",
        chroma_path=tmp_path / "chroma",
        connector_dirs=[],
        top_k=4,
    )


@pytest.fixture()
def pipeline(settings) -> RAGPipeline:
    return RAGPipeline(settings=settings)


SOLAR_TEXT = (
    "La energía solar fotovoltaica convierte la luz del sol en electricidad mediante "
    "celdas de silicio. Un panel doméstico típico de 450 vatios produce entre 1,8 y 2,5 "
    "kWh al día en climas soleados.\n\n"
    "Los sistemas de bombeo solar permiten extraer agua sin baterías: el panel alimenta "
    "directamente la bomba durante las horas de sol y el agua se almacena en un tanque.\n\n"
    "El mantenimiento de un sistema solar es mínimo: limpieza de paneles cada tres meses "
    "y revisión anual del inversor. La vida útil típica supera los 25 años."
)

WIND_TEXT = (
    "La energía eólica aprovecha el viento mediante aerogeneradores. Un aerogenerador "
    "moderno de 3 MW abastece a unas 1.500 familias. Los parques eólicos offshore "
    "producen un 40 por ciento más que los terrestres por la constancia del viento marino."
)


@pytest.fixture()
def pipeline_con_datos(pipeline) -> RAGPipeline:
    """Pipeline con dos documentos ingeridos (solar y eólico)."""
    pipeline.ingest_text(SOLAR_TEXT, title="Guía de energía solar")
    pipeline.ingest_text(WIND_TEXT, title="Manual de energía eólica")
    return pipeline
