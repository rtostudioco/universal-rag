"""Tests end-to-end del pipeline (ingesta → recuperación → respuesta con citas)."""

from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline
from universal_rag.core.store import InMemoryVectorStore


def test_ingest_text_basico(pipeline):
    result = pipeline.ingest_text("La energía solar es renovable y limpia.", title="Solar")
    assert result.documents == 1
    assert result.chunks == 1
    assert result.connector == "text"
    assert len(result.document_ids) == 1


def test_ingest_text_vacio_se_omite(pipeline):
    result = pipeline.ingest_text("   ", title="Vacío")
    assert result.documents == 0
    assert result.skipped == 1


def test_query_con_citas(pipeline_con_datos):
    answer = pipeline_con_datos.query("¿cada cuánto se limpian los paneles solares?")
    assert answer.context_count >= 1
    assert answer.citations, "debe tener al menos una cita"
    first = answer.citations[0]
    assert first.source == "text"
    assert first.uri.startswith("text://")
    assert first.quote
    assert first.score >= 0
    assert answer.latency_ms >= 0


def test_query_relevancia(pipeline_con_datos):
    answer = pipeline_con_datos.query("¿cuántas familias abastece un aerogenerador?")
    top_quote = answer.citations[0].quote.lower()
    assert "aerogenerador" in top_quote or "familias" in top_quote


def test_query_sin_resultados(pipeline):
    answer = pipeline.query("pregunta sobre un tema inexistente en el índice vacío")
    assert "No encontré" in answer.answer
    assert answer.citations == []


def test_reingesta_reemplaza(pipeline):
    """La misma URI con contenido nuevo reemplaza la versión anterior."""
    first = pipeline.ingest_text("Versión antigua del documento.", title="Doc")
    doc_id = first.document_ids[0]
    second = pipeline.ingest_text("Versión nueva con contenido diferente.", title="Doc")
    assert second.document_ids[0] == doc_id  # misma URI → mismo ID
    assert pipeline.store.count_chunks() == 1  # no duplica
    answer = pipeline.query("versión nueva")
    assert all(c.document_id == doc_id for c in answer.citations)


def test_ingest_fuente_desconocida(pipeline):
    result = pipeline.ingest("ftp://servidor/archivo.xyz")
    assert result.documents == 0
    assert result.errors


def test_ingest_conector_forzado_inexistente(pipeline):
    result = pipeline.ingest("texto cualquiera", connector="no-existe")
    assert result.errors


def test_ingest_archivo_completo(pipeline, tmp_path):
    """Ingesta de un archivo real (ruta) resolviendo el conector file."""
    target = tmp_path / "guia.txt"
    target.write_text("El inversor convierte corriente continua en alterna. Revisión anual.", encoding="utf-8")
    result = pipeline.ingest(str(target))
    assert result.documents == 1
    assert result.connector == "file"
    answer = pipeline.query("¿qué hace el inversor?")
    assert answer.citations
    assert answer.citations[0].source == "file"


def test_where_filter(pipeline_con_datos):
    """El filtro where limita la recuperación por metadatos."""
    from universal_rag.core.utils import make_id

    solar_id = make_id("doc", "text://Guía de energía solar")
    answer = pipeline_con_datos.query("energía", where={"document_id": solar_id})
    assert all(c.document_id == solar_id for c in answer.citations)


def test_delete_document(pipeline_con_datos):
    docs = pipeline_con_datos.documents()
    assert len(docs) == 2
    doc_id = docs[0]["document_id"]
    assert pipeline_con_datos.delete_document(doc_id)
    assert len(pipeline_con_datos.documents()) == 1
    assert not pipeline_con_datos.delete_document(doc_id)  # ya no existe


def test_stats(pipeline_con_datos):
    stats = pipeline_con_datos.stats()
    assert stats["documents"] == 2
    assert stats["store"] == "memory"
    assert stats["embedder"] == "hash"
    assert set(stats["connectors"]) == {"text", "file", "url"}


def test_clear(pipeline_con_datos):
    pipeline_con_datos.clear()
    assert pipeline_con_datos.stats()["documents"] == 0


def test_componentes_inyectados():
    """Se pueden inyectar componentes propios (sin Settings reales)."""
    pipeline = RAGPipeline(
        settings=Settings(store_provider="memory", embedding_provider="hash", llm_provider="echo"),
        store=InMemoryVectorStore(),
    )
    pipeline.ingest_text("Contenido inyectado para probar DI.", title="DI")
    assert pipeline.query("contenido inyectado").citations
