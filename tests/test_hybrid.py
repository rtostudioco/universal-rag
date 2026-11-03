"""Tests de búsqueda híbrida: fusión RRF/weighted + integración en el retriever."""

import pytest

from universal_rag.core.embeddings import BaseEmbedder
from universal_rag.core.hybrid import rrf_fuse, weighted_fuse
from universal_rag.core.pipeline import RAGPipeline


# ------------------------------------------------------------------ fusión RRF
def test_rrf_fusion_basica():
    fused = rrf_fuse([[0, 1], [1]], k=60, top_n=5)
    scores = dict(fused)
    # idx 1 aparece en ambas listas → gana; 1/(60+1) + 1/(60+2)
    assert fused[0][0] == 1
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[0] == pytest.approx(1 / 61)


def test_rrf_rank_vacio():
    assert rrf_fuse([], k=60, top_n=5) == []
    assert rrf_fuse([[], []], k=60, top_n=5) == []


def test_rrf_top_n():
    fused = rrf_fuse([[0, 1, 2], [3, 4]], k=60, top_n=2)
    assert len(fused) == 2


# --------------------------------------------------------- fusión ponderada
def test_weighted_alpha_uno_es_puramente_vectorial():
    vector = [(0, 0.9), (1, 0.5)]
    bm25 = [(1, 5.0), (0, 1.0)]
    fused = weighted_fuse(vector, bm25, alpha=1.0, top_n=5)
    assert fused[0][0] == 0  # orden vectorial


def test_weighted_alpha_cero_es_puramente_lexico():
    vector = [(0, 0.9), (1, 0.5)]
    bm25 = [(1, 5.0), (0, 1.0)]
    fused = weighted_fuse(vector, bm25, alpha=0.0, top_n=5)
    assert fused[0][0] == 1  # orden BM25


def test_weighted_empate_normalizado():
    # ambas listas con un solo candidato → normalizado a 1.0 → suma 1.0
    fused = weighted_fuse([(7, 0.3)], [(7, 4.2)], alpha=0.5, top_n=5)
    assert fused == [(7, pytest.approx(1.0))]


# ------------------------------------------------- integración en el retriever
class _ConstantEmbedder(BaseEmbedder):
    """Embedder degenerado: mismo vector para todo (la señal debe darla BM25)."""

    name = "constant"
    dimension = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 7 for _ in texts]


@pytest.fixture()
def pipeline_hibrido(settings):
    settings.retrieval = "hybrid"
    return RAGPipeline(settings=settings, embedder=_ConstantEmbedder())


def test_hibrido_rescata_por_lexico(pipeline_hibrido):
    """Con embeddings sin señal, el canal léxico (BM25) decide el ranking."""
    pipeline_hibrido.ingest_text("Receta de pan integral con masa madre y fermentación lenta.", title="Cocina")
    pipeline_hibrido.ingest_text("El aerogenerador convierte la energía cinética del viento.", title="Eólica")
    answer = pipeline_hibrido.query("aerogenerador")
    assert answer.citations
    assert answer.citations[0].title == "Eólica"
    assert answer.retrieval == "hybrid"


def test_hibrido_invalidacion_tras_ingesta(pipeline_hibrido):
    pipeline_hibrido.ingest_text("Texto inicial de la base de conocimiento.", title="Inicio")
    # Nueva ingesta: el índice BM25 se invalida y reconstruye automáticamente.
    pipeline_hibrido.ingest_text("La ciudad de Quimbaya es famosa por su festival nacional del café.", title="Quimbaya")
    answer = pipeline_hibrido.query("Quimbaya festival café")
    assert answer.citations
    assert answer.citations[0].title == "Quimbaya"


def test_hibrido_where_filtra_lexico(pipeline_hibrido):
    pipeline_hibrido.ingest_text("Receta de pan integral con masa madre.", title="Cocina")
    pipeline_hibrido.ingest_text("El aerogenerador convierte el viento en energía.", title="Eólica")
    # Filtro por título: BM25 solo ve el subconjunto filtrado.
    answer = pipeline_hibrido.query("aerogenerador", where={"title": "Cocina"})
    assert answer.citations
    assert all(c.title == "Cocina" for c in answer.citations)


def test_hibrido_corpus_vacio_devuelve_vectorial(pipeline_hibrido):
    answer = pipeline_hibrido.query("pregunta sin documentos")
    assert "No encontré" in answer.answer
    assert answer.citations == []


def test_vector_modo_por_defecto_igual_que_fase_1(settings):
    """Sin configuración explícita, la búsqueda es vectorial pura (compatibilidad)."""
    pipeline = RAGPipeline(settings=settings)
    assert pipeline.retrieval_label() == "vector"
    pipeline.ingest_text("La energía solar es renovable y limpia.", title="Solar")
    answer = pipeline.query("energía solar")
    assert answer.citations
    assert answer.citations[0].title == "Solar"
