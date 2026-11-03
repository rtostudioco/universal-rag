"""Tests de la caché semántica de consultas (Fase 6)."""

from __future__ import annotations

import time

import pytest

from universal_rag.core.cache import SemanticCache, cosine_similarity
from universal_rag.core.config import Settings
from universal_rag.core.embeddings import BaseEmbedder
from universal_rag.core.pipeline import RAGPipeline

SOLAR = (
    "La energía solar fotovoltaica convierte la luz del sol en electricidad mediante "
    "celdas de silicio. Un panel doméstico de 450 vatios produce entre 1,8 y 2,5 kWh al día."
)

EOLICA = (
    "La energía eólica aprovecha el viento mediante aerogeneradores de 3 MW que "
    "abastecen a unas 1.500 familias cada uno."
)

PREGUNTA = "¿Cuánto produce un panel solar al día?"


# ------------------------------------------------------------------ helpers
class FixedEmbedder(BaseEmbedder):
    """Embedder determinista por diccionario (para probar umbrales con precisión)."""

    name = "fixed"

    def __init__(self, vectors: dict[str, list[float]], dimension: int = 2) -> None:
        self.vectors = vectors
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vectors[text]) for text in texts]


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        store_provider="memory",
        embedding_provider="hash",
        embedding_dimensions=128,
        llm_provider="echo",
        chroma_path=tmp_path / "chroma",
        connector_dirs=[],
        top_k=4,
    )
    base.update(overrides)
    return Settings(**base)


# ------------------------------------------------------------------ coseno
class TestCosineSimilarity:
    def test_identicos_es_uno(self):
        assert cosine_similarity([1.0, 0.0, 2.0], [1.0, 0.0, 2.0]) == pytest.approx(1.0)

    def test_ortogonales_es_cero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opuestos_es_menos_uno(self):
        assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)

    def test_vacias_o_dim_distinta_es_cero(self):
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_vector_cero_es_cero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ------------------------------------------------------------------ caché
class TestSemanticCache:
    def test_put_y_lookup_exacto(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0]})
        cache = SemanticCache(embedder, threshold=0.9)
        cache.put("q1", {"answer": "hola"})
        assert cache.lookup("q1") == {"answer": "hola"}
        assert cache.stats()["hits"] == 1

    def test_lookup_similar_por_coseno(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0], "q2": [0.99, 0.14]})
        cache = SemanticCache(embedder, threshold=0.9)
        cache.put("q1", {"answer": "A"})
        # coseno([1,0],[0.99,0.14]) ≈ 0.99 → acierto
        assert cache.lookup("q2") == {"answer": "A"}

    def test_misses_por_debajo_del_umbral(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0], "otra": [0.0, 1.0]})
        cache = SemanticCache(embedder, threshold=0.9)
        cache.put("q1", {"answer": "A"})
        assert cache.lookup("otra") is None
        assert cache.stats()["misses"] == 1

    def test_umbral_uno_solo_acepta_identicas(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0], "q2": [0.99, 0.14]})
        cache = SemanticCache(embedder, threshold=1.0)
        cache.put("q1", {"answer": "A"})
        assert cache.lookup("q2") is None
        assert cache.lookup("q1") == {"answer": "A"}

    def test_lru_evict(self):
        # vectores ortogonales: cada pregunta solo matchea su propia entrada
        embedder = FixedEmbedder(
            {"q1": [1.0, 0.0], "q2": [0.0, 1.0], "q3": [-1.0, 0.0]}, dimension=2
        )
        cache = SemanticCache(embedder, threshold=0.9, max_entries=2)
        cache.put("q1", {"answer": "1"})
        cache.put("q2", {"answer": "2"})
        cache.put("q3", {"answer": "3"})  # expulsa a q1
        assert cache.lookup("q1") is None
        assert cache.lookup("q2") == {"answer": "2"}
        assert cache.lookup("q3") == {"answer": "3"}
        assert cache.stats()["entries"] == 2

    def test_ttl_expira(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0]})
        cache = SemanticCache(embedder, threshold=0.9, ttl_seconds=0.01)
        cache.put("q1", {"answer": "A"})
        time.sleep(0.03)
        assert cache.lookup("q1") is None

    def test_clear_vacia(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0]})
        cache = SemanticCache(embedder)
        cache.put("q1", {"answer": "A"})
        cache.clear()
        assert cache.lookup("q1") is None
        assert cache.stats()["entries"] == 0

    def test_preguntas_vacias_se_ignoran(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0]})
        cache = SemanticCache(embedder)
        cache.put("  ", {"answer": "A"})
        assert cache.lookup("") is None
        assert cache.stats()["entries"] == 0

    def test_hit_rate(self):
        embedder = FixedEmbedder({"q1": [1.0, 0.0], "q2": [0.0, 1.0]})
        cache = SemanticCache(embedder, threshold=0.9)
        cache.put("q1", {"answer": "A"})
        cache.lookup("q1")  # hit
        cache.lookup("q2")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# --------------------------------------------------------------- pipeline
class TestCacheEnPipeline:
    def test_segunda_consulta_igual_viene_de_cache(self, tmp_path):
        pipeline = RAGPipeline(settings=make_settings(tmp_path))
        pipeline.ingest_text(SOLAR, title="Solar")
        primera = pipeline.query(PREGUNTA)
        assert primera.cached is False
        segunda = pipeline.query(PREGUNTA)
        assert segunda.cached is True
        assert segunda.answer == primera.answer
        assert segunda.citations == primera.citations
        cache_stats = pipeline.stats()["cache"]
        assert cache_stats["enabled"] is True
        assert cache_stats["hits"] == 1
        assert cache_stats["misses"] == 1

    def test_ingesta_invalida_la_cache(self, tmp_path):
        pipeline = RAGPipeline(settings=make_settings(tmp_path))
        pipeline.ingest_text(SOLAR, title="Solar")
        assert pipeline.query(PREGUNTA).cached is False
        assert pipeline.query(PREGUNTA).cached is True
        pipeline.ingest_text(EOLICA, title="Eólica")  # cambia el índice
        assert pipeline.query(PREGUNTA).cached is False

    def test_delete_y_clear_invalidan(self, tmp_path):
        pipeline = RAGPipeline(settings=make_settings(tmp_path))
        result = pipeline.ingest_text(SOLAR, title="Solar")
        pipeline.query(PREGUNTA)
        doc_id = result.document_ids[0]
        assert pipeline.delete_document(doc_id) is True
        assert pipeline.cache is not None
        assert pipeline.cache.stats()["entries"] == 0
        pipeline.clear()
        assert pipeline.cache.stats()["entries"] == 0

    def test_cache_desactivada(self, tmp_path):
        pipeline = RAGPipeline(settings=make_settings(tmp_path, cache_enabled=False))
        assert pipeline.cache is None
        pipeline.ingest_text(SOLAR, title="Solar")
        assert pipeline.query(PREGUNTA).cached is False
        assert pipeline.query(PREGUNTA).cached is False
        assert pipeline.stats()["cache"] == {"enabled": False}

    def test_k_custom_bypassa_cache(self, tmp_path):
        pipeline = RAGPipeline(settings=make_settings(tmp_path))
        pipeline.ingest_text(SOLAR, title="Solar")
        pipeline.query(PREGUNTA)
        # con k explícito la caché no aplica (podría cambiar el resultado)
        assert pipeline.query(PREGUNTA, k=2).cached is False

    def test_ttl_settings(self, tmp_path):
        pipeline = RAGPipeline(settings=make_settings(tmp_path, cache_ttl=0.01))
        pipeline.ingest_text(SOLAR, title="Solar")
        primera = pipeline.query(PREGUNTA)
        assert primera.cached is False
        time.sleep(0.03)
        assert pipeline.query(PREGUNTA).cached is False

    def test_threshold_settings_fase6(self, tmp_path):
        settings = make_settings(tmp_path)
        assert settings.cache_enabled is True
        assert settings.cache_threshold == 0.93
        assert settings.cache_max_entries == 512
        assert settings.cache_ttl is None
