"""Tests de rerankers pluggables (Fase 3)."""

import math
import sys
import types

import pytest

from universal_rag.core.config import Settings
from universal_rag.core.models import Chunk, RetrievedChunk
from universal_rag.core.rerank import (
    BaseReranker,
    CrossEncoderReranker,
    NoopReranker,
    _sigmoid,
    create_reranker,
)


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(id=f"c{index}", document_id="d1", text=text, index=index, total=3)


@pytest.fixture()
def candidatos() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk=_chunk("texto sin relevancia una", 0), score=0.9),
        RetrievedChunk(chunk=_chunk("el diamante es la gema más dura", 1), score=0.8),
        RetrievedChunk(chunk=_chunk("texto sin relevancia dos", 2), score=0.7),
    ]


def test_sigmoid_valores_conocidos():
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert _sigmoid(-100.0) == pytest.approx(0.0, abs=1e-9)


def test_noop_conserva_orden(candidatos):
    reranker = NoopReranker()
    assert reranker.rerank("consulta", candidatos) == candidatos


def test_base_reranker_contrato(candidatos):
    class _Minimo(BaseReranker):
        name = "minimo"

    assert _Minimo().rerank("consulta", candidatos) == candidatos


def test_cross_encoder_falso_reordena(candidatos, monkeypatch):
    """Inyecta un módulo fake de sentence_transformers (sin descargar modelos)."""
    llamadas: list[list[tuple[str, str]]] = []

    class _FakeCrossEncoder:
        def __init__(self, model_name: str, device=None) -> None:
            self.model_name = model_name

        def predict(self, pairs):
            llamadas.append(list(pairs))
            return [2.0 if "diamante" in text else -2.0 for _, text in pairs]

    fake = types.ModuleType("sentence_transformers")
    fake.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    reranker = CrossEncoderReranker("fake/model")
    reordenados = reranker.rerank("¿qué gema es más dura?", candidatos)

    assert reordenados[0].chunk.text.startswith("el diamante")
    assert reordenados[0].score == pytest.approx(_sigmoid(2.0), abs=1e-4)
    assert all(0.0 <= r.score <= 1.0 for r in reordenados)
    scores = [r.score for r in reordenados]
    assert scores == sorted(scores, reverse=True)  # monótono con el orden
    assert llamadas and llamadas[0][0][0] == "¿qué gema es más dura?"


def test_cross_encoder_carga_perezosa(candidatos, monkeypatch):
    """Si sentence_transformers no está, solo falla al reranquear (no al construir)."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # ImportError forzado
    reranker = CrossEncoderReranker("fake/model")
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        reranker.rerank("consulta", candidatos)


def test_cross_encoder_un_solo_candidato_no_llama_modelo(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker("fake/model")
    unico = [RetrievedChunk(chunk=_chunk("único", 0), score=0.5)]
    assert reranker.rerank("consulta", unico) == unico


def test_create_reranker_fabrica(settings):
    settings.reranker = "none"
    assert isinstance(create_reranker(settings), NoopReranker)
    settings.reranker = "cross-encoder"
    settings.rerank_model = "test/model"
    reranker = create_reranker(settings)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "test/model"


def test_settings_fase3_defaults():
    config = Settings(store_provider="memory", connector_dirs=[])
    assert config.retrieval == "vector"
    assert config.reranker == "none"
    assert config.hybrid_fusion == "rrf"
    assert config.candidate_k == 25
