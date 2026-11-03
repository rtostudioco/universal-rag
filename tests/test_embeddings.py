"""Tests de embeddings."""

import math

import pytest

from universal_rag.core.config import Settings
from universal_rag.core.embeddings import (
    HashEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


def test_hash_determinista():
    embedder = HashEmbedder(dimensions=128)
    v1 = embedder.embed_one("hola mundo")
    v2 = embedder.embed_one("hola mundo")
    assert v1 == v2
    assert len(v1) == 128


def test_hash_normalizado():
    embedder = HashEmbedder(dimensions=64)
    vector = embedder.embed_one("texto de ejemplo para normalizar")
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) < 1e-6


def test_hash_textos_diferentes_vectors_diferentes():
    embedder = HashEmbedder(dimensions=128)
    a = embedder.embed_one("energía solar fotovoltaica")
    b = embedder.embed_one("recetas de cocina italiana")
    assert a != b


def test_hash_similitud_semantica_superficial():
    """Textos con vocabulario compartido deben ser más parecidos que temas opuestos."""
    embedder = HashEmbedder(dimensions=256)
    solar = embedder.embed_one("la energía solar fotovoltaica genera electricidad limpia")
    solar2 = embedder.embed_one("los paneles de energía solar generan electricidad")
    cocina = embedder.embed_one("la receta de paella lleva arroz azafrán y mariscos")

    def cosine(u, v):
        return sum(x * y for x, y in zip(u, v, strict=False))

    assert cosine(solar, solar2) > cosine(solar, cocina)


def test_batch():
    embedder = HashEmbedder(dimensions=32)
    vectors = embedder.embed(["uno", "dos", "tres"])
    assert len(vectors) == 3 and all(len(v) == 32 for v in vectors)
    assert embedder.embed([]) == []


def test_sentence_transformer_lazy_sin_descargar():
    """Instanciar no debe cargar el modelo (carga perezosa)."""
    embedder = SentenceTransformerEmbedder("paraphrase-multilingual-MiniLM-L12-v2")
    assert embedder._model is None
    assert embedder.dimension == 0


def test_factory_auto_sin_st(tmp_path, monkeypatch):
    """auto + sentence-transformers no instalado → HashEmbedder con warning."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no instalado")
        return real_import(name, *args, **kwargs)

    settings = Settings(embedding_provider="auto", chroma_path=tmp_path)
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", fake_import)
        embedder = create_embedder(settings)
    assert embedder.name == "hash"


def test_factory_forzado_hash(tmp_path):
    settings = Settings(embedding_provider="hash", chroma_path=tmp_path)
    assert isinstance(create_embedder(settings), HashEmbedder)


def test_factory_st_explicito_sin_lib(tmp_path, monkeypatch):
    """provider=sentence-transformers sin la lib instalada debe fallar claro."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no instalado")
        return real_import(name, *args, **kwargs)

    settings = Settings(embedding_provider="sentence-transformers", chroma_path=tmp_path)
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError):
            create_embedder(settings)
