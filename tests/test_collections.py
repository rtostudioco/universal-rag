"""Tests de colecciones múltiples (Fase 4)."""

import pytest
from fastapi.testclient import TestClient

from universal_rag.api.app import create_app
from universal_rag.core.collections import CollectionManager, normalize_name


# ------------------------------------------------------------------ nombres
def test_normalize_name_defaults():
    assert normalize_name(None) == "default"
    assert normalize_name("") == "default"
    assert normalize_name(" Legal ") == "Legal"


@pytest.mark.parametrize("bad", ["../etc", "a/b", "colección con espacios", "x" * 65])
def test_normalize_name_invalido(bad):
    with pytest.raises(ValueError):
        normalize_name(bad)


def test_normalize_name_vacio_es_default():
    """El nombre vacío/blanco no es error: normaliza a 'default'."""
    assert normalize_name("") == "default"
    assert normalize_name("   ") == "default"


# ------------------------------------------------------------------ manager
def test_misma_instancia_por_nombre(settings):
    manager = CollectionManager(settings)
    a1 = manager.get("legal")
    a2 = manager.get("legal")
    assert a1 is a2
    assert manager.get("default") is manager.get(None)


def test_colecciones_son_pipelines_distintos(settings):
    manager = CollectionManager(settings)
    assert manager.get("a") is not manager.get("b")
    assert set(manager.names()) >= {"a", "b"}


def test_colecciones_aisladas(settings):
    manager = CollectionManager(settings)
    legal = manager.get("legal")
    faq = manager.get("faq")
    legal.ingest_text("El contrato vence el 30 de junio de 2025.", title="Contrato")
    faq.ingest_text("El horario de atención es de 9 a 17.", title="FAQ")

    assert len(legal.documents()) == 1
    assert len(faq.documents()) == 1
    answer = faq.query("¿cuál es el horario de atención?")
    assert answer.citations
    assert answer.citations[0].uri == "text://FAQ"
    # lo de legal no aparece en faq
    assert all("contrato" not in c.quote.lower() for c in answer.citations)


def test_embedder_compartido_entre_colecciones(settings):
    from universal_rag.core.embeddings import create_embedder

    shared = create_embedder(settings)
    manager = CollectionManager(settings, embedder=shared)
    assert manager.get("x").embedder is shared
    assert manager.get("y").embedder is shared


def test_register_pipelines_preconstruidos(pipeline, settings):
    manager = CollectionManager(settings)
    manager.register("custom", pipeline)
    assert manager.get("custom") is pipeline
    assert "custom" in manager


def test_delete_coleccion(settings):
    manager = CollectionManager(settings)
    legal = manager.get("legal")
    legal.ingest_text("Documento temporal.", title="Tmp")
    assert manager.delete("legal") is True
    assert "legal" not in manager
    assert legal.store.count_chunks() == 0
    assert manager.delete("legal") is False  # ya no existe


def test_list_collections_resumen(settings):
    manager = CollectionManager(settings)
    manager.get("faq").ingest_text("Texto de la faq.", title="F")
    rows = manager.list_collections()
    by_name = {row["name"]: row for row in rows}
    assert by_name["faq"]["documents"] == 1
    assert by_name["faq"]["chunks"] >= 1


def test_settings_coleccion_aisla_chroma_path(tmp_path):
    from universal_rag.core.config import Settings

    settings = Settings(
        store_provider="chroma", chroma_path=tmp_path / "chroma", embedding_provider="hash", llm_provider="echo"
    )
    manager = CollectionManager(settings)
    default_settings = manager.get().settings
    legal_settings = manager.get("legal").settings
    assert str(default_settings.chroma_path) == str(tmp_path / "chroma")
    assert str(legal_settings.chroma_path).endswith("chroma/legal")


def test_nombre_invalido_en_manager(settings):
    manager = CollectionManager(settings)
    with pytest.raises(ValueError):
        manager.get("../escape")


# ----------------------------------------------------------------- API REST
@pytest.fixture()
def api_client(settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def test_api_colecciones_flow(api_client):
    rows = api_client.get("/collections").json()
    assert rows == []

    api_client.post("/ingest/text", json={"text": "Solar es renovable.", "title": "S", "collection": "energy"})
    rows = api_client.get("/collections").json()
    assert [r["name"] for r in rows] == ["energy"]
    assert rows[0]["documents"] == 1

    answer = api_client.post("/query", json={"question": "¿qué es solar?", "collection": "energy"}).json()
    assert answer["citations"]
    # la colección default sigue vacía
    assert api_client.get("/documents").json() == []
    assert api_client.get("/documents?collection=energy").json()


def test_api_delete_coleccion(api_client):
    api_client.post("/ingest/text", json={"text": "Dato.", "title": "D", "collection": "tmp"})
    assert api_client.delete("/collections/tmp").status_code == 200
    assert api_client.get("/collections").json() == []
    assert api_client.delete("/collections/tmp").status_code == 404


def test_api_nombre_coleccion_invalido_422(api_client):
    response = api_client.post(
        "/ingest/text", json={"text": "X", "title": "X", "collection": "../evil"}
    )
    assert response.status_code == 422
