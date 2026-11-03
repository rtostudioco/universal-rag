"""Tests de almacenes vectoriales."""

import pytest

from universal_rag.core.models import Chunk
from universal_rag.core.store import InMemoryVectorStore
from universal_rag.core.utils import make_id


def chunk(doc_id: str, index: int, text: str, source: str = "text") -> Chunk:
    return Chunk(
        id=make_id("chunk", doc_id, index),
        document_id=doc_id,
        text=text,
        index=index,
        total=3,
        metadata={"document_id": doc_id, "source": source, "uri": f"text://{doc_id}", "title": f"Doc {doc_id}"},
    )


@pytest.fixture()
def store() -> InMemoryVectorStore:
    s = InMemoryVectorStore()
    s.add(
        [chunk("d1", 0, "La energía solar produce electricidad"), chunk("d1", 1, "Los paneles necesitan limpieza")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    s.add([chunk("d2", 0, "La cocina italiana usa pasta")], [[0.9, 0.1, 0.0]])
    return s


def test_add_y_count(store):
    assert store.count_chunks() == 3


def test_query_ranking(store):
    results = store.query([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0].chunk.document_id in ("d1", "d2")
    assert results[0].score >= results[1].score
    assert results[0].score > 0.9  # casi idéntico al primero


def test_query_k_mayor_que_contenido(store):
    assert len(store.query([0.0, 0.0, 1.0], k=100)) == 3


def test_query_con_filtro_where(store):
    results = store.query([1.0, 0.0, 0.0], k=10, where={"source": "text"})
    assert len(results) == 3
    results_file = store.query([1.0, 0.0, 0.0], k=10, where={"source": "file"})
    assert results_file == []


def test_delete_document(store):
    removed = store.delete_document("d1")
    assert removed == 2
    assert store.count_chunks() == 1
    assert store.delete_document("no-existe") == 0


def test_list_documents(store):
    docs = store.list_documents()
    assert len(docs) == 2
    by_id = {d["document_id"]: d for d in docs}
    assert by_id["d1"]["chunks"] == 2
    assert by_id["d2"]["chunks"] == 1
    assert by_id["d1"]["title"] == "Doc d1"


def test_clear(store):
    store.clear()
    assert store.count_chunks() == 0
    assert store.list_documents() == []


def test_add_longitud_incorrecta(store):
    with pytest.raises(ValueError):
        store.add([chunk("x", 0, "t"), chunk("x", 1, "u")], [[1.0]])


def test_metadata_plana_en_resultados(store):
    results = store.query([1.0, 0.0, 0.0], k=1)
    meta = results[0].chunk.metadata
    assert meta["source"] == "text"
    assert meta["document_id"] == results[0].chunk.document_id
