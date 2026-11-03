"""Tests del chunker."""

import pytest

from universal_rag.core.chunker import Chunker
from universal_rag.core.models import Document
from universal_rag.core.utils import make_id


def doc(text: str) -> Document:
    return Document(id=make_id("doc", "test"), content=text, source="text", uri="text://test")


def test_texto_corto_un_solo_chunk():
    chunker = Chunker(size=500, overlap=50)
    pieces = chunker.split("Texto corto.")
    assert pieces == ["Texto corto."]


def test_texto_vacio():
    assert Chunker().split("") == []
    assert Chunker().split("   \n\n  ") == []


def test_respetar_tamano_maximo():
    chunker = Chunker(size=300, overlap=40)
    long_text = " ".join(f"Oración número {i} con contenido suficiente para llenar el espacio." for i in range(60))
    pieces = chunker.split(long_text)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)


def test_solapamiento_entre_chunks():
    chunker = Chunker(size=200, overlap=60)
    text = " ".join(f"Palabra{i}" for i in range(200))
    pieces = chunker.split(text)
    assert len(pieces) >= 2
    # el inicio del segundo chunk debe compartir contenido con el final del primero
    tail_first = pieces[0][-40:]
    assert any(word in pieces[1] for word in tail_first.split()[:5])


def test_separacion_por_parrafos():
    chunker = Chunker(size=500, overlap=0)
    text = "Párrafo uno sobre perros.\n\nPárrafo dos sobre gatos.\n\nPárrafo tres sobre aves."
    pieces = chunker.split(text)
    assert len(pieces) == 1  # cabe todo en un chunk
    assert "perros" in pieces[0] and "aves" in pieces[0]


def test_overlap_invalido():
    with pytest.raises(ValueError):
        Chunker(size=100, overlap=100)


def test_chunk_document_metadatos():
    chunker = Chunker(size=120, overlap=20)
    document = Document(
        id="doc123",
        content=" ".join(f"contenido {i} " for i in range(80)),
        source="file",
        uri="file:///x.txt",
        title="X",
        metadata={"custom": "valor"},
    )
    chunks = chunker.chunk_document(document)
    assert all(c.document_id == "doc123" for c in chunks)
    assert all(c.metadata["custom"] == "valor" for c in chunks)
    assert all(c.metadata["source"] == "file" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.total == len(chunks) for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)  # ids únicos
