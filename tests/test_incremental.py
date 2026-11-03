"""Tests de ingesta incremental (Fase 4): hash de contenido y saltos."""

from universal_rag.core.utils import content_fingerprint


def test_ingesta_repetida_es_unchanged(pipeline):
    r1 = pipeline.ingest_text("Contenido estable para el test incremental.", title="Incremental")
    assert r1.documents == 1
    chunks_despues_de_r1 = pipeline.store.count_chunks()

    r2 = pipeline.ingest_text("Contenido estable para el test incremental.", title="Incremental")
    assert r2.unchanged == 1
    assert r2.documents == 0
    assert pipeline.store.count_chunks() == chunks_despues_de_r1  # no se re-vectorizó


def test_force_reingesta(pipeline):
    pipeline.ingest_text("Texto con force.", title="Force")
    r = pipeline.ingest_text("Texto con force.", title="Force", force=True)
    assert r.documents == 1
    assert r.unchanged == 0


def test_contenido_distinto_reingesta(pipeline, tmp_path):
    """Misma URI (mismo archivo), contenido cambiado → re-ingesta."""
    target = tmp_path / "nota.md"
    target.write_text("Primera versión del contenido.", encoding="utf-8")
    r1 = pipeline.ingest(str(target))
    assert r1.documents == 1

    target.write_text("Segunda versión, distinta.", encoding="utf-8")
    r2 = pipeline.ingest(str(target))
    assert r2.documents == 1
    assert r2.unchanged == 0

    # y una tercera ingesta sin cambios → unchanged
    r3 = pipeline.ingest(str(target))
    assert r3.unchanged == 1


def test_document_hash_en_store(pipeline):
    assert pipeline.store.document_hash("no-existe") is None
    pipeline.ingest_text("Hash en metadata.", title="Hashed")
    docs = pipeline.store.list_documents()
    doc_id = docs[0]["document_id"]
    value = pipeline.store.document_hash(doc_id)
    assert value == content_fingerprint("Hash en metadata.")


def test_content_fingerprint_estable_ante_espacios():
    a = content_fingerprint("Hola\n\nmundo   blanco.")
    b = content_fingerprint("Hola\nmundo blanco.\n")
    assert a == b
    assert a != content_fingerprint("Hola mundo distinto.")


def test_documentos_de_texto_distinto_no_colisionan(pipeline):
    r1 = pipeline.ingest_text("Primer documento A.", title="Doc")
    r2 = pipeline.ingest_text("Segundo documento B.", title="Doc2")
    assert r1.documents == 1 and r2.documents == 1
    assert len(pipeline.documents()) == 2
