"""Tests del módulo de transcripciones y del chunking temporal (Fase 2)."""

import pytest

from universal_rag.core.chunker import Chunker
from universal_rag.core.models import Document
from universal_rag.core.transcript import Segment, Transcriber, format_range, format_timestamp
from universal_rag.core.utils import make_id


# ---------------------------------------------------------------- timestamps
class TestTimestamps:
    def test_format_timestamp(self):
        assert format_timestamp(0) == "00:00"
        assert format_timestamp(5.4) == "00:05"
        assert format_timestamp(59) == "00:59"
        assert format_timestamp(60) == "01:00"
        assert format_timestamp(754) == "12:34"
        assert format_timestamp(3671) == "1:01:11"
        assert format_timestamp(None) == ""

    def test_format_range(self):
        assert format_range(0, 754) == "[00:00–12:34]"
        assert format_range(None, None) is None
        assert format_range(120, None) == "[02:00–00:00]" or format_range(120, None) == "[02:00–]"


# ---------------------------------------------------------------- transcriber
class TestTranscriber:
    def test_lazy_no_modelo_al_instanciar(self):
        t = Transcriber(model_size="base")
        assert t._model is None  # no descarga nada al crear

    def test_config_desde_settings(self, monkeypatch):
        monkeypatch.setenv("RAG_ASR_MODEL", "tiny")
        monkeypatch.setenv("RAG_ASR_LANGUAGE", "es")
        t = Transcriber()
        assert t.model_size == "tiny"
        assert t.language == "es"

    def test_ensure_model_sin_faster_whisper(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("no instalado")
            return real_import(name, *args, **kwargs)

        t = Transcriber(model_size="base")
        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", fake_import)
            with pytest.raises(RuntimeError, match="universal-rag\\[media\\]"):
                t.ensure_model()

    def test_segment_roundtrip(self):
        seg = Segment(start=1.5, end=3.2, text="hola")
        data = seg.to_dict()
        assert Segment.from_dict(data) == seg


# --------------------------------------------------------------- chunker F2
SEGS = [
    {"start": 0.0, "end": 4.0, "text": "Bienvenidos al podcast de energía."},
    {"start": 4.0, "end": 9.5, "text": "Hoy hablamos de paneles solares."},
    {"start": 9.5, "end": 15.0, "text": "El mantenimiento es cada tres meses."},
    {"start": 15.0, "end": 21.0, "text": "Y el inversor se revisa una vez al año."},
]


def media_doc(metadata_extra: dict | None = None) -> Document:
    return Document(
        id=make_id("doc", "audio://x.mp3"),
        content=" ".join(s["text"] for s in SEGS),
        source="audio",
        uri="audio://x.mp3",
        title="Podcast",
        metadata={"segments": SEGS, "duration": 21.0, **(metadata_extra or {})},
    )


class TestTranscriptChunker:
    def test_un_chunk_pequeno(self):
        chunks = Chunker(size=500, overlap=50).chunk_document(media_doc())
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata["start_time"] == 0.0
        assert chunk.metadata["end_time"] == 21.0
        assert "Bienvenidos" in chunk.text and "inversor" in chunk.text
        assert "segments" not in chunk.metadata  # no arrastra los segmentos crudos
        assert chunk.metadata["duration"] == 21.0

    def test_varios_chunks_con_tiempos(self):
        chunks = Chunker(size=60, overlap=0).chunk_document(media_doc())
        assert len(chunks) > 1
        assert all(c.metadata["end_time"] > c.metadata["start_time"] for c in chunks)
        # contigüidad temporal
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            assert nxt.metadata["start_time"] >= prev.metadata["start_time"]
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert all(c.total == len(chunks) for c in chunks)
        assert len({c.id for c in chunks}) == len(chunks)

    def test_segmentos_invalidos_se_ignoran(self):
        dirty = [{"start": "x", "end": "y", "text": "roto"}, {"start": 99, "end": 100, "text": ""}, *SEGS]
        chunks = Chunker(size=500, overlap=0).chunk_document(media_doc({"segments": dirty}))
        assert len(chunks) == 1
        assert "Bienvenidos" in chunks[0].text

    def test_doc_normal_no_se_afecta(self):
        doc = Document(
            id=make_id("doc", "text://a"),
            content="Texto plano normal sin segmentos.",
            source="text",
            uri="text://a",
        )
        chunks = Chunker().chunk_document(doc)
        assert len(chunks) == 1
        assert "start_time" not in chunks[0].metadata
