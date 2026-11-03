"""Tests de retriever y generadores."""

import pytest

from universal_rag.core.generator import (
    EchoGenerator,
    GeneratorError,
    OpenAICompatGenerator,
    create_generator,
)
from universal_rag.core.models import Chunk, RetrievedChunk
from universal_rag.core.utils import make_id


def retrieved(text: str, score: float = 0.9, title: str = "T") -> RetrievedChunk:
    chunk = Chunk(
        id=make_id("c", text[:10]),
        document_id="d1",
        text=text,
        index=0,
        total=1,
        metadata={"source": "text", "uri": "text://t", "title": title, "document_id": "d1"},
    )
    return RetrievedChunk(chunk=chunk, score=score)


# ------------------------------------------------------------------ retriever
class TestRetriever:
    def test_recupera_ranking(self, pipeline_con_datos):
        results = pipeline_con_datos.retriever.retrieve("¿cómo limpio los paneles solares?", k=3)
        assert 1 <= len(results) <= 3
        assert all(r.score >= 0 for r in results)
        texts = " ".join(r.chunk.text for r in results)
        assert "limpieza" in texts or "limpiar" in texts

    def test_pregunta_vacia(self, pipeline_con_datos):
        assert pipeline_con_datos.retriever.retrieve("   ") == []

    def test_top_k_default(self, pipeline_con_datos):
        pipeline_con_datos.retriever.top_k = 1
        assert len(pipeline_con_datos.retriever.retrieve("energía")) == 1


# ------------------------------------------------------------------ generators
class TestEchoGenerator:
    def test_genera_extractivo(self):
        gen = EchoGenerator()
        contexts = [retrieved("Fragmento uno sobre solar."), retrieved("Fragmento dos sobre eólica.")]
        text, model = gen.generate("¿qué es la energía solar?", contexts)
        assert model == "echo"
        assert "Fragmento uno" in text
        assert "[1]" in text and "[2]" in text


class TestOpenAICompatGenerator:
    def test_construccion_prompt(self):
        gen = OpenAICompatGenerator(base_url="http://x/v1", model="m", max_context_chars=2000)
        prompt = gen._user_prompt("¿qué es solar?", [retrieved("Contenido del fragmento.")])
        assert "Contenido del fragmento." in prompt
        assert "[1]" in prompt
        assert "¿qué es solar?" in prompt

    def test_truncado_contexto(self):
        gen = OpenAICompatGenerator(base_url="http://x/v1", model="m", max_context_chars=50)
        prompt = gen._user_prompt("q", [retrieved("x" * 500)])
        assert len(prompt) < 300

    def test_llamada_error_red(self):
        gen = OpenAICompatGenerator(base_url="http://127.0.0.1:1", model="m", timeout=0.5)
        with pytest.raises(GeneratorError):
            gen.generate("q", [retrieved("ctx")])


class TestFactory:
    def test_echo_forzado(self):
        from universal_rag.core.config import Settings

        assert isinstance(create_generator(Settings(llm_provider="echo")), EchoGenerator)

    def test_auto_sin_base_url_es_echo(self):
        from universal_rag.core.config import Settings

        assert isinstance(create_generator(Settings(llm_provider="auto")), EchoGenerator)

    def test_openai_compat_requiere_base_url(self):
        from universal_rag.core.config import Settings

        with pytest.raises(GeneratorError):
            create_generator(Settings(llm_provider="openai-compat"))
