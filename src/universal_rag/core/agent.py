"""RAGAgent — agente multi-hop sobre el pipeline (Fase 8).

En lugar de una sola búsqueda, el agente:

1. **Planifica**: descompone la pregunta en sub-consultas independientes
   (LLM si hay generador con ``chat()``, heurística determinista en caso
   contrario — modo echo/offline/tests).
2. **Busca** cada sub-consulta y fusiona los fragmentos recuperados
   (dedupe por ``chunk_id`` conservando el mejor score).
3. **Refina** las sub-consultas sin resultados (reformulación con LLM o
   extracción de palabras de contenido) antes de rendirse.
4. **Sintetiza**: una sola generación final sobre los contextos fusionados,
   con citas ``[n]`` verificables sobre el ranking combinado.

Trazabilidad total: cada paso queda registrado en :class:`AgentStep`, y la
respuesta final :class:`AgentAnswer` expone plan, sub-consultas y pasos.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

from universal_rag.core.generator import GeneratorError
from universal_rag.core.models import AgentAnswer, AgentStep, RetrievedChunk
from universal_rag.core.pipeline import RAGPipeline
from universal_rag.core.utils import get_logger

logger = get_logger("agent")

AGENT_NO_CONTEXT_ANSWER = (
    "No encontré información relevante en la base de conocimiento para esa pregunta "
    "(ni con sub-consultas reformuladas). Prueba con otros términos o ingesta más contenido."
)

_PLAN_SYSTEM_PROMPT = (
    "Eres un planificador de búsquedas. Descompón la pregunta del usuario en "
    "sub-consultas de búsqueda independientes y autocontenidas (sin pronombres "
    "ambiguos) que, combinadas, cubran todo lo necesario para responderla. "
    "Responde ÚNICAMENTE un JSON array de strings (entre 1 y {max_steps}), "
    "sin explicaciones ni texto adicional."
)

_REFINE_SYSTEM_PROMPT = (
    "Reformula esta consulta de búsqueda: elimina palabras interrogativas y de "
    "relleno, conserva solo los términos de contenido clave. Responde ÚNICAMENTE "
    "con la consulta reformulada, sin comillas ni explicaciones."
)

# Stopwords mínimas es/en para la reformulación heurística (sin dependencias).
_STOPWORDS = frozenset(
    """
    a al algo alguna algunas alguno algunos ante antes como con contra cual cuales
    cuando de del desde donde dos e el ella ellas ellos en entre era eran eres es
    esa esas ese eso esos esta estaba estan estar estas este esto estos fue fueron
    ha habia han hasta hay la las le les lo los mas me mi mis mucho muy nada ni no
    nos o os otra otras otro otros para pero poco por porque que quien quienes se
    sea sean segun ser si sin sobre solo son su sus tambien tanto te tiene tienen
    toda todas todo todos tras tu tus un una unas uno unos ustedes y ya yo
    qué cómo dónde cuándo quién quiénes cuál cuáles cuánto cuánta cuántos cuántas
    además sé sabe sabes
    a about after all also an and any are as at be because been before being but
    by can could did do does for from had has have how i if in into is it its
    may might of on or our shall should than that the their them then there
    these they this those to was we were what when where which who will with would
    """.split()
)

_SPLIT_RE = re.compile(r"\n+|;|\?|¿|(?<=\.)\s+")


def heuristic_plan(question: str, max_steps: int = 3) -> list[str]:
    """Descomposición heurística determinista (sin LLM).

    Parte la pregunta en separadores fuertes (saltos de línea, ``;``,
    interrogaciones, fin de frase) y, si solo hay un fragmento, intenta la
    conjunción `` y `` cuando ambos lados tienen peso suficiente. Los
    fragmentos demasiado cortos (menos de 3 palabras) se fusionan con el
    anterior. Si nada parte, devuelve la pregunta original intacta.
    """
    text = (question or "").strip()
    if not text:
        return [""]
    parts = [p.strip(" .¿?¡!") for p in _SPLIT_RE.split(text) if p and p.strip(" .¿?¡!")]
    if len(parts) <= 1:
        halves = [h.strip(" .¿?¡!") for h in text.split(" y ")]
        if len(halves) == 2 and all(len(h.split()) >= 3 for h in halves):
            return halves[: max(1, max_steps)]
        return [text]
    merged: list[str] = []
    for part in parts:
        if len(part.split()) < 3 and merged:
            merged[-1] = f"{merged[-1]} {part}".strip()
        elif part:
            merged.append(part)
    if not merged:
        return [text]
    return merged[: max(1, max_steps)]


def content_words_query(question: str, *, min_words: int = 2) -> str | None:
    """Reformulación heurística: solo palabras de contenido (sin stopwords)."""
    words = [w.lower() for w in re.findall(r"\w+", question or "", re.UNICODE)]
    content = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if len(content) < min_words:
        return None
    return " ".join(content)


def _parse_json_list(raw: str) -> list[str] | None:
    """Extrae el primer JSON array de strings de una respuesta LLM (tolerante)."""
    text = (raw or "").strip()
    if not text:
        return None
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    queries = [item.strip() for item in data if isinstance(item, str) and item.strip()]
    return queries or None


class RAGAgent:
    """Agente multi-hop stateless sobre un :class:`RAGPipeline` (Fase 8).

    ``max_steps`` — máximo de sub-consultas (default: ``RAG_AGENT_MAX_STEPS``).
    ``planning`` — ``"auto"`` (LLM si el generador soporta ``chat()``),
    ``"heuristic"`` (determinista) o ``"llm"`` (fuerza LLM con fallback).
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        max_steps: int | None = None,
        planning: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.settings = pipeline.settings
        self.max_steps = max_steps or self.settings.agent_max_steps
        self.planning = planning or self.settings.agent_planning

    # ------------------------------------------------------------------ plan
    def plan(self, question: str) -> tuple[list[str], str]:
        """Devuelve ``(sub_consultas, nombre_del_planificador)``.

        El planificador puede ser ``"heuristic"`` o el label del modelo LLM
        (p. ej. ``gpt-4o-mini``) cuando la planificación se hizo con LLM.
        """
        if self.planning == "heuristic":
            return heuristic_plan(question, self.max_steps), "heuristic"
        sub_queries = self._llm_plan(question)
        if sub_queries is not None:
            model = getattr(self.pipeline.generator, "model_label", "llm")
            return sub_queries[: self.max_steps], str(model)
        return heuristic_plan(question, self.max_steps), "heuristic"

    def _llm_plan(self, question: str) -> list[str] | None:
        """Planificación con LLM (None si no hay ``chat()`` o el plan falla)."""
        generator = self.pipeline.generator
        if not hasattr(generator, "chat"):
            return None
        try:
            raw = generator.chat(
                [
                    {
                        "role": "system",
                        "content": _PLAN_SYSTEM_PROMPT.format(max_steps=self.max_steps),
                    },
                    {"role": "user", "content": question.strip()},
                ],
                temperature=0.0,
            )
        except GeneratorError as exc:
            logger.warning("Planificación LLM falló (%s); fallback heurístico", exc)
            return None
        sub_queries = _parse_json_list(raw)
        if not sub_queries:
            return None
        # dedupe estable + límite de pasos
        seen: set[str] = set()
        unique: list[str] = []
        for query in sub_queries:
            key = query.lower()
            if key not in seen:
                seen.add(key)
                unique.append(query)
        return unique[: self.max_steps]

    def _llm_refine(self, query: str) -> str | None:
        """Reformulación con LLM (None si no procede o falla)."""
        generator = self.pipeline.generator
        if not hasattr(generator, "chat"):
            return None
        try:
            raw = generator.chat(
                [
                    {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
                    {"role": "user", "content": query.strip()},
                ],
                temperature=0.0,
            )
        except GeneratorError as exc:
            logger.warning("Reformulación LLM falló (%s); fallback heurístico", exc)
            return None
        return raw.strip().strip('"') or None

    def _refine(self, query: str) -> str | None:
        """Mejor reformulación disponible para una sub-consulta sin resultados."""
        if self.planning != "heuristic":
            refined = self._llm_refine(query)
            if refined and refined.lower() != query.lower():
                return refined
        refined = content_words_query(query)
        if refined and refined != query.lower():
            return refined
        return None

    # ------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> AgentAnswer:
        """Respuesta multi-hop con trazabilidad completa de pasos."""
        started = time.perf_counter()
        steps: list[AgentStep] = []
        sub_queries, planner = self.plan(question)
        steps.append(
            AgentStep(
                step=0,
                action="plan",
                note=f"{planner}: {len(sub_queries)} sub-consulta(s)",
            )
        )
        merged: dict[str, RetrievedChunk] = {}
        for i, sub_query in enumerate(sub_queries, start=1):
            contexts = self.pipeline.retriever.retrieve(sub_query, k=k, where=where)
            note = None
            if not contexts:
                refined = self._refine(sub_query)
                if refined:
                    contexts = self.pipeline.retriever.retrieve(refined, k=k, where=where)
                    note = f"reformulada → «{refined}»" if contexts else "reformulada, sin resultados"
                else:
                    note = "sin resultados"
            for retrieved in contexts:
                current = merged.get(retrieved.chunk.id)
                if current is None or retrieved.score > current.score:
                    merged[retrieved.chunk.id] = retrieved
            steps.append(
                AgentStep(step=i, action="search", query=sub_query, found=len(contexts), note=note)
            )
        final_contexts = self._rank(merged, k)
        if not final_contexts:
            # Última oportunidad: reformular la pregunta completa.
            refined = self._refine(question)
            if refined:
                contexts = self.pipeline.retriever.retrieve(refined, k=k, where=where)
                steps.append(
                    AgentStep(step=-1, action="refine", query=refined, found=len(contexts))
                )
                final_contexts = self._rank({rc.chunk.id: rc for rc in contexts}, k)
        if not final_contexts:
            return AgentAnswer(
                question=question,
                answer=AGENT_NO_CONTEXT_ANSWER,
                citations=[],
                model=self.pipeline.generator.name,
                planner=planner,
                steps=steps,
                sub_queries=sub_queries,
                context_count=0,
                retrieval=self.pipeline.retrieval_label(),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        text, model = self.pipeline.generator.generate(question, final_contexts)
        citations = self.pipeline.build_citations(text, final_contexts)
        steps.append(
            AgentStep(
                step=len(steps),
                action="synthesize",
                found=len(final_contexts),
                note=f"{len(final_contexts)} contexto(s) fusionado(s)",
            )
        )
        return AgentAnswer(
            question=question,
            answer=text,
            citations=citations,
            model=model,
            planner=planner,
            steps=steps,
            sub_queries=sub_queries,
            context_count=len(final_contexts),
            retrieval=self.pipeline.retrieval_label(),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # ---------------------------------------------------------------- stream
    def stream_ask(
        self,
        question: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Versión en streaming de :meth:`ask` (Fase 8).

        Eventos dict en orden::

            {"type": "plan",       "sub_queries": [...], "planner": …}   # 1 vez
            {"type": "step",       "step": {…AgentStep…}}                # N veces
            {"type": "synthesize", "context_count": …}                   # 1 vez
            {"type": "token",      "value": "…"}                         # M veces
            {"type": "done",       "answer": {…AgentAnswer…}}            # 1 vez
        """
        started = time.perf_counter()
        sub_queries, planner = self.plan(question)
        yield {"type": "plan", "sub_queries": sub_queries, "planner": planner}
        steps: list[AgentStep] = [
            AgentStep(step=0, action="plan", note=f"{planner}: {len(sub_queries)} sub-consulta(s)")
        ]
        merged: dict[str, RetrievedChunk] = {}
        for i, sub_query in enumerate(sub_queries, start=1):
            contexts = self.pipeline.retriever.retrieve(sub_query, k=k, where=where)
            note = None
            if not contexts:
                refined = self._refine(sub_query)
                if refined:
                    contexts = self.pipeline.retriever.retrieve(refined, k=k, where=where)
                    note = f"reformulada → «{refined}»" if contexts else "reformulada, sin resultados"
                else:
                    note = "sin resultados"
            for retrieved in contexts:
                current = merged.get(retrieved.chunk.id)
                if current is None or retrieved.score > current.score:
                    merged[retrieved.chunk.id] = retrieved
            step = AgentStep(
                step=i, action="search", query=sub_query, found=len(contexts), note=note
            )
            steps.append(step)
            yield {"type": "step", "step": step.model_dump()}
        final_contexts = self._rank(merged, k)
        if not final_contexts:
            refined = self._refine(question)
            if refined:
                contexts = self.pipeline.retriever.retrieve(refined, k=k, where=where)
                step = AgentStep(step=-1, action="refine", query=refined, found=len(contexts))
                steps.append(step)
                yield {"type": "step", "step": step.model_dump()}
                final_contexts = self._rank({rc.chunk.id: rc for rc in contexts}, k)
        yield {"type": "synthesize", "context_count": len(final_contexts)}
        if not final_contexts:
            answer = AgentAnswer(
                question=question,
                answer=AGENT_NO_CONTEXT_ANSWER,
                citations=[],
                model=self.pipeline.generator.name,
                planner=planner,
                steps=steps,
                sub_queries=sub_queries,
                context_count=0,
                retrieval=self.pipeline.retrieval_label(),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            yield {"type": "done", "answer": answer.model_dump()}
            return
        parts: list[str] = []
        for piece in self.pipeline.generator.stream_generate(question, final_contexts):
            parts.append(piece)
            yield {"type": "token", "value": piece}
        text = "".join(parts)
        citations = self.pipeline.build_citations(text, final_contexts)
        steps.append(
            AgentStep(
                step=len(steps),
                action="synthesize",
                found=len(final_contexts),
                note=f"{len(final_contexts)} contexto(s) fusionado(s)",
            )
        )
        answer = AgentAnswer(
            question=question,
            answer=text,
            citations=citations,
            model=self.pipeline.generator.model_label,
            planner=planner,
            steps=steps,
            sub_queries=sub_queries,
            context_count=len(final_contexts),
            retrieval=self.pipeline.retrieval_label(),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        yield {"type": "done", "answer": answer.model_dump()}

    # ------------------------------------------------------------- internals
    def _rank(self, merged: dict[str, RetrievedChunk], k: int | None) -> list[RetrievedChunk]:
        """Ordena por score descendente y limita al top-k efectivo."""
        limit = k or self.settings.top_k
        ranked = sorted(merged.values(), key=lambda rc: rc.score, reverse=True)
        return ranked[:limit]
