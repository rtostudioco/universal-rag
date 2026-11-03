"""Generación plugable de respuestas.

* :class:`EchoGenerator` — fallback extractivo sin LLM: cita el contexto
  recuperado tal cual (offline, determinista, ideal para tests).
* :class:`OpenAICompatGenerator` — funciona con cualquier endpoint compatible
  OpenAI: OpenAI, z-ai, Ollama (`/v1`), LM Studio, vLLM, OpenRouter…

El contrato es simple: dado (pregunta, contextos) → (texto_respuesta, modelo).
La construcción de :class:`Answer` con :class:`Citation` la hace el pipeline,
extrayendo los marcadores ``[1]``, ``[2]``… del texto generado.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

import httpx

from universal_rag.core.config import Settings
from universal_rag.core.models import RetrievedChunk
from universal_rag.core.utils import get_logger, truncate

logger = get_logger("generator")

SYSTEM_PROMPT_ES = (
    "Eres un asistente que responde preguntas usando EXCLUSIVAMENTE el contexto "
    "proporcionado (fragmentos numerados). Reglas:\n"
    "1. Responde en el idioma de la pregunta.\n"
    "2. Cita las fuentes inline con su número entre corchetes, ej. [1] o [2][3].\n"
    "3. Si el contexto no contiene la respuesta, dilo claramente.\n"
    "4. No inventes información ni cites fragmentos que no existan."
)


class GeneratorError(RuntimeError):
    """Fallo del generador LLM."""


def tokenize_stream(text: str, words_per_piece: int = 3) -> Iterator[str]:
    """Parte un texto en trozos pequeños (~tokens de LLM) para streaming.

    La concatenación de los trozos reproduce el texto original **exactamente**
    (los separadores entre trozos viajan dentro del trozo siguiente).
    """
    words = (text or "").split(" ")
    if not words or words == [""]:
        return
    groups = [" ".join(words[start : start + words_per_piece]) for start in range(0, len(words), words_per_piece)]
    for i, piece in enumerate(groups):
        yield piece if i == 0 else " " + piece


class BaseGenerator(ABC):
    name: ClassVar[str] = "base"

    @abstractmethod
    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        """Devuelve (texto_respuesta, nombre_modelo).

        ``history`` — mensajes previos de la conversación (Fase 7):
        ``[{"role": "user"|"assistant", "content": …}, …]``. Opcional.
        """

    @property
    def model_label(self) -> str:
        """Nombre del modelo para mostrar (default: nombre del generador)."""
        return self.name

    def stream_generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Streaming por defecto: genera todo y lo emite en trozos pequeños.

        Los generadores con streaming real (SSE del LLM) lo sobreescriben.
        """
        text, _ = self.generate(question, contexts, history=history)
        yield from tokenize_stream(text)


# ------------------------------------------------------------------------- echo
class EchoGenerator(BaseGenerator):
    """Extractivo: sintetiza el contexto sin LLM (offline / tests).

    Ignora el historial conversacional (no hay LLM al que inyectárselo).
    """

    name = "echo"

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        lines = [f"Resumen extractivo para: «{question.strip()}»", ""]
        for i, retrieved in enumerate(contexts, start=1):
            title = retrieved.chunk.metadata.get("title") or "(sin título)"
            lines.append(f"[{i}] {title}")
            lines.append(truncate(retrieved.chunk.text, 400))
            lines.append("")
        lines.append("(Modo extractivo: no hay LLM configurado; se muestran los fragmentos recuperados.)")
        return "\n".join(lines).strip(), self.name

    def stream_generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        text, _ = self.generate(question, contexts, history=history)
        yield from tokenize_stream(text)


# ------------------------------------------------------------------- openai-compat
class OpenAICompatGenerator(BaseGenerator):
    """Generador LLM vía cualquier API compatible OpenAI (/chat/completions)."""

    name = "openai-compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_context_chars: int = 6000,
        timeout: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.timeout = timeout
        self._client = client  # inyectable para tests (httpx.MockTransport)

    @property
    def model_label(self) -> str:
        return self.model

    # ------------------------------------------------------------------ API
    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        try:
            response = self._post(self._payload(question, contexts, history), self._headers())
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            raise GeneratorError(f"Error llamando al LLM ({self.base_url}): {exc}") from exc
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise GeneratorError(f"Respuesta del LLM con formato inesperado: {data!r}") from exc
        return text.strip(), self.model

    def stream_generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """Streaming real vía SSE del endpoint (/chat/completions stream=true)."""
        payload = self._payload(question, contexts, history)
        payload["stream"] = True
        pieces: list[str] = []
        try:
            with self._stream(payload, self._headers()) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    piece = self._parse_sse_line(line)
                    if piece is None:
                        continue
                    pieces.append(piece)
                    yield piece
        except httpx.HTTPError as exc:
            if pieces:
                # El stream se cortó a mitad: devolvemos lo recibido (parcial).
                logger.warning("Stream interrumpido tras %d pieza(s): %s", len(pieces), exc)
                return
            raise GeneratorError(f"Error llamando al LLM ({self.base_url}): {exc}") from exc
        if not pieces:
            # Endpoint que ignora stream=true o devolvió vacío: fallback clásico.
            text, _ = self.generate(question, contexts)
            yield text

    # ------------------------------------------------------------- internals
    def _payload(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> dict:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_ES}]
        # Memoria conversacional (Fase 7): turnos previos entre system y user.
        messages.extend(self._history_messages(history))
        messages.append({"role": "user", "content": self._user_prompt(question, contexts)})
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }

    @staticmethod
    def _history_messages(history: list[dict[str, str]] | None, limit: int = 16) -> list[dict[str, str]]:
        """Sanitiza el historial: roles válidos, strings, últimos ``limit`` mensajes."""
        if not history:
            return []
        clean: list[dict[str, str]] = []
        for message in history:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                clean.append({"role": role, "content": content})
        return clean[-limit:]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict, headers: dict[str, str]) -> httpx.Response:
        if self._client is not None:
            return self._client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        return httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

    def _stream(self, payload: dict, headers: dict[str, str]):
        if self._client is not None:
            return self._client.stream("POST", f"{self.base_url}/chat/completions", json=payload, headers=headers)
        return httpx.stream(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )

    @staticmethod
    def _parse_sse_line(line: str) -> str | None:
        """Extrae el contenido de una línea SSE ``data: {...}`` (None si no hay)."""
        line = (line or "").strip()
        if not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            return None
        choices = chunk.get("choices") or [{}]
        delta = (choices[0] or {}).get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) and content else None

    def _user_prompt(self, question: str, contexts: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        used = 0
        for i, retrieved in enumerate(contexts, start=1):
            meta = retrieved.chunk.metadata
            label = meta.get("title") or meta.get("uri") or "(sin título)"
            source = f"fuente={meta.get('source', '?')} uri={meta.get('uri', '?')}"
            # citas por minuto (audio/video)
            start_time, end_time = meta.get("start_time"), meta.get("end_time")
            if start_time is not None or end_time is not None:
                from universal_rag.core.transcript import format_range

                when = format_range(start_time, end_time)
                if when:
                    source += f" minuto {when}"
            body = truncate(retrieved.chunk.text, self.max_context_chars // max(1, len(contexts)))
            block = f"[{i}] {label} ({source})\n{body}"
            if used + len(block) > self.max_context_chars and blocks:
                break
            blocks.append(block)
            used += len(block)
        context_text = "\n\n".join(blocks) if blocks else "(sin contexto)"
        return f"Contexto:\n{context_text}\n\nPregunta: {question.strip()}\n\nRespuesta (cita con [n]):"


# --------------------------------------------------------------------- fábrica
def create_generator(settings: Settings) -> BaseGenerator:
    """Resuelve el generador: openai-compat si hay base_url; echo como fallback."""
    provider = settings.llm_provider
    if provider == "echo":
        return EchoGenerator()
    if provider in ("auto", "openai-compat"):
        if settings.llm_base_url:
            return OpenAICompatGenerator(
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                temperature=settings.llm_temperature,
                max_context_chars=settings.max_context_chars,
                timeout=settings.llm_timeout,
            )
        if provider == "openai-compat":
            raise GeneratorError("llm_provider=openai-compat requiere RAG_LLM_BASE_URL")
        logger.info("Sin LLM configurado (RAG_LLM_BASE_URL vacío); se usa el generador extractivo.")
        return EchoGenerator()
    raise ValueError(f"Proveedor de LLM desconocido: {provider}")  # pragma: no cover
