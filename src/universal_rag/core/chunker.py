"""Chunker: divide documentos en fragmentos con solapamiento configurable.

Estrategia (robusta y determinista):
1. Divide el texto en unidades que caben en ``size`` (párrafos; los párrafos
   grandes se subdividen en frases; las frases enormes se cortan por caracteres).
2. Empaqueta unidades de forma greedy hasta ``size``.
3. Al abrir un fragmento nuevo, antepone la cola del anterior (~``overlap``
   caracteres, cortando en límite de palabra) para preservar contexto.
"""

from __future__ import annotations

import re
from typing import ClassVar

from universal_rag.core.models import Chunk, Document
from universal_rag.core.utils import make_id, normalize_whitespace

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


class Chunker:
    """Divisor de texto configurable (tamaño + solapamiento)."""

    _SEP_PARA: ClassVar[str] = "\n\n"

    def __init__(self, size: int = 900, overlap: int = 150) -> None:
        if overlap >= size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        self.size = max(100, int(size))
        self.overlap = max(0, min(int(overlap), self.size // 2))

    # ------------------------------------------------------------------ API
    def split(self, text: str) -> list[str]:
        """Devuelve la lista de fragmentos de texto (sin metadatos)."""
        text = normalize_whitespace(text)
        if not text:
            return []
        if len(text) <= self.size:
            return [text]
        return self._pack(self._units(text))

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Divide un documento en :class:`Chunk` con IDs deterministas."""
        pieces = self.split(document.content)
        total = len(pieces)
        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=make_id("chunk", document.id, index, piece),
                    document_id=document.id,
                    text=piece,
                    index=index,
                    total=total,
                    metadata={
                        "source": document.source,
                        "uri": document.uri,
                        "title": document.title,
                        **document.metadata,
                    },
                )
            )
        return chunks

    # ------------------------------------------------------------- internals
    def _units(self, text: str) -> list[str]:
        """Convierte el texto en unidades atómicas (palabras) para empaquetar."""
        units: list[str] = []
        for paragraph in text.split(self._SEP_PARA):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self.size:
                units.append(paragraph)
                continue
            for sentence in _SENTENCE_RE.split(paragraph):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) <= self.size:
                    units.append(sentence)
                else:
                    # sin fronteras de frase: baja a palabras (nunca corta a mitad de palabra)
                    units.extend(self._split_words(sentence))
        return units

    def _split_words(self, text: str) -> list[str]:
        """Unidades por palabra; una sola palabra gigante se corta por caracteres."""
        units: list[str] = []
        for word in text.split():
            if len(word) <= self.size:
                units.append(word)
            else:  # palabra patológica mayor que size
                units.extend(self._hard_split(word))
        return units

    def _hard_split(self, text: str) -> list[str]:
        step = self.size
        return [part.strip() for part in (text[i : i + step] for i in range(0, len(text), step)) if part.strip()]

    def _pack(self, units: list[str]) -> list[str]:
        """Empaqueta unidades en fragmentos <= size, con solapamiento contextual."""
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_len = 0

        def flush() -> None:
            nonlocal buffer, buffer_len
            if buffer:
                chunks.append(" ".join(buffer).strip())
                buffer, buffer_len = [], 0

        for unit in units:
            if buffer and buffer_len + 1 + len(unit) > self.size:
                flush()
                if self.overlap and chunks:
                    budget = self.size - len(unit) - 1
                    tail = self._tail(chunks[-1], limit=budget)
                    if tail:
                        buffer = [tail, unit]
                        buffer_len = len(tail) + 1 + len(unit)
                        continue
                buffer = [unit]
                buffer_len = len(unit)
                continue
            buffer.append(unit)
            buffer_len += (1 if buffer_len else 0) + len(unit)
        flush()
        return [chunk for chunk in chunks if chunk]

    def _tail(self, chunk: str, limit: int | None = None) -> str:
        """Cola del fragmento anterior cortada en límite de palabra.

        ``limit`` ajusta el tamaño máximo del solapamiento (p. ej. cuando la
        siguiente unidad ya ocupa casi todo el fragmento).
        """
        limit = self.overlap if limit is None else min(self.overlap, max(0, limit))
        if limit <= 0 or len(chunk) <= limit:
            return ""
        tail = chunk[-limit:].strip()
        space = tail.find(" ")
        if 0 < space < len(tail) - 1:
            tail = tail[space + 1 :]
        return tail
