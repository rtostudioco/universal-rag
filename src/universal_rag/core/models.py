"""Modelos de datos normalizados de Universal RAG.

Todos los conectores, sin importar su origen (archivo, URL, audio, video…),
producen :class:`Document`; el chunker produce :class:`Chunk`; la búsqueda
produce :class:`RetrievedChunk`; la generación produce :class:`Answer` con
:class:`Citation`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from universal_rag.core.utils import utcnow


class Document(BaseModel):
    """Unidad universal de conocimiento ingestado por un conector."""

    id: str
    content: str
    source: str  # nombre del conector: "file", "url", "text", …
    uri: str  # ruta, URL o URI sintética (text://…)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None  # p. ej. fecha de grabación de un MP3/MP4 (Fase 2)
    ingested_at: datetime = Field(default_factory=utcnow)


class Chunk(BaseModel):
    """Fragmento de un documento listo para vectorizar."""

    id: str
    document_id: str
    text: str
    index: int  # posición del fragmento dentro del documento (0-based)
    total: int  # total de fragmentos del documento
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Fragmento recuperado con su score de similitud (mayor = más relevante)."""

    chunk: Chunk
    score: float = 0.0


class Citation(BaseModel):
    """Referencia verificable a la fuente usada en la respuesta."""

    chunk_id: str
    document_id: str
    source: str
    uri: str
    title: str | None = None
    quote: str
    score: float = 0.0
    # Citas por minuto (audio/video — Fase 2)
    start_time: float | None = None
    end_time: float | None = None
    timestamp: str | None = None  # "[03:12–04:05]" ya formateado


class Answer(BaseModel):
    """Respuesta final con citas a las fuentes recuperadas."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str  # nombre del generador usado
    latency_ms: float = 0.0
    context_count: int = 0
    retrieval: str | None = None  # modo de búsqueda: "vector", "hybrid", "hybrid+cross-encoder"…
    cached: bool = False  # servida desde la caché semántica (Fase 6)
    conversation_id: str | None = None  # conversación multi-turno (Fase 7)


class IngestResult(BaseModel):
    """Resumen de una operación de ingesta."""

    connector: str
    documents: int = 0
    chunks: int = 0
    skipped: int = 0  # documentos vacíos o duplicados omitidos
    unchanged: int = 0  # documentos sin cambios saltados (ingesta incremental — Fase 4)
    document_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    """Un paso trazable del agente RAG (Fase 8): plan, búsqueda, refinamiento o síntesis."""

    step: int  # 0 = planificación; 1..n = búsquedas; -1 = refinamiento
    action: Literal["plan", "search", "refine", "synthesize"]
    query: str | None = None  # sub-consulta ejecutada (search/refine)
    found: int = 0  # fragmentos recuperados en el paso
    note: str | None = None  # detalle opcional (p. ej. "reformulada", "sin resultados")


class AgentAnswer(BaseModel):
    """Respuesta del agente RAG (Fase 8): síntesis multi-hop con trazabilidad de pasos."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str  # generador usado en la síntesis
    planner: str = "heuristic"  # "heuristic" | nombre del modelo que planificó
    steps: list[AgentStep] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    context_count: int = 0  # contextos fusionados únicos usados en la síntesis
    retrieval: str | None = None
    latency_ms: float = 0.0
