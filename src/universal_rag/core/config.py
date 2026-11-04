"""Configuración central de Universal RAG (variables de entorno con prefijo ``RAG_``).

Ejemplos::

    RAG_EMBEDDING_PROVIDER=auto          # auto | sentence-transformers | hash
    RAG_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
    RAG_STORE_PROVIDER=chroma            # chroma | memory
    RAG_CHROMA_PATH=./data/chroma
    RAG_LLM_PROVIDER=auto                # auto | openai-compat | echo
    RAG_LLM_BASE_URL=https://api.openai.com/v1
    RAG_LLM_API_KEY=sk-...
    RAG_LLM_MODEL=gpt-4o-mini
    RAG_CHUNK_SIZE=900
    RAG_CHUNK_OVERLAP=150
    RAG_TOP_K=5
    RAG_CONNECTOR_DIRS=./mis_conectores

Fase 4 (producción)::

    RAG_DEFAULT_COLLECTION=default       # colección activa por defecto
    RAG_API_KEY=secreto                  # si se define, exige X-API-Key / Bearer en la API
    RAG_JOBS_WORKERS=2                   # hilos del pool de ingesta asíncrona
    RAG_JOBS_MAX_HISTORY=200             # historial máximo de jobs en memoria

Fase 6 (streaming, caché y workbench)::

    RAG_CACHE_ENABLED=1                  # caché semántica de consultas
    RAG_CACHE_THRESHOLD=0.93             # similitud coseno mínima para acertar
    RAG_CACHE_MAX_ENTRIES=512            # LRU máximo en memoria
    RAG_CACHE_TTL=                       # segundos de vida (vacío = sin TTL)

Fase 7 (memoria conversacional y sincronización)::

    RAG_MEMORY_MAX_TURNS=8               # turnos recordados por conversación
    RAG_MEMORY_MAX_CONVERSATIONS=200     # LRU de conversaciones vivas
    RAG_SYNC_ENABLED=0                   # arranca el planificador de sincronización
    RAG_SYNC_TICK=5                      # granularidad del bucle (segundos)
    RAG_SYNC_STATE_PATH=                 # ruta JSON opcional para persistir jobs

Fase 8 (agente multi-hop)::

    RAG_AGENT_MAX_STEPS=3                # máximo de sub-consultas por pregunta
    RAG_AGENT_PLANNING=auto              # auto | heuristic | llm
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Ajustes del pipeline. Todo es overridable por variables ``RAG_*``."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Embeddings -------------------------------------------------------
    embedding_provider: Literal["auto", "sentence-transformers", "hash"] = "auto"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str | None = None  # "cpu", "cuda", None = auto
    embedding_dimensions: int = 256  # solo para el embedder hash

    # --- Vector store -----------------------------------------------------
    store_provider: Literal["chroma", "memory"] = "chroma"
    chroma_path: Path = Path("./data/chroma")
    collection_name: str = "universal_rag"

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(default=900, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    # --- LLM / generación -------------------------------------------------
    llm_provider: Literal["auto", "openai-compat", "echo"] = "auto"
    llm_base_url: str | None = None  # ej. https://api.openai.com/v1 (o cualquier endpoint compatible)
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout: float = 90.0
    max_context_chars: int = 6000

    # --- Búsqueda ---------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)

    # --- Búsqueda híbrida (Fase 3) -----------------------------------------
    # hybrid = fusión del ranking vectorial con BM25 (léxico) sobre el corpus.
    retrieval: Literal["vector", "hybrid"] = "vector"
    hybrid_fusion: Literal["rrf", "weighted"] = "rrf"  # rrf | weighted (min-max)
    hybrid_rrf_k: int = Field(default=60, ge=1)  # constante k de Reciprocal Rank Fusion
    hybrid_alpha: float = Field(default=0.55, ge=0.0, le=1.0)  # peso vectorial en "weighted"
    candidate_k: int = Field(default=25, ge=1, le=200)  # candidatos antes de rerank/top-k

    # --- Reranking (Fase 3/5) -----------------------------------------------
    reranker: Literal["none", "cross-encoder", "llm"] = "none"
    rerank_model: str = "BAAI/bge-reranker-base"  # multilingüe (buen español); requiere [local]
    rerank_llm_model: str | None = None  # modelo para reranker=llm (default: RAG_LLM_MODEL)

    # --- ASR (audio/video — Fase 2) ---------------------------------------
    asr_model: str = "base"  # tiny | base | small | medium | large-v3 (faster-whisper)
    asr_device: str | None = None  # "cpu", "cuda", None = auto
    asr_compute_type: str = "int8"  # int8 es rápido y ligero en CPU
    asr_language: str | None = None  # None = detección automática; "es", "en", …

    # --- Fase 4: producción y escalado -------------------------------------
    default_collection: str = "default"  # colección activa si no se indica otra
    api_key: str | None = None  # si se define, la API exige X-API-Key o Bearer
    jobs_workers: int = Field(default=2, ge=1, le=16)  # pool de ingesta asíncrona
    jobs_max_history: int = Field(default=200, ge=10)  # jobs recordados en memoria

    # --- Fase 6: caché semántica -------------------------------------------
    cache_enabled: bool = True  # caché semántica de respuestas (por colección)
    cache_threshold: float = Field(default=0.93, ge=0.50, le=1.0)  # coseno mínimo
    cache_max_entries: int = Field(default=512, ge=10)  # LRU máximo
    cache_ttl: float | None = Field(default=None, ge=0)  # segundos; None = sin expiración

    # --- Fase 7: memoria conversacional -------------------------------------
    memory_max_turns: int = Field(default=8, ge=1, le=100)  # turnos por conversación
    memory_max_conversations: int = Field(default=200, ge=1)  # LRU de conversaciones

    # --- Fase 7: sincronización programada ----------------------------------
    sync_enabled: bool = False  # arranca el planificador con la API
    sync_tick: float = Field(default=5.0, ge=0.5)  # granularidad del bucle (s)
    sync_state_path: Path | None = None  # JSON opcional para persistir los jobs

    # --- Fase 8: agente multi-hop -------------------------------------------
    agent_max_steps: int = Field(default=3, ge=1, le=10)  # máximo de sub-consultas
    agent_planning: Literal["auto", "heuristic", "llm"] = "auto"
    # auto = plan LLM si hay generador con chat(); heuristic en caso contrario

    # --- Conectores externos ---------------------------------------------
    # Directorios con conectores externos: <dir>/<nombre>/connector.py
    connector_dirs: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("connector_dirs", mode="before")
    @classmethod
    def _split_connector_dirs(cls, value: object) -> object:
        """Acepta ``RAG_CONNECTOR_DIRS=a:b`` o ``a,b`` además de listas JSON."""
        if isinstance(value, str):
            parts = [p for chunk in value.replace(":", ",").split(",") for p in [chunk.strip()]]
            return [p for p in parts if p]
        return value

    # --- Atajos -----------------------------------------------------------
    def resolved_store_kwargs(self) -> dict:
        return {
            "path": str(self.chroma_path),
            "collection_name": self.collection_name,
        }
