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
