"""Utilidades compartidas del núcleo."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"universal_rag.{name}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_id(*parts: object) -> str:
    """ID estable y determinista a partir de partes arbitrarias."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def short_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


_WS_RE = re.compile(r"[ \t]+")
_BLANK_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Colapsa espacios repetidos y líneas en blanco excesivas."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLANK_RE.sub("\n\n", text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines).strip()


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def chunk_quote(text: str, limit: int = 240) -> str:
    return truncate(text, limit)


def flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convierte metadatos arbitrarios a valores primitivos (para Chroma y similares)."""
    flat: dict[str, str | int | float | bool] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif isinstance(value, (list, tuple, dict)):
            try:
                flat[key] = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                flat[key] = str(value)
        else:
            flat[key] = str(value)
    return flat


_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def extract_citation_indexes(text: str) -> list[int]:
    """Extrae índices de citas ``[1]``, ``[2]``… en orden de aparición, sin duplicados."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen
