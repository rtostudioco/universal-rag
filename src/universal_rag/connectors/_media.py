"""Helpers compartidos para conectores de media (Fase 2)."""

from __future__ import annotations

from typing import Any

from universal_rag.core.models import Document

AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".flac",
    ".opus",
    ".wma",
)
VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".m4v",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mpg",
    ".mpeg",
    ".ts",
)


def build_transcript_document(
    *,
    content_source: str,
    uri: str,
    segments: list[Any],
    asr_info: dict[str, Any],
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    media_type: str = "audio",
) -> Document:
    """Construye un Document de transcripción con segmentos temporales.

    El documento guarda los segmentos ``(start, end, text)`` en metadatos; el
    chunker los convierte en fragmentos acotados por tiempo para las citas
    por minuto.
    """
    from universal_rag.core.connectors import BaseConnector

    segment_dicts = [seg.to_dict() if hasattr(seg, "to_dict") else dict(seg) for seg in segments]
    full_text = " ".join(str(s.get("text", "")).strip() for s in segment_dicts if s.get("text"))
    doc_metadata = {
        **(metadata or {}),
        "media_type": media_type,
        "transcribed": True,
        "language": asr_info.get("language"),
        "duration": asr_info.get("duration"),
        "segments_count": len(segment_dicts),
        "segments": segment_dicts,
    }
    return BaseConnector.build_document(
        content=full_text,
        source=content_source,
        uri=uri,
        title=title,
        metadata=doc_metadata,
    )
