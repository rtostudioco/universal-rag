"""Conector ``video``: MP4, MKV, MOV, WEBM… — extrae la pista de audio y la transcribe."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from universal_rag.connectors._media import VIDEO_EXTENSIONS, build_transcript_document
from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.transcript import Transcriber


class VideoConnector(BaseConnector):
    """Ingesta de video local: la pista de audio se transcribe con faster-whisper.

    faster-whisper decodifica el contenedor directamente (PyAV/ffmpeg), así que
    no hace falta extraer el audio manualmente. Las citas quedan acotadas por
    minuto según el momento del video en que se dijo cada cosa.
    """

    name: ClassVar[str] = "video"
    description: ClassVar[str] = (
        "Ingesta de video (MP4, MKV, MOV, WEBM…): transcribe la pista de audio "
        "localmente y permite citas por minuto."
    )
    schemes: ClassVar[tuple[str, ...]] = ("video",)
    extensions: ClassVar[tuple[str, ...]] = VIDEO_EXTENSIONS
    priority: ClassVar[int] = 10

    def can_handle(self, source: str) -> bool:
        source = (source or "").strip()
        if source.lower().startswith(("video://", "video:")):
            return True
        if "://" in source:
            return False
        return super().can_handle(source)

    def fetch(self, source: str, **options: Any) -> list:
        path = self._resolve_path(source)
        if not path.exists():
            raise ConnectorError(f"El archivo de video no existe: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ConnectorError(f"Formato de video no soportado: {path.suffix}")

        transcriber = Transcriber(
            model_size=options.get("asr_model"),
            device=options.get("asr_device"),
            compute_type=options.get("asr_compute_type"),
            language=options.get("language"),
        )
        segments, asr_info = transcriber.transcribe(path, language=options.get("language"))
        if not segments:
            raise ConnectorError(
                f"No se detectó voz transcribible en: {path.name} "
                "(pista de audio silenciosa, inexistente o idioma no reconocido)"
            )

        stat = path.stat()
        metadata = {
            "format": path.suffix.lower().lstrip("."),
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "file_size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
        document = build_transcript_document(
            content_source=self.name,
            uri=self._uri_for(path),
            segments=segments,
            asr_info=asr_info,
            title=options.get("title") or path.stem,
            metadata=metadata,
            media_type="video",
        )
        return [document]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _resolve_path(source: str) -> Path:
        raw = source.strip()
        for prefix in ("video://", "video:", "file://"):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix) :]
                break
        return Path(raw).expanduser()

    @staticmethod
    def _uri_for(path: Path) -> str:
        return f"video://{path.resolve()}"
