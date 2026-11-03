"""Conector ``audio``: MP3, WAV, M4A, OGG, FLAC… con ASR local y citas por minuto."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from universal_rag.connectors._media import AUDIO_EXTENSIONS, build_transcript_document
from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.transcript import Transcriber


class AudioConnector(BaseConnector):
    """Transcribe archivos de audio localmente (faster-whisper) y los indexa.

    Cada fragmento conserva su rango temporal, así que las citas muestran
    ``[mm:ss–mm:ss]``. Sin APIs externas: todo corre en tu máquina.
    """

    name: ClassVar[str] = "audio"
    description: ClassVar[str] = (
        "Ingesta de audio (MP3, WAV, M4A, OGG, FLAC…) con transcripción local "
        "(faster-whisper) y citas por minuto."
    )
    schemes: ClassVar[tuple[str, ...]] = ("audio",)
    extensions: ClassVar[tuple[str, ...]] = AUDIO_EXTENSIONS
    priority: ClassVar[int] = 10  # antes que file/url cuando hay solape

    def can_handle(self, source: str) -> bool:
        source = (source or "").strip()
        if source.lower().startswith(("audio://", "audio:")):
            return True
        if "://" in source:
            return False  # URLs remotas: las maneja url/youtube/drive
        return super().can_handle(source)

    def fetch(self, source: str, **options: Any) -> list:
        path = self._resolve_path(source)
        if not path.exists():
            raise ConnectorError(f"El archivo de audio no existe: {path}")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ConnectorError(f"Formato de audio no soportado: {path.suffix}")

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
                "(audio silencioso o idioma no reconocido)"
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
            media_type="audio",
        )
        return [document]

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _resolve_path(source: str) -> Path:
        raw = source.strip()
        for prefix in ("audio://", "audio:", "file://"):
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix) :]
                break
        return Path(raw).expanduser()

    @staticmethod
    def _uri_for(path: Path) -> str:
        return f"audio://{path.resolve()}"
