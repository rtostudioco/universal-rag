"""Conector ``drive``: archivos públicos/compartidos de Google Drive.

Resuelve ``drive://<fileId>``, URLs ``drive.google.com/file/d/<id>/…`` o
``uc?id=<id>``, descarga el archivo (gdown) y lo procesa según su formato:

* audio/video → transcripción local con citas por minuto;
* PDF/DOCX/TXT/MD/CSV → extracción de texto (reutiliza el conector file).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from universal_rag.connectors._media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, build_transcript_document
from universal_rag.connectors.file.connector import FileConnector
from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.transcript import Transcriber

_DRIVE_ID_PATTERNS = (
    re.compile(r"drive\.google\.com/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]{10,})"),
)

_TEXT_EXTENSIONS = FileConnector.extensions


class DriveConnector(BaseConnector):
    """Ingesta de archivos de Google Drive (enlaces públicos o compartidos)."""

    name: ClassVar[str] = "drive"
    description: ClassVar[str] = (
        "Ingesta de archivos de Google Drive (drive://<id> o URL compartida): "
        "documentos con extracción de texto y audio/video con citas por minuto."
    )
    schemes: ClassVar[tuple[str, ...]] = ("drive",)
    priority: ClassVar[int] = 15  # antes que url

    def can_handle(self, source: str) -> bool:
        raw = (source or "").strip()
        if raw.lower().startswith(("drive://", "gdrive://")):
            return True
        return any(p.search(raw) for p in _DRIVE_ID_PATTERNS)

    def fetch(self, source: str, **options: Any) -> list:
        file_id = self.extract_file_id(source)
        if not file_id:
            raise ConnectorError(f"No se pudo extraer el fileId de Drive de: {source}")

        workdir = Path(tempfile.mkdtemp(prefix="universal_rag_drive_"))
        try:
            path = self._download(file_id, workdir)
            drive_metadata = {
                "drive_file_id": file_id,
                "source_url": f"https://drive.google.com/file/d/{file_id}/view",
            }
            suffix = path.suffix.lower()

            if suffix in AUDIO_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
                return self._transcribe(path, file_id, options, drive_metadata)
            if suffix in _TEXT_EXTENSIONS:
                return self._extract_text(path, file_id, options, drive_metadata)
            raise ConnectorError(
                f"Formato no soportado para Drive {file_id}: '{suffix or 'desconocido'}'. "
                f"Soportados: documentos {', '.join(_TEXT_EXTENSIONS)} y media "
                f"{', '.join(AUDIO_EXTENSIONS + VIDEO_EXTENSIONS)}"
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def extract_file_id(source: str) -> str | None:
        raw = (source or "").strip()
        if raw.lower().startswith(("drive://", "gdrive://")):
            candidate = re.sub(r"^[a-z]+://", "", raw, flags=re.IGNORECASE).strip("/")
            return candidate or None
        for pattern in _DRIVE_ID_PATTERNS:
            match = pattern.search(raw)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _download(file_id: str, workdir: Path) -> Path:
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("gdown no está instalado (pip install universal-rag)") from exc
        try:
            output = gdown.download(id=file_id, output=str(workdir / "file"), quiet=True)
        except Exception as exc:  # noqa: BLE001 — gdown lanza errores variados
            raise ConnectorError(
                f"No se pudo descargar el archivo de Drive {file_id}: {exc}. "
                "Asegúrate de que el enlace sea público o compartido."
            ) from exc
        if not output or not Path(output).exists():
            raise ConnectorError(f"La descarga de Drive no produjo archivo para {file_id}")
        return Path(output)

    def _transcribe(
        self, path: Path, file_id: str, options: dict[str, Any], drive_metadata: dict[str, Any]
    ) -> list:
        transcriber = Transcriber(
            model_size=options.get("asr_model"),
            device=options.get("asr_device"),
            compute_type=options.get("asr_compute_type"),
            language=options.get("language"),
        )
        segments, asr_info = transcriber.transcribe(path, language=options.get("language"))
        if not segments:
            raise ConnectorError(f"No se detectó voz transcribible en el archivo de Drive {file_id}")
        document = build_transcript_document(
            content_source=self.name,
            uri=f"drive://{file_id}",
            segments=segments,
            asr_info=asr_info,
            title=options.get("title") or path.stem,
            metadata=drive_metadata,
            media_type="video" if path.suffix.lower() in VIDEO_EXTENSIONS else "audio",
        )
        return [document]

    def _extract_text(
        self, path: Path, file_id: str, options: dict[str, Any], drive_metadata: dict[str, Any]
    ) -> list:
        # Reutiliza el extractor de FileConnector y re-origina el documento a Drive.
        local_docs = FileConnector().fetch(str(path))
        if not local_docs or not local_docs[0].content.strip():
            raise ConnectorError(f"No se pudo extraer texto del archivo de Drive {file_id}")
        base = local_docs[0]
        uri = f"drive://{file_id}"
        document = self.build_document(
            content=base.content,
            source=self.name,
            uri=uri,
            title=options.get("title") or base.title,
            metadata={
                **base.metadata,
                **drive_metadata,
                "original_format": base.metadata.get("format"),
            },
            created_at=base.created_at,
        )
        return [document]
