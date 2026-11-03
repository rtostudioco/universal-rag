"""Conector ``youtube``: transcribe videos de YouTube con citas por minuto.

Descarga solo la pista de audio con yt-dlp y la transcribe localmente con
faster-whisper. El documento conserva ``video_id``, título, canal y duración;
cada fragmento cita su rango temporal ``[mm:ss–mm:ss]``.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from universal_rag.connectors._media import build_transcript_document
from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.transcript import Transcriber
from universal_rag.core.utils import get_logger

logger = get_logger("connectors.youtube")

_YT_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{6,20})"
)


class YouTubeConnector(BaseConnector):
    """Ingesta de videos de YouTube: descarga audio → ASR local → índice con minutos."""

    name: ClassVar[str] = "youtube"
    description: ClassVar[str] = (
        "Ingesta de videos de YouTube: descarga la pista de audio (yt-dlp), la "
        "transcribe localmente (faster-whisper) y permite citas por minuto."
    )
    schemes: ClassVar[tuple[str, ...]] = ("youtube",)
    priority: ClassVar[int] = 20  # antes que url (que solo maneja páginas web)

    def can_handle(self, source: str) -> bool:
        lower = (source or "").strip().lower()
        if lower.startswith(("youtube://", "yt://")):
            return True
        return bool(_YT_ID_RE.search(source))

    def fetch(self, source: str, **options: Any) -> list:
        video_id = self.extract_video_id(source)
        if not video_id:
            raise ConnectorError(f"No se pudo extraer el ID de video de: {source}")
        watch_url = f"https://www.youtube.com/watch?v={video_id}"

        workdir = Path(tempfile.mkdtemp(prefix="universal_rag_yt_"))
        try:
            audio_path, yt_info = self._download_audio(watch_url, workdir)
            transcriber = Transcriber(
                model_size=options.get("asr_model"),
                device=options.get("asr_device"),
                compute_type=options.get("asr_compute_type"),
                language=options.get("language"),
            )
            segments, asr_info = transcriber.transcribe(audio_path, language=options.get("language"))
            if not segments:
                raise ConnectorError(f"No se detectó voz transcribible en el video {video_id}")

            metadata = {
                "video_id": video_id,
                "uploader": yt_info.get("uploader"),
                "webpage_url": yt_info.get("webpage_url") or watch_url,
                "yt_duration": yt_info.get("duration"),
            }
            document = build_transcript_document(
                content_source=self.name,
                uri=yt_info.get("webpage_url") or watch_url,
                segments=segments,
                asr_info=asr_info,
                title=options.get("title") or yt_info.get("title") or f"YouTube {video_id}",
                metadata={k: v for k, v in metadata.items() if v is not None},
                media_type="video",
            )
            return [document]
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def extract_video_id(source: str) -> str | None:
        raw = (source or "").strip()
        if raw.lower().startswith(("youtube://", "yt://")):
            candidate = re.sub(r"^[a-z]+://", "", raw, flags=re.IGNORECASE).strip("/")
            return candidate or None
        match = _YT_ID_RE.search(raw)
        return match.group(1) if match else None

    @staticmethod
    def _download_audio(url: str, workdir: Path) -> tuple[Path, dict[str, Any]]:
        """Descarga la mejor pista de audio con yt-dlp (sin convertir si es posible)."""
        try:
            import yt_dlp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "yt-dlp no está instalado. Instálalo con: pip install 'universal-rag[media]'"
            ) from exc

        options = {
            "format": "bestaudio/best",
            "outtmpl": str(workdir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                path = Path(ydl.prepare_filename(info))
        except Exception as exc:  # noqa: BLE001 — yt-dlp lanza errores variados
            raise ConnectorError(f"Fallo descargando {url}: {exc}") from exc

        if not path.exists():
            # yt-dlp puede haber guardado con otra extensión; buscar por ID
            matches = sorted(workdir.glob("*"))
            if not matches:
                raise ConnectorError(f"yt-dlp no produjo ningún archivo para {url}")
            path = matches[0]

        yt_info = {
            "title": info.get("title") if isinstance(info, dict) else None,
            "uploader": info.get("uploader") if isinstance(info, dict) else None,
            "duration": info.get("duration") if isinstance(info, dict) else None,
            "webpage_url": info.get("webpage_url") if isinstance(info, dict) else None,
        }
        return path, yt_info
