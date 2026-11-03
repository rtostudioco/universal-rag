"""Transcripciones con marcas de tiempo (Fase 2).

Núcleo de las "citas por minuto": :class:`Transcriber` usa **faster-whisper**
(CTranslate2, local y privado) para convertir audio/video en segmentos
``(inicio, fin, texto)``; esos segmentos viajan en los metadatos del
:class:`Document` y el :class:`~universal_rag.core.chunker.Chunker` los
convierte en fragmentos acotados por tiempo, que las citas muestran como
``[mm:ss–mm:ss]``.

El modelo se carga perezosamente: importar este módulo nunca descarga pesos.
Formatos soportados: todo lo que PyAV/ffmpeg decodifica (MP3, WAV, M4A, OGG,
FLAC, MP4, MKV, MOV, WEBM…).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from universal_rag.core.config import Settings
from universal_rag.core.utils import get_logger

logger = get_logger("transcript")


class ConnectorNoAudioError(RuntimeError):
    """El archivo de media no tiene pista de audio utilizable."""


@dataclass
class Segment:
    """Fragmento transcrito con marcas de tiempo en segundos."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(start=float(data["start"]), end=float(data["end"]), text=str(data.get("text", "")))


def format_timestamp(seconds: float | None) -> str:
    """Segundos → ``mm:ss`` (o ``h:mm:ss`` si supera una hora)."""
    if seconds is None:
        return ""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_range(start: float | None, end: float | None) -> str | None:
    """``[00:42–01:17]`` para citas; None si no hay tiempos."""
    if start is None and end is None:
        return None
    return f"[{format_timestamp(start)}–{format_timestamp(end)}]"


_UNSET = object()  # centinela: "no se pasó idioma; usa Settings"


class Transcriber:
    """ASR local con faster-whisper (carga perezosa del modelo)."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: Any = _UNSET,
    ) -> None:
        settings = Settings()
        self.model_size = model_size or settings.asr_model
        self.device = device or settings.asr_device
        self.compute_type = compute_type or settings.asr_compute_type
        self.language = settings.asr_language if language is _UNSET else language
        self._model: Any = None

    # ------------------------------------------------------------------ API
    def ensure_model(self) -> Any:
        """Carga (una sola vez) el modelo Whisper local."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "faster-whisper no está instalado. "
                    "Instálalo con: pip install 'universal-rag[media]'"
                ) from exc
            logger.info("Cargando modelo ASR '%s' (compute=%s)…", self.model_size, self.compute_type)
            self._model = WhisperModel(
                self.model_size,
                device=self.device or "auto",
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, media_path: str | Path, **options: Any) -> tuple[list[Segment], dict[str, Any]]:
        """Transcribe un archivo de audio/video y devuelve (segmentos, info).

        Info incluye ``language`` (detectada o forzada) y ``duration``.
        """
        model = self.ensure_model()
        language = options.get("language", self.language)
        kwargs: dict[str, Any] = {
            "vad_filter": options.get("vad_filter", True),  # filtra silencios
            "beam_size": options.get("beam_size", 5),
        }
        if language:
            kwargs["language"] = language

        audio = self._load_audio(media_path)

        logger.info("Transcribiendo %s (modelo %s)…", Path(media_path).name, self.model_size)
        raw_segments, info = model.transcribe(audio, **kwargs)
        segments = [
            Segment(start=float(seg.start), end=float(seg.end), text=str(seg.text).strip())
            for seg in raw_segments
            if str(seg.text).strip()
        ]
        meta = {
            "language": getattr(info, "language", None),
            "duration": float(getattr(info, "duration", 0.0) or 0.0) or len(audio) / 16000.0,
        }
        return segments, meta

    # ------------------------------------------------------------- internals
    @staticmethod
    def _load_audio(media_path: str | Path):
        """Decodifica audio/video a float32 mono 16 kHz, sin depender de la versión de PyAV.

        faster-whisper intenta ``decode_audio`` (compatible con versiones
        antiguas de PyAV); si falla por incompatibilidad, decodificamos con
        PyAV directamente y pasamos el array a Whisper (misma calidad).
        """
        import numpy as np

        try:
            from faster_whisper.audio import decode_audio

            return decode_audio(str(media_path), sampling_rate=16000)
        except TypeError:
            pass  # PyAV nuevo: decode_audio usa kwargs removidos
        except Exception:  # noqa: BLE001 — reintentamos con nuestro decodificador
            logger.debug("decode_audio de faster-whisper falló; usando decodificador propio", exc_info=True)

        try:
            import av

            container = av.open(str(media_path))
            try:
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                chunks: list[np.ndarray] = []
                for frame in container.decode(audio=0):
                    for resampled in resampler.resample(frame):
                        array = resampled.to_ndarray()
                        if array.ndim > 1:
                            array = array.reshape(-1)
                        chunks.append(array)
            finally:
                container.close()
            if not chunks:
                raise ConnectorNoAudioError(f"El archivo no contiene pista de audio: {media_path}")
            audio = np.concatenate(chunks).astype(np.float32) / 32768.0
            return audio
        except Exception as exc:
            raise RuntimeError(f"No se pudo decodificar el audio de {media_path}: {exc}") from exc
