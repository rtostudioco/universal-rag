"""Tests de los conectores de Fase 2: audio, video, youtube, drive (sin red real)."""

from __future__ import annotations

from pathlib import Path

import pytest

from universal_rag.connectors.audio.connector import AudioConnector
from universal_rag.connectors.drive.connector import DriveConnector
from universal_rag.connectors.video.connector import VideoConnector
from universal_rag.connectors.youtube.connector import YouTubeConnector
from universal_rag.core.connectors import ConnectorError, ConnectorRegistry
from universal_rag.core.transcript import Segment

SEGMENTS = [
    Segment(start=0.0, end=3.0, text="Bienvenidos a la reunión semanal."),
    Segment(start=3.0, end=7.0, text="El presupuesto de infraestructura se aprobó."),
    Segment(start=7.0, end=11.0, text="Y la migración a contenedores empieza el lunes."),
]

ASR_INFO = {"language": "es", "duration": 11.0}


@pytest.fixture()
def fake_transcriber(monkeypatch):
    """Reemplaza Transcriber en audio/video/drive por uno que no llama a Whisper."""

    class FakeTranscriber:
        def __init__(self, *args, **kwargs):
            self.model_size = "fake"

        def transcribe(self, path, **kwargs):
            return [s for s in SEGMENTS], dict(ASR_INFO)

    for module in (
        "universal_rag.connectors.audio.connector",
        "universal_rag.connectors.video.connector",
        "universal_rag.connectors.drive.connector",
        "universal_rag.connectors.youtube.connector",
    ):
        monkeypatch.setattr(f"{module}.Transcriber", FakeTranscriber)
    return FakeTranscriber


# ---------------------------------------------------------------------- audio
class TestAudioConnector:
    def test_can_handle(self):
        conn = AudioConnector()
        assert conn.can_handle("audio://podcast.mp3")
        assert conn.can_handle("/ruta/cancion.MP3")
        assert conn.can_handle("audio:/absoluto.wav")
        assert not conn.can_handle("https://x.com/a.mp3")  # remota → url/drive
        assert not conn.can_handle("/ruta/video.mp4")
        assert not conn.can_handle("/ruta/doc.pdf")

    def test_fetch_mp3(self, tmp_path, fake_transcriber):
        audio = tmp_path / "podcast.mp3"
        audio.write_bytes(b"id3-fake")
        docs = AudioConnector().fetch(str(audio))
        assert len(docs) == 1
        doc = docs[0]
        assert doc.source == "audio"
        assert doc.uri.startswith("audio://")
        assert doc.title == "podcast"
        assert doc.metadata["media_type"] == "audio"
        assert doc.metadata["transcribed"] is True
        assert doc.metadata["language"] == "es"
        assert doc.metadata["duration"] == 11.0
        assert len(doc.metadata["segments"]) == 3
        assert "presupuesto" in doc.content

    def test_fetch_esquema_audio(self, tmp_path, fake_transcriber):
        audio = tmp_path / "charla.ogg"
        audio.write_bytes(b"x")
        docs = AudioConnector().fetch(f"audio://{audio}")
        assert docs[0].uri == f"audio://{audio.resolve()}"

    def test_formato_no_soportado(self, tmp_path):
        (tmp_path / "x.txt").write_text("hola")
        with pytest.raises(ConnectorError):
            AudioConnector().fetch(str(tmp_path / "x.txt"))

    def test_inexistente(self):
        with pytest.raises(ConnectorError):
            AudioConnector().fetch("/no/existe/nunca.mp3")

    def test_sin_voz(self, tmp_path, monkeypatch):
        class SilentTranscriber:
            def __init__(self, *a, **k):
                pass

            def transcribe(self, path, **k):
                return [], {"language": "es", "duration": 3.0}

        monkeypatch.setattr("universal_rag.connectors.audio.connector.Transcriber", SilentTranscriber)
        audio = tmp_path / "silencio.mp3"
        audio.write_bytes(b"x")
        with pytest.raises(ConnectorError, match="No se detectó voz"):
            AudioConnector().fetch(str(audio))


# ---------------------------------------------------------------------- video
class TestVideoConnector:
    def test_can_handle(self):
        conn = VideoConnector()
        assert conn.can_handle("video://clase.mp4")
        assert conn.can_handle("/ruta/grabacion.mkv")
        assert not conn.can_handle("https://x.com/v.mp4")
        assert not conn.can_handle("/ruta/cancion.mp3")

    def test_fetch_mp4(self, tmp_path, fake_transcriber):
        video = tmp_path / "clase.mp4"
        video.write_bytes(b"ftypfake")
        docs = VideoConnector().fetch(str(video))
        doc = docs[0]
        assert doc.source == "video"
        assert doc.metadata["media_type"] == "video"
        assert doc.metadata["format"] == "mp4"
        assert doc.uri.startswith("video://")
        assert "contenedores" in doc.content


# -------------------------------------------------------------------- youtube
class TestYouTubeConnector:
    def test_can_handle(self):
        conn = YouTubeConnector()
        assert conn.can_handle("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert conn.can_handle("https://youtu.be/dQw4w9WgXcQ?t=30")
        assert conn.can_handle("https://www.youtube.com/shorts/AbCdEf12345")
        assert conn.can_handle("youtube://dQw4w9WgXcQ")
        assert not conn.can_handle("https://vimeo.com/12345")
        assert not conn.can_handle("https://ejemplo.com/watch?v=x")

    def test_extract_video_id(self):
        ext = YouTubeConnector.extract_video_id
        assert ext("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert ext("https://youtu.be/dQw4w9WgXcQ?t=30") == "dQw4w9WgXcQ"
        assert ext("youtube://abc123def45") == "abc123def45"
        assert ext("https://example.com/x") is None

    def test_fetch_con_mock(self, monkeypatch, fake_transcriber):
        def fake_download(url, workdir):
            audio = Path(workdir) / "abc123.webm"
            audio.write_bytes(b"audio-fake")
            return audio, {"title": "Reunión semanal", "uploader": "Canal Demo", "duration": 11.0,
                           "webpage_url": "https://www.youtube.com/watch?v=abc123def45"}

        monkeypatch.setattr(YouTubeConnector, "_download_audio", staticmethod(fake_download))
        docs = YouTubeConnector().fetch("https://www.youtube.com/watch?v=abc123def45")
        doc = docs[0]
        assert doc.source == "youtube"
        assert doc.title == "Reunión semanal"
        assert doc.metadata["video_id"] == "abc123def45"
        assert doc.metadata["uploader"] == "Canal Demo"
        assert doc.uri == "https://www.youtube.com/watch?v=abc123def45"
        assert "presupuesto" in doc.content

    def test_descarga_falla(self, monkeypatch):
        def boom(url, workdir):
            raise ConnectorError("red caída")

        monkeypatch.setattr(YouTubeConnector, "_download_audio", staticmethod(boom))
        with pytest.raises(ConnectorError):
            YouTubeConnector().fetch("https://youtu.be/abc123def45")


# ---------------------------------------------------------------------- drive
class TestDriveConnector:
    def test_can_handle(self):
        conn = DriveConnector()
        assert conn.can_handle("drive://1AbCdEfGhIjKlMnOpQrStUvWxYz0123456")
        assert conn.can_handle("https://drive.google.com/file/d/1AbCdEfGhIjK/view?usp=sharing")
        assert conn.can_handle("https://drive.google.com/uc?id=1AbCdEfGhIjK&export=download")
        assert not conn.can_handle("https://docs.google.com/document/d/xxx/edit")
        assert not conn.can_handle("https://ejemplo.com/archivo.pdf")

    def test_extract_file_id(self):
        ext = DriveConnector.extract_file_id
        assert ext("drive://1AbCdEfGhIjKlMnOpQrStUvWxYz0123456") == "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456"
        assert ext("https://drive.google.com/file/d/1AbCdEfGhIjK/view") == "1AbCdEfGhIjK"
        assert ext("https://drive.google.com/uc?id=1AbCdEfGhIjK&export=download") == "1AbCdEfGhIjK"
        assert ext("https://ejemplo.com") is None

    def test_fetch_documento(self, monkeypatch, tmp_path):
        local = tmp_path / "file.txt"
        local.write_text("Contenido del documento de Drive.", encoding="utf-8")
        monkeypatch.setattr(DriveConnector, "_download", staticmethod(lambda fid, wd: local))
        docs = DriveConnector().fetch("https://drive.google.com/file/d/1AbCdEfGhIjK/view")
        doc = docs[0]
        assert doc.source == "drive"
        assert doc.uri == "drive://1AbCdEfGhIjK"
        assert doc.metadata["drive_file_id"] == "1AbCdEfGhIjK"
        assert doc.metadata["original_format"] == "plain-text"
        assert "Contenido del documento" in doc.content

    def test_fetch_audio(self, monkeypatch, tmp_path, fake_transcriber):
        local = tmp_path / "file.mp3"
        local.write_bytes(b"audio-fake")
        monkeypatch.setattr(DriveConnector, "_download", staticmethod(lambda fid, wd: local))
        docs = DriveConnector().fetch("drive://1AbCdEfGhIjKlMnOpQrStUvWxYz0123456")
        doc = docs[0]
        assert doc.source == "drive"
        assert doc.metadata["media_type"] == "audio"
        assert "presupuesto" in doc.content

    def test_formato_no_soportado(self, monkeypatch, tmp_path):
        local = tmp_path / "file.exe"
        local.write_bytes(b"MZfake")
        monkeypatch.setattr(DriveConnector, "_download", staticmethod(lambda fid, wd: local))
        with pytest.raises(ConnectorError, match="no soportado"):
            DriveConnector().fetch("drive://1AbCdEfGhIjKlMnOpQrStUvWxYz0123456")


# ------------------------------------------------------------------- routing
class TestRouting:
    def test_registry_incluye_fase2(self):
        registry = ConnectorRegistry.discover()
        assert {"audio", "video", "youtube", "drive"} <= set(registry.names())

    def test_youtube_gana_a_url(self):
        registry = ConnectorRegistry.discover()
        connector = registry.resolve("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert connector.name == "youtube"

    def test_drive_gana_a_url(self):
        registry = ConnectorRegistry.discover()
        connector = registry.resolve("https://drive.google.com/file/d/1AbCdEfGhIjK/view")
        assert connector.name == "drive"

    def test_mp3_va_a_audio(self):
        registry = ConnectorRegistry.discover()
        assert registry.resolve("/musica/cancion.mp3").name == "audio"
        assert registry.resolve("audio://cancion.mp3").name == "audio"

    def test_mp4_va_a_video(self):
        registry = ConnectorRegistry.discover()
        assert registry.resolve("/peliculas/peli.mp4").name == "video"

    def test_url_normal_sigue_en_url(self):
        registry = ConnectorRegistry.discover()
        assert registry.resolve("https://ejemplo.com/articulo").name == "url"


# ------------------------------------------------------ citas por minuto E2E
class TestCitasPorMinutoPipeline:
    def test_ingesta_audio_query_con_minuto(self, pipeline, tmp_path, fake_transcriber):
        audio = tmp_path / "reunion.mp3"
        audio.write_bytes(b"fake")
        result = pipeline.ingest(str(audio))
        assert result.connector == "audio"
        assert result.documents == 1

        answer = pipeline.query("¿qué pasa con el presupuesto de infraestructura?")
        assert answer.citations, "debe citar la transcripción"
        citation = answer.citations[0]
        assert citation.source == "audio"
        assert citation.start_time is not None
        assert citation.end_time is not None
        assert citation.timestamp is not None and citation.timestamp.startswith("[")
        assert "presupuesto" in citation.quote.lower() or "–" in citation.quote
