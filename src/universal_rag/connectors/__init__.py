"""Conectores builtin de Universal RAG.

Convención (igual que los módulos de Universal MCP): cada subpaquete expone
su conector en ``connector.py`` con una clase que hereda de BaseConnector.

Fase 1: text, file, url · Fase 2: audio, video, youtube, drive.
"""

from universal_rag.connectors.audio.connector import AudioConnector
from universal_rag.connectors.drive.connector import DriveConnector
from universal_rag.connectors.file.connector import FileConnector
from universal_rag.connectors.text.connector import TextConnector
from universal_rag.connectors.url.connector import URLConnector
from universal_rag.connectors.video.connector import VideoConnector
from universal_rag.connectors.youtube.connector import YouTubeConnector

__all__ = [
    "AudioConnector",
    "DriveConnector",
    "FileConnector",
    "TextConnector",
    "URLConnector",
    "VideoConnector",
    "YouTubeConnector",
]
