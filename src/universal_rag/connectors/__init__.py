"""Conectores builtin de Universal RAG.

Convención (igual que los módulos de Universal MCP): cada subpaquete expone
su conector en ``connector.py`` con una clase que hereda de BaseConnector.
"""

from universal_rag.connectors.file.connector import FileConnector
from universal_rag.connectors.text.connector import TextConnector
from universal_rag.connectors.url.connector import URLConnector

__all__ = ["FileConnector", "TextConnector", "URLConnector"]
