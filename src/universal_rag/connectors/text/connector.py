"""Conector ``text``: ingesta de texto directo (inline)."""

from __future__ import annotations

from typing import Any, ClassVar

from universal_rag.core.connectors import BaseConnector
from universal_rag.core.models import Document
from universal_rag.core.utils import make_id


class TextConnector(BaseConnector):
    """Normaliza texto en crudo (string, ``text://…``) a Document."""

    name: ClassVar[str] = "text"
    description: ClassVar[str] = (
        "Ingesta de texto directo: contenido inline con título y metadatos opcionales."
    )
    schemes: ClassVar[tuple[str, ...]] = ("text",)

    def can_handle(self, source: str) -> bool:
        return (source or "").strip().lower().startswith("text:")

    def fetch(self, source: str, **options: Any) -> list[Document]:
        """Acepta ``text://<contenido>`` o ``fetch("text:", content="…")``."""
        content = options.pop("content", None)
        if content is None and source.startswith("text://"):
            content = source[len("text://") :]
        if not content:
            from universal_rag.core.connectors import ConnectorError

            raise ConnectorError("TextConnector requiere el contenido (content=… o text://…)")
        title = options.pop("title", None)
        metadata = options.pop("metadata", None)
        return [self.from_text(str(content), title=title, metadata=metadata)]

    def from_text(
        self,
        text: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Crea un Document a partir de texto en crudo (lo usa pipeline.ingest_text)."""
        uri = f"text://{title}" if title else f"text://{make_id('t', text)}"
        return self.build_document(
            content=text,
            source=self.name,
            uri=uri,
            title=title,
            metadata=metadata,
        )
