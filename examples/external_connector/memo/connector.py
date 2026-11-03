"""Conector externo de ejemplo: esquema ``memo://`` (conocimiento en memoria).

Demuestra la convención de conectores externos:

    examples/external_connector/memo/connector.py

Uso:

    from universal_rag.core.connectors import ConnectorRegistry
    from universal_rag.core.pipeline import RAGPipeline

    registry = ConnectorRegistry.discover(["examples/external_connector"])
    rag = RAGPipeline(registry=registry)
    rag.ingest("memo://reunion-semanal")
"""

from __future__ import annotations

from typing import Any, ClassVar

from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.models import Document

_MEMOS = {
    "memo://reunion-semanal": (
        "Acta de reunión semanal: el equipo decidió migrar el despliegue a contenedores, "
        "revisar el presupuesto de infraestructura y publicar la versión 0.1 del RAG."
    ),
    "memo://cliente-aurora": (
        "Notas del cliente Aurora: solicita facturación mensual, soporte prioritario por "
        "correo y un informe mensual de métricas de uso."
    ),
}


class MemoConnector(BaseConnector):
    name: ClassVar[str] = "memo"
    description: ClassVar[str] = "Conector demo que sirve documentos desde un diccionario en memoria."
    schemes: ClassVar[tuple[str, ...]] = ("memo",)

    def can_handle(self, source: str) -> bool:
        return source.lower().startswith("memo://")

    def fetch(self, source: str, **options: Any) -> list[Document]:
        text = _MEMOS.get(source)
        if text is None:
            raise ConnectorError(
                f"Memo desconocido: {source}. Disponibles: {', '.join(sorted(_MEMOS))}"
            )
        return [
            self.build_document(
                content=text,
                source=self.name,
                uri=source,
                title=source.split("/")[-1].replace("-", " ").capitalize(),
                metadata={"demo": True},
            )
        ]
