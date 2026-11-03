"""Servidor MCP de Universal RAG.

Expone el pipeline como herramientas MCP (Model Context Protocol) — consumible
desde Claude Desktop, su Universal MCP, u otro cliente MCP::

    # stdio (Claude Desktop / mcp inspector)
    universal-rag mcp

    # streamable-http
    universal-rag mcp --transport streamable-http --port 8200

Fase 4: las herramientas aceptan ``collection`` para operar sobre bases de
conocimiento aisladas y expone ``rag_list_collections``.
"""

from __future__ import annotations

import argparse
from typing import Any

from universal_rag.core.collections import CollectionManager
from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline

INSTRUCTIONS = (
    "Universal RAG: ingesta universal de conocimiento (texto, archivos, URLs, "
    "audio, video, YouTube, Drive) y preguntas con respuestas citadas. "
    "Usa rag_query para preguntar; las respuestas incluyen citas verificables. "
    "Las herramientas aceptan 'collection' para aislar bases de conocimiento."
)


def create_mcp_server(
    manager: CollectionManager | None = None,
    *,
    pipeline: RAGPipeline | None = None,
    settings: Settings | None = None,
):
    """Construye el FastMCP con las herramientas del RAG.

    ``pipeline`` (deprecado) registra un pipeline pre-construido como colección
    por defecto; la forma recomendada es pasar un :class:`CollectionManager`.
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("universal-rag", instructions=INSTRUCTIONS)
    rag = manager or CollectionManager(settings)
    if pipeline is not None:
        rag.register(rag.settings.default_collection, pipeline)

    @mcp.tool()
    def rag_ingest(
        source: str, connector: str | None = None, collection: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Ingesta una fuente (ruta de archivo, URL, globo…) resolviendo el conector.

        Args:
            source: ruta, URL o fuente (ej. /docs/guia.pdf, https://ejemplo.com/articulo).
            connector: conector concreto a usar (opcional; auto-detección si se omite).
            collection: colección destino (opcional; 'default' si se omite).
            force: re-ingestar aunque el contenido no haya cambiado.
        """
        return rag.get(collection).ingest(source, connector=connector, force=force).model_dump()

    @mcp.tool()
    def rag_ingest_text(
        text: str, title: str | None = None, collection: str | None = None
    ) -> dict[str, Any]:
        """Ingesta texto directo como documento de conocimiento.

        Args:
            text: contenido a ingestar.
            title: título opcional del documento.
            collection: colección destino (opcional; 'default' si se omite).
        """
        return rag.get(collection).ingest_text(text, title=title).model_dump()

    @mcp.tool()
    def rag_query(question: str, k: int | None = None, collection: str | None = None) -> dict[str, Any]:
        """Responde una pregunta con citas a las fuentes ingestado.

        Args:
            question: pregunta en lenguaje natural.
            k: número de fragmentos a recuperar (opcional).
            collection: colección donde consultar (opcional; 'default' si se omite).
        """
        return rag.get(collection).query(question, k=k).model_dump()

    @mcp.tool()
    def rag_list_collections() -> list[dict[str, Any]]:
        """Lista las colecciones activas con documentos y fragmentos."""
        return rag.list_collections()

    @mcp.tool()
    def rag_stats(collection: str | None = None) -> dict[str, Any]:
        """Estadísticas del pipeline: documentos, fragmentos, conectores, modelos.

        Args:
            collection: colección a inspeccionar (opcional; 'default' si se omite).
        """
        return rag.get(collection).stats()

    @mcp.tool()
    def rag_list_documents(collection: str | None = None) -> list[dict[str, Any]]:
        """Lista los documentos ingestado con su origen y número de fragmentos.

        Args:
            collection: colección a listar (opcional; 'default' si se omite).
        """
        return rag.get(collection).documents()

    @mcp.tool()
    def rag_delete_document(document_id: str, collection: str | None = None) -> dict[str, Any]:
        """Elimina un documento y todos sus fragmentos del índice.

        Args:
            document_id: id del documento a eliminar.
            collection: colección donde eliminar (opcional; 'default' si se omite).
        """
        deleted = rag.get(collection).delete_document(document_id)
        return {"deleted": deleted, "document_id": document_id}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(prog="universal-rag-mcp", description="Servidor MCP de Universal RAG")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transporte MCP (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    mcp = create_mcp_server()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
