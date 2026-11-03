"""Servidor MCP de Universal RAG.

Expone el pipeline como herramientas MCP (Model Context Protocol) — consumible
desde Claude Desktop, su Universal MCP, u otro cliente MCP::

    # stdio (Claude Desktop / mcp inspector)
    universal-rag mcp

    # streamable-http
    universal-rag mcp --transport streamable-http --port 8200
"""

from __future__ import annotations

import argparse
from typing import Any

from universal_rag.core.pipeline import RAGPipeline

INSTRUCTIONS = (
    "Universal RAG: ingesta universal de conocimiento (texto, archivos, URLs; "
    "audio/video/Drive en fases siguientes) y preguntas con respuestas citadas. "
    "Usa rag_query para preguntar; las respuestas incluyen citas verificables."
)


def create_mcp_server(pipeline: RAGPipeline | None = None):
    """Construye el FastMCP con las herramientas del RAG."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("universal-rag", instructions=INSTRUCTIONS)
    rag = pipeline or RAGPipeline()

    @mcp.tool()
    def rag_ingest(source: str, connector: str | None = None) -> dict[str, Any]:
        """Ingesta una fuente (ruta de archivo, URL, globo…) resolviendo el conector.

        Args:
            source: ruta, URL o fuente (ej. /docs/guia.pdf, https://ejemplo.com/articulo).
            connector: conector concreto a usar (opcional; auto-detección si se omite).
        """
        return rag.ingest(source, connector=connector).model_dump()

    @mcp.tool()
    def rag_ingest_text(text: str, title: str | None = None) -> dict[str, Any]:
        """Ingesta texto directo como documento de conocimiento.

        Args:
            text: contenido a ingestar.
            title: título opcional del documento.
        """
        return rag.ingest_text(text, title=title).model_dump()

    @mcp.tool()
    def rag_query(question: str, k: int | None = None) -> dict[str, Any]:
        """Responde una pregunta con citas a las fuentes ingestado.

        Args:
            question: pregunta en lenguaje natural.
            k: número de fragmentos a recuperar (opcional).
        """
        return rag.query(question, k=k).model_dump()

    @mcp.tool()
    def rag_stats() -> dict[str, Any]:
        """Estadísticas del pipeline: documentos, fragmentos, conectores, modelos."""
        return rag.stats()

    @mcp.tool()
    def rag_list_documents() -> list[dict[str, Any]]:
        """Lista los documentos ingestado con su origen y número de fragmentos."""
        return rag.documents()

    @mcp.tool()
    def rag_delete_document(document_id: str) -> dict[str, Any]:
        """Elimina un documento y todos sus fragmentos del índice."""
        deleted = rag.delete_document(document_id)
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


if __name__ == "__main__":
    main()
