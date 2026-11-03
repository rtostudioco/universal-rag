"""Tests del servidor MCP."""

import asyncio
import json

from universal_rag.core.collections import CollectionManager
from universal_rag.mcp_server import create_mcp_server


def _manager(settings):
    """Manager de colecciones con settings rápidos (memory/hash/echo)."""
    return CollectionManager(settings)


def test_crea_servidor_con_tools(settings):
    mcp = create_mcp_server(_manager(settings))
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "rag_ingest",
        "rag_ingest_text",
        "rag_query",
        "rag_stats",
        "rag_list_documents",
        "rag_delete_document",
        "rag_list_collections",
    } <= names


def test_tool_query_funciona_end_to_end(settings):
    mcp = create_mcp_server(_manager(settings))
    # llamada directa a la función subyacente de la tool
    result = asyncio.run(mcp.call_tool("rag_ingest_text", {"text": "Solar: energía limpia.", "title": "S"}))
    assert result
    result = asyncio.run(mcp.call_tool("rag_query", {"question": "¿qué es solar?"}))
    content = result[0][0].text if result and result[0] else ""
    assert "citations" in content or "answer" in content


def test_tool_stats(settings):
    mcp = create_mcp_server(_manager(settings))
    result = asyncio.run(mcp.call_tool("rag_stats", {}))
    assert result


def test_tool_colecciones_aisladas(settings):
    mcp = create_mcp_server(_manager(settings))
    asyncio.run(
        mcp.call_tool("rag_ingest_text", {"text": "Contrato legal: plazo de 30 días.", "title": "L", "collection": "legal"})
    )
    asyncio.run(mcp.call_tool("rag_ingest_text", {"text": "La energía solar es renovable.", "title": "S", "collection": "faq"}))
    raw = asyncio.run(mcp.call_tool("rag_list_collections", {}))
    part = raw[0] if isinstance(raw, tuple) else raw
    rows: list[dict] = []
    for item in part:
        if not getattr(item, "text", None):
            continue
        data = json.loads(item.text)
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.append(data)
    names = {row["name"] for row in rows}
    assert {"legal", "faq"} <= names


def test_pipeline_preconstruido_compat(settings):
    """Compatibilidad: create_mcp_server(pipeline=...) registra la colección default."""
    from universal_rag.core.pipeline import RAGPipeline

    mcp = create_mcp_server(pipeline=RAGPipeline(settings=settings))
    result = asyncio.run(mcp.call_tool("rag_stats", {}))
    assert result
