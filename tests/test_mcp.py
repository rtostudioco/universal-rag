"""Tests del servidor MCP."""

import asyncio

from universal_rag.mcp_server import create_mcp_server


def _pipeline(settings):
    from universal_rag.core.pipeline import RAGPipeline

    return RAGPipeline(settings=settings)


def test_crea_servidor_con_tools(settings):
    mcp = create_mcp_server(_pipeline(settings))
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "rag_ingest",
        "rag_ingest_text",
        "rag_query",
        "rag_stats",
        "rag_list_documents",
        "rag_delete_document",
    } <= names


def test_tool_query_funciona_end_to_end(settings):
    mcp = create_mcp_server(_pipeline(settings))
    # llamada directa a la función subyacente de la tool
    result = asyncio.run(mcp.call_tool("rag_ingest_text", {"text": "Solar: energía limpia.", "title": "S"}))
    assert result
    result = asyncio.run(mcp.call_tool("rag_query", {"question": "¿qué es solar?"}))
    content = result[0][0].text if result and result[0] else ""
    assert "citations" in content or "answer" in content


def test_tool_stats(settings):
    mcp = create_mcp_server(_pipeline(settings))
    result = asyncio.run(mcp.call_tool("rag_stats", {}))
    assert result
