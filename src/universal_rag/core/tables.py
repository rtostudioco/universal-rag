"""Extracción de tablas a Markdown (Fase 5).

Las tablas de DOCX y HTML (y de PDF vía ``pdfplumber``, extra opcional
``[tables]``) se normalizan a bloques Markdown y se anexan al texto del
documento: así el chunker las indexa y las preguntas sobre celdas concretas
recuperan la tabla completa como contexto citable.
"""

from __future__ import annotations

from typing import Any

from universal_rag.core.utils import get_logger

logger = get_logger("tables")


def _cell(value: Any) -> str:
    """Celda plana, sin saltos y con ``|`` escapado."""
    return " ".join(str(value if value is not None else "").split()).replace("|", "\\|")


def table_to_markdown(rows: list[list[Any]]) -> str:
    """Convierte filas (lista de listas de celdas) en una tabla Markdown.

    La primera fila no vacía actúa de cabecera. Devuelve ``""`` si no hay
    contenido útil (todas las celdas vacías).
    """
    clean = [
        [_cell(cell) for cell in row]
        for row in rows
        if row and any(cell is not None and str(cell).strip() for cell in row)
    ]
    if not clean:
        return ""
    width = max(len(row) for row in clean)
    clean = [row + [""] * (width - len(row)) for row in clean]
    header, *body = clean
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def tables_to_markdown(
    tables: list[list[list[Any]]], heading: str = "Tabla"
) -> str:
    """Renderiza varias tablas como bloques ``### Tabla N`` separados."""
    blocks = [table for table in (table_to_markdown(t) for t in tables) if table]
    if not blocks:
        return ""
    parts = [f"### {heading} {index}\n\n{block}\n" for index, block in enumerate(blocks, start=1)]
    return "\n".join(parts).strip()


def html_tables(soup: Any) -> list[list[list[str]]]:
    """Extrae tablas de un árbol BeautifulSoup como filas de celdas de texto.

    Cada tabla es ``list[fila]`` y cada fila ``list[celda]`` (``<th>``/``<td>``).
    """
    extracted: list[list[list[str]]] = []
    for table_tag in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table_tag.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if cells:
                rows.append([cell.get_text(" ", strip=True) for cell in cells])
        extracted.append(rows)
    return extracted


def docx_tables(document: Any) -> list[list[list[str]]]:
    """Extrae tablas de un ``docx.Document`` (python-docx) como filas de texto."""
    extracted: list[list[list[str]]] = []
    for table in document.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        extracted.append(rows)
    return extracted
