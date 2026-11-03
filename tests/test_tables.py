"""Tests de extracción de tablas a Markdown (Fase 5)."""

from __future__ import annotations

import sys

import pytest
from bs4 import BeautifulSoup
from test_connectors import build_minimal_pdf, make_docx

from universal_rag.connectors.file.connector import FileConnector, extract_pdf_tables
from universal_rag.connectors.url.connector import extract_html_with_tables
from universal_rag.core.tables import docx_tables, html_tables, table_to_markdown, tables_to_markdown


# ----------------------------------------------------------------- markdown
class TestTableToMarkdown:
    def test_basico(self):
        rows = [["Producto", "Precio"], ["Panel 450W", "199 €"], ["Inversor", "450 €"]]
        markdown = table_to_markdown(rows)
        assert markdown.splitlines() == [
            "| Producto | Precio |",
            "| --- | --- |",
            "| Panel 450W | 199 € |",
            "| Inversor | 450 € |",
        ]

    def test_rellena_celdas_faltantes(self):
        markdown = table_to_markdown([["a", "b", "c"], ["solo-una"]])
        assert markdown.splitlines()[2] == "| solo-una |  |  |"

    def test_escapa_pipes_y_limpia_saltos(self):
        markdown = table_to_markdown([["h"], ["con | pipe", "con\nsalto"]])
        assert "con \\| pipe" in markdown
        assert "con salto" in markdown

    def test_vacio_devuelve_cadena_vacia(self):
        assert table_to_markdown([]) == ""
        assert table_to_markdown([["", "  ", None]]) == ""

    def test_varias_tablas_con_encabezado(self):
        texto = tables_to_markdown(
            [[["h1"], ["v1"]], [[], []], [["h2"], ["v2"]]], heading="Tabla"
        )
        assert "### Tabla 1" in texto
        assert "### Tabla 2" in texto  # la tabla vacía se numera igual pero sin contenido
        assert "h1" in texto and "v2" in texto


# ------------------------------------------------------------------ fuentes
class TestHTMLTables:
    def test_html_tables_extrae_celdas(self):
        soup = BeautifulSoup(
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>",
            "lxml",
        )
        tables = html_tables(soup)
        assert tables == [[["A", "B"], ["1", "2"]]]

    def test_extract_html_anexa_markdown_sin_duplicar(self):
        html = """
        <html><head><title>Reporte</title></head><body><main>
        <p>Resumen del trimestre con métricas clave.</p>
        <table><tr><th>Métrica</th><th>Valor</th></tr>
        <tr><td>LTV</td><td>3.2x</td></tr></table>
        </main></body></html>
        """
        title, text, count = extract_html_with_tables(html)
        assert title == "Reporte"
        assert count == 1
        assert "Resumen del trimestre" in text
        assert "| Métrica | Valor |" in text
        assert "| LTV | 3.2x |" in text
        # sin duplicado: la tabla solo aparece una vez
        assert text.count("LTV") == 1

    def test_extract_html_sin_tablas(self):
        _, text, count = extract_html_with_tables(
            "<html><body><main><p>Solo párrafos.</p></main></body></html>"
        )
        assert count == 0 and "Solo párrafos." in text


class TestDocxTables:
    def test_docx_con_tabla(self, tmp_path):
        import docx as docx_module

        path = tmp_path / "informe.docx"
        make_docx(path, ["Informe trimestral."])
        document = docx_module.Document(str(path))
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Concepto"
        table.cell(0, 1).text = "Importe"
        table.cell(1, 0).text = "Paneles"
        table.cell(1, 1).text = "12.000 €"
        document.save(str(path))

        tables = docx_tables(docx_module.Document(str(path)))
        assert tables[0][0] == ["Concepto", "Importe"]

        docs = FileConnector().fetch(str(path))
        doc = docs[0]
        assert doc.metadata["tables"] == 1
        assert "### Tabla 1" in doc.content
        assert "| Paneles | 12.000 € |" in doc.content


class TestPdfTables:
    def test_extract_pdf_tables_sin_pdfplumber(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pdfplumber", None)  # import → ImportError
        assert extract_pdf_tables("cualquier.pdf") == []

    def test_pdf_con_tablas_vía_pdfplumber_falso(self, tmp_path, monkeypatch):
        class FakePage:
            def extract_tables(self):
                return [
                    [
                        ["Producto", "Precio"],
                        ["Panel 450W", "199 €"],
                        [None, None],  # fila basura: se limpia en el markdown
                    ]
                ]

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakePdfPlumber:
            @staticmethod
            def open(_path):
                return FakePdf()

        monkeypatch.setitem(sys.modules, "pdfplumber", FakePdfPlumber)
        pdf_path = tmp_path / "catalogo.pdf"
        pdf_path.write_bytes(build_minimal_pdf("Catalogo de productos"))

        content, pages, table_count = FileConnector._read_pdf(pdf_path)
        assert pages == 1
        assert table_count == 1
        assert "Catalogo de productos" in content
        assert "| Panel 450W | 199 € |" in content

    def test_extract_pdf_tables_fallo_no_rompe(self, tmp_path, monkeypatch):
        class Exploding:
            def __enter__(self):
                raise RuntimeError("pdf corrupto")

            def __exit__(self, *args):
                return False

        class FakePdfPlumber:
            @staticmethod
            def open(_path):
                return Exploding()

        monkeypatch.setitem(sys.modules, "pdfplumber", FakePdfPlumber)
        assert extract_pdf_tables(tmp_path / "roto.pdf") == []

    def test_pdf_integracion_sin_tablas(self, tmp_path):
        """Sin pdfplumber instalado, el PDF se ingesta igual (solo texto)."""
        pdf_path = tmp_path / "simple.pdf"
        pdf_path.write_bytes(build_minimal_pdf("Documento sin tablas"))
        docs = FileConnector().fetch(str(pdf_path))
        assert "Documento sin tablas" in docs[0].content
        assert docs[0].metadata["tables"] == 0


# ------------------------------------------------------------------ pipeline
def test_tablas_llegan_al_indice(pipeline, tmp_path):
    """El contenido de la tabla es recuperable tras la ingesta."""
    import docx as docx_module

    path = tmp_path / "precios.docx"
    make_docx(path, ["Lista de precios oficiales."])
    document = docx_module.Document(str(path))
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Servicio"
    table.cell(0, 1).text = "Tarifa"
    table.cell(1, 0).text = "Instalación solar"
    table.cell(1, 1).text = "4.500 €"
    document.save(str(path))

    result = pipeline.ingest(str(path))
    assert result.documents == 1
    answer = pipeline.query("¿Cuánto cuesta la instalación solar?")
    assert "4.500 €" in answer.answer


@pytest.mark.parametrize(
    ("rows", "esperado"),
    [
        ([["única"]], "| única |"),
        ([["a", "b"], ["c"]], "| a | b |"),
    ],
)
def test_markdown_minimo(rows, esperado):
    assert esperado in table_to_markdown(rows)
