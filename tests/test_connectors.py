"""Tests de conectores builtin: text, file, url (sin red real)."""

from __future__ import annotations

import httpx
import pytest

from universal_rag.connectors.file.connector import FileConnector
from universal_rag.connectors.text.connector import TextConnector
from universal_rag.connectors.url.connector import URLConnector, extract_html
from universal_rag.core.connectors import ConnectorError


# ---------------------------------------------------------------------- text
class TestTextConnector:
    def test_can_handle(self):
        conn = TextConnector()
        assert conn.can_handle("text://algo")
        assert conn.can_handle("TEXT:algo")
        assert not conn.can_handle("/ruta/archivo.txt")
        assert not conn.can_handle("https://ejemplo.com")

    def test_from_text(self):
        document = TextConnector().from_text("Contenido de prueba", title="Prueba")
        assert document.content == "Contenido de prueba"
        assert document.source == "text"
        assert document.uri == "text://Prueba"
        assert document.title == "Prueba"

    def test_fetch_con_content(self):
        docs = TextConnector().fetch("text:", content="hola", title="t")
        assert len(docs) == 1 and docs[0].content == "hola"

    def test_fetch_sin_content_falla(self):
        with pytest.raises(ConnectorError):
            TextConnector().fetch("text:")


# ---------------------------------------------------------------------- file
class TestFileConnector:
    def test_can_handle(self, tmp_path):
        conn = FileConnector()
        f = tmp_path / "a.pdf"
        f.write_bytes(b"x")
        assert conn.can_handle(str(f))
        assert conn.can_handle(f"file://{f}")
        assert conn.can_handle(str(tmp_path / "*.md"))
        assert not conn.can_handle(str(tmp_path / "b.xyz"))
        assert not conn.can_handle("https://ejemplo.com")

    def test_txt_y_md(self, tmp_path):
        (tmp_path / "notas.txt").write_text("Primera línea.\nSegunda línea.", encoding="utf-8")
        (tmp_path / "guia.md").write_text("# Guía\n\nContenido markdown útil.", encoding="utf-8")
        docs = FileConnector().fetch(str(tmp_path / "notas.txt"))
        assert len(docs) == 1
        assert docs[0].metadata["format"] == "plain-text"
        assert "Segunda línea" in docs[0].content
        assert docs[0].title == "notas"
        docs_md = FileConnector().fetch(f"file://{tmp_path / 'guia.md'}")
        assert docs_md[0].metadata["format"] == "markdown"

    def test_csv(self, tmp_path):
        (tmp_path / "datos.csv").write_text("nombre,rol\nAna,admin\nLuis,user\n", encoding="utf-8")
        docs = FileConnector().fetch(str(tmp_path / "datos.csv"))
        content = docs[0].content
        assert "nombre | rol" in content
        assert "nombre: Ana | rol: admin" in content
        assert docs[0].metadata["format"] == "csv"

    def test_pdf(self, tmp_path):
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(build_minimal_pdf("Hola RAG universal desde PDF"))
        docs = FileConnector().fetch(str(pdf_path))
        assert docs[0].metadata["format"] == "pdf"
        assert docs[0].metadata["pages"] == 1
        assert "Hola RAG universal desde PDF" in docs[0].content

    def test_docx(self, tmp_path):
        docx_path = tmp_path / "informe.docx"
        make_docx(docx_path, ["Párrafo uno del informe.", "Párrafo dos del informe."])
        docs = FileConnector().fetch(str(docx_path))
        assert docs[0].metadata["format"] == "docx"
        assert "Párrafo dos del informe." in docs[0].content

    def test_glob(self, tmp_path):
        for name in ("a.md", "b.md", "c.txt"):
            (tmp_path / name).write_text(f"contenido {name}", encoding="utf-8")
        docs = FileConnector().fetch(str(tmp_path / "*.md"))
        assert len(docs) == 2

    def test_archivo_inexistente(self):
        with pytest.raises(ConnectorError):
            FileConnector().fetch("/no/existe/nunca.txt")

    def test_extension_no_soportada(self, tmp_path):
        (tmp_path / "bin.xyz").write_text("data")
        docs = FileConnector().fetch(str(tmp_path / "bin.xyz"))
        assert docs == []


# ----------------------------------------------------------------------- url
HTML_EJEMPLO = """
<html><head><title>Página de prueba</title></head>
<body>
<nav>Navegación que debe ignorarse</nav>
<script>console.log("script malicioso")</script>
<style>.a { color: red }</style>
<main>
<h1>Contenido principal</h1>
<p>Este es el artículo con información relevante sobre energía solar.</p>
</main>
<footer>Pie de página irrelevante</footer>
</body></html>
"""


class TestURLConnector:
    def test_can_handle(self):
        conn = URLConnector()
        assert conn.can_handle("https://ejemplo.com/a")
        assert conn.can_handle("http://ejemplo.com")
        assert not conn.can_handle("ftp://ejemplo.com")
        assert not conn.can_handle("/archivo.pdf")

    def test_extract_html_bs4(self):
        title, text = extract_html(HTML_EJEMPLO)
        assert "Contenido principal" in text
        assert "energía solar" in text
        assert "script malicioso" not in text
        assert "Navegación que debe ignorarse" not in text
        assert "Pie de página irrelevante" not in text
        assert title == "Página de prueba"

    def test_fetch_con_mock(self, monkeypatch):
        def fake_get(url, **kwargs):
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                text=HTML_EJEMPLO,
                headers={"content-type": "text/html; charset=utf-8"},
                request=request,
            )

        monkeypatch.setattr("universal_rag.connectors.url.connector.httpx.get", fake_get)
        docs = URLConnector().fetch("https://ejemplo.com/articulo")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.uri == "https://ejemplo.com/articulo"
        assert doc.metadata["status_code"] == 200
        assert doc.metadata["content_type"] == "text/html"
        assert "energía solar" in doc.content

    def test_fetch_error_http(self, monkeypatch):
        def fake_get(url, **kwargs):
            request = httpx.Request("GET", url)
            return httpx.Response(404, text="no existe", request=request)

        monkeypatch.setattr("universal_rag.connectors.url.connector.httpx.get", fake_get)
        with pytest.raises(httpx.HTTPStatusError):
            URLConnector().fetch("https://ejemplo.com/404")


# ------------------------------------------------------------------- helpers
def build_minimal_pdf(text: str) -> bytes:
    """Genera un PDF mínimo de 1 página con texto (sin dependencias extra)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(_pdf_stream(text))).encode() + b" >>\nstream\n"
        + _pdf_stream(text)
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    return bytes(out)


def _pdf_stream(text: str) -> bytes:
    return f"BT /F1 14 Tf 72 720 Td ({text}) Tj ET".encode()


def make_docx(path, paragraphs: list[str]) -> None:
    import docx

    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(str(path))
