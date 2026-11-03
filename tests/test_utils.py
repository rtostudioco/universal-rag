"""Tests de utilidades del núcleo."""

from universal_rag.core.utils import (
    extract_citation_indexes,
    flatten_metadata,
    make_id,
    normalize_whitespace,
    truncate,
)


def test_make_id_estable_y_determinista():
    assert make_id("doc", "/ruta.pdf") == make_id("doc", "/ruta.pdf")
    assert make_id("a") != make_id("b")
    assert len(make_id("x")) == 16


def test_normalize_whitespace():
    text = "Hola\r\n\r\n\r\nmundo   multiple    espacios\n\n\n\nfin"
    normalized = normalize_whitespace(text)
    assert normalized == "Hola\n\nmundo multiple espacios\n\nfin"


def test_truncate():
    assert truncate("corto", 10) == "corto"
    result = truncate("a" * 50, 10)
    assert len(result) == 10
    assert result.endswith("…")


def test_extract_citation_indexes():
    assert extract_citation_indexes("Según [1] y [3], también [1] otra vez [12]") == [1, 3, 12]
    assert extract_citation_indexes("sin citas") == []
    assert extract_citation_indexes("corchetes [no es numero]") == []


def test_flatten_metadata():
    flat = flatten_metadata({"a": 1, "b": None, "c": [1, 2], "d": "x", "e": {"k": "v"}})
    assert flat["a"] == 1
    assert flat["d"] == "x"
    assert flat["c"] == "[1, 2]"
    assert flat["e"] == '{"k": "v"}'
    assert "b" not in flat
