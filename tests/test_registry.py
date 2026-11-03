"""Tests del registro de conectores (auto-descubrimiento)."""


import pytest

from universal_rag.connectors.text.connector import TextConnector
from universal_rag.core.connectors import (
    ConnectorNotFoundError,
    ConnectorRegistry,
)


@pytest.fixture()
def registry() -> ConnectorRegistry:
    return ConnectorRegistry.discover()


def test_descubre_builtins(registry):
    names = registry.names()
    assert "text" in names
    assert "file" in names
    assert "url" in names


def test_get_y_resolve(registry):
    assert isinstance(registry.get("text"), TextConnector)
    assert isinstance(registry.resolve("/algo/documento.pdf"), type(registry.get("file")))
    assert isinstance(registry.resolve("https://x.com"), type(registry.get("url")))
    assert isinstance(registry.resolve("text://contenido"), TextConnector)


def test_get_inexistente(registry):
    with pytest.raises(ConnectorNotFoundError):
        registry.get("no-existe")


def test_resolve_sin_candidato(registry):
    with pytest.raises(ConnectorNotFoundError):
        registry.resolve("ftp://servidor/archivo")


def test_describe(registry):
    info = {entry["name"]: entry for entry in registry.describe()}
    assert info["file"]["extensions"].startswith(".txt")
    assert info["url"]["schemes"] == "http, https"


def test_conector_externo(tmp_path, registry):
    """Un conector externo colgado en un directorio se descubre por convención."""
    external = tmp_path / "mi_conectores" / "wiki"
    external.mkdir(parents=True)
    (external / "connector.py").write_text(
        "from universal_rag.core.connectors import BaseConnector\n"
        "from universal_rag.core.models import Document\n\n"
        "class WikiConnector(BaseConnector):\n"
        "    name = 'wiki'\n"
        "    description = 'Conector de prueba'\n"
        "    schemes = ('wiki',)\n"
        "    def can_handle(self, source):\n"
        "        return source.startswith('wiki://')\n"
        "    def fetch(self, source, **options):\n"
        "        return [self.build_document(content='wiki', source=self.name, uri=source)]\n",
        encoding="utf-8",
    )
    registry2 = ConnectorRegistry.discover(extra_dirs=[str(tmp_path / "mi_conectores")])
    assert "wiki" in registry2.names()
    connector = registry2.resolve("wiki://España")
    assert connector.name == "wiki"
    docs = connector.fetch("wiki://España")
    assert docs[0].content == "wiki"


def test_conector_roto_aislado(tmp_path):
    """Un connector.py con error de sintaxis no derriba el registro."""
    bad = tmp_path / "bad" / "connector.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("esto no es python valido ((((", encoding="utf-8")
    registry = ConnectorRegistry.discover(extra_dirs=[str(tmp_path)])
    assert "bad" not in registry.names()
    records = [r for r in registry.records if r.origin.endswith("bad")]
    assert records and not records[0].ok


def test_registro_manual():
    registry = ConnectorRegistry()
    registry.register(TextConnector(), origin="manual")
    assert registry.names() == ["text"]
