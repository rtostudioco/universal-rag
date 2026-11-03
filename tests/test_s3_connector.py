"""Tests del conector s3 (Fase 5) — cliente boto3 falso, sin red real."""

from __future__ import annotations

import sys

import pytest
from test_connectors import build_minimal_pdf

from universal_rag.connectors.s3.connector import S3Connector
from universal_rag.core.connectors import ConnectorError, ConnectorRegistry


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """Doble de prueba con la superficie mínima de boto3 que usa el conector."""

    def __init__(self, objects: dict[str, bytes], listing: list[str] | None = None) -> None:
        self.objects = objects
        self.listing = listing if listing is not None else list(objects)
        self.get_calls: list[str] = []
        self.list_calls: list[dict] = []

    def get_object(self, Bucket: str, Key: str) -> dict:
        self.get_calls.append(Key)
        if Key not in self.objects:
            raise KeyError(Key)  # boto3 lanzaría NoSuchKey
        return {"Body": FakeBody(self.objects[Key]), "ETag": '"abc123"'}

    def list_objects_v2(self, Bucket: str, Prefix: str = "") -> dict:
        self.list_calls.append({"Bucket": Bucket, "Prefix": Prefix})
        contents = [{"Key": key} for key in self.listing if key.startswith(Prefix)]
        return {"Contents": contents, "IsTruncated": False}


@pytest.fixture()
def client() -> FakeS3Client:
    return FakeS3Client(
        {
            "docs/notas.txt": b"Notas compartidas desde el bucket.",
            "docs/guia.md": b"# Guia\n\nContenido markdown remoto.",
            "docs/informe.pdf": build_minimal_pdf("Informe remoto desde S3"),
            "docs/imagen.png": b"\x89PNG basura",
        }
    )


# --------------------------------------------------------------------- tests
class TestS3CanHandleAndParse:
    def test_can_handle(self):
        conn = S3Connector()
        assert conn.can_handle("s3://bucket/clave.pdf")
        assert conn.can_handle("S3://BUCKET/x")
        assert not conn.can_handle("https://bucket.s3.amazonaws.com/x")
        assert not conn.can_handle("/ruta/local.pdf")

    def test_parse_valido(self):
        assert S3Connector._parse("s3://mi-bucket/carpeta/informe.pdf") == (
            "mi-bucket",
            "carpeta/informe.pdf",
        )
        assert S3Connector._parse("s3://mi-bucket") == ("mi-bucket", "")

    def test_parse_sin_bucket(self):
        with pytest.raises(ConnectorError, match="bucket"):
            S3Connector._parse("s3://")

    def test_parse_esquema_invalido(self):
        with pytest.raises(ConnectorError, match="inválida"):
            S3Connector._parse("gs://bucket/x")


class TestS3Object:
    def test_objeto_txt(self, client):
        docs = S3Connector(client=client).fetch("s3://mi-bucket/docs/notas.txt")
        (doc,) = docs
        assert doc.source == "s3"
        assert doc.uri == "s3://mi-bucket/docs/notas.txt"
        assert doc.content == "Notas compartidas desde el bucket."
        assert doc.title == "notas"
        assert doc.metadata["bucket"] == "mi-bucket"
        assert doc.metadata["key"] == "docs/notas.txt"
        assert doc.metadata["etag"] == "abc123"
        assert doc.metadata["object_size"] == len(b"Notas compartidas desde el bucket.")
        assert doc.metadata["format"] == "plain-text"
        assert "file_path" not in doc.metadata  # no filtra rutas locales temporales

    def test_objeto_pdf(self, client):
        docs = S3Connector(client=client).fetch("s3://mi-bucket/docs/informe.pdf")
        assert "Informe remoto desde S3" in docs[0].content
        assert docs[0].metadata["format"] == "pdf"

    def test_extension_no_soportada(self, client):
        with pytest.raises(ConnectorError, match="no soportada"):
            S3Connector(client=client).fetch("s3://mi-bucket/docs/imagen.png")

    def test_objeto_inexistente(self, client):
        with pytest.raises(ConnectorError, match="descargar"):
            S3Connector(client=client).fetch("s3://mi-bucket/docs/falta.txt")


class TestS3Prefix:
    def test_prefijo_con_barra_final(self, client):
        docs = S3Connector(client=client).fetch("s3://mi-bucket/docs/", throttle=0)
        assert sorted(doc.uri for doc in docs) == [
            "s3://mi-bucket/docs/guia.md",
            "s3://mi-bucket/docs/informe.pdf",
            "s3://mi-bucket/docs/notas.txt",
        ]  # imagen.png filtrada por extensión

    def test_bucket_completo_sin_clave(self, client):
        docs = S3Connector(client=client).fetch("s3://mi-bucket")
        assert len(docs) == 3

    def test_opcion_prefix(self, client):
        docs = S3Connector(client=client).fetch(
            "s3://mi-bucket/docs", prefix=True, throttle=0
        )
        assert len(docs) == 3

    def test_limit(self, client):
        docs = S3Connector(client=client).fetch("s3://mi-bucket/docs/", limit=1, throttle=0)
        assert len(docs) == 1

    def test_prefijo_vacio(self):
        client = FakeS3Client({"docs/notas.txt": b"x"}, listing=[])
        with pytest.raises(ConnectorError, match="Sin objetos"):
            S3Connector(client=client).fetch("s3://mi-bucket/vacio/")

    def test_lote_con_objeto_roto_no_para(self, client):
        # .docx con bytes corruptos: falla la extracción pero el lote continúa
        client.objects["docs/roto.docx"] = b"esto no es un docx"
        client.listing.append("docs/roto.docx")
        docs = S3Connector(client=client).fetch("s3://mi-bucket/docs/", throttle=0)
        assert len(docs) == 3  # txt + md + pdf


class TestS3ClientFactory:
    def test_sin_boto3(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "boto3", None)  # import → ImportError
        with pytest.raises(ConnectorError, match="universal-rag\\[s3\\]"):
            S3Connector().fetch("s3://bucket/x.txt")

    def test_boto3_falso_con_opciones(self, monkeypatch):
        created: dict = {}

        class FakeModule:
            @staticmethod
            def client(service: str, **kwargs) -> str:
                created["service"] = service
                created.update(kwargs)
                return FakeS3Client({"x.txt": b"desde cliente real"})

        monkeypatch.setitem(sys.modules, "boto3", FakeModule())
        docs = S3Connector().fetch(
            "s3://b/x.txt",
            endpoint_url="http://localhost:9000",
            region="us-east-1",
            aws_access_key_id="AKIA",
            aws_secret_access_key="secreto",
        )
        assert created["service"] == "s3"
        assert created["endpoint_url"] == "http://localhost:9000"
        assert created["region_name"] == "us-east-1"
        assert docs[0].content == "desde cliente real"

    def test_endpoint_url_por_entorno(self, monkeypatch):
        created: dict = {}

        class FakeModule:
            @staticmethod
            def client(_service: str, **kwargs) -> FakeS3Client:
                created.update(kwargs)
                return FakeS3Client({"y.txt": b"ok"})

        monkeypatch.setitem(sys.modules, "boto3", FakeModule())
        monkeypatch.setenv("RAG_S3_ENDPOINT_URL", "http://minio.local:9000")
        S3Connector().fetch("s3://b/y.txt")
        assert created["endpoint_url"] == "http://minio.local:9000"


class TestS3Registro:
    def test_descubierto_y_ruteado(self):
        registry = ConnectorRegistry.discover()
        assert "s3" in registry.names()
        conn = registry.resolve("s3://bucket/archivo.pdf")
        assert conn.name == "s3"
