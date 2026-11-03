"""Conector ``s3``: objetos S3 / MinIO con extracción de texto local.

Fase 5. URIs soportadas::

    s3://bucket/clave/informe.pdf     objeto concreto (PDF/DOCX/TXT/MD/CSV)
    s3://bucket/prefijo/              lista el prefijo e ingesta lo soportado
    s3://bucket                       todo el bucket (prefijo "")

Requiere ``boto3``: ``pip install 'universal-rag[s3]'``. Las credenciales se
resuelven con la cadena estándar de AWS (env ``AWS_*``, ``~/.aws/credentials``,
roles IAM…) o con las opciones ``aws_access_key_id``/``aws_secret_access_key``.
Para MinIO/R2/Spaces pasa ``endpoint_url=`` (o ``RAG_S3_ENDPOINT_URL``).

Opciones de ingesta:
    prefix: trata la clave como prefijo y lista los objetos debajo.
    limit: máximo de objetos en modo prefijo (default 50).
    endpoint_url / region: configuración del cliente (MinIO, R2, …).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from universal_rag.connectors.file.connector import FileConnector
from universal_rag.core.connectors import BaseConnector, ConnectorError
from universal_rag.core.models import Document
from universal_rag.core.utils import get_logger

logger = get_logger("connectors.s3")

DEFAULT_LIMIT = 50


class S3Connector(BaseConnector):
    """Ingesta de objetos S3/MinIO delegando la extracción en el conector `file`."""

    name: ClassVar[str] = "s3"
    description: ClassVar[str] = (
        "Ingesta de objetos S3/MinIO: s3://bucket/clave.pdf o s3://bucket/prefijo/ "
        "(requiere extra [s3])."
    )
    schemes: ClassVar[tuple[str, ...]] = ("s3",)
    priority: ClassVar[int] = 10  # gana a `file`, que reclama rutas *.pdf/docx/…

    def __init__(self, client: Any | None = None) -> None:
        self._client = client  # cliente boto3 o doble-inyectable (tests)

    # ------------------------------------------------------------------ API
    def fetch(self, source: str, **options: Any) -> list[Document]:
        bucket, key = self._parse(source)
        client = self._client or self._boto_client(options)
        prefix_mode = bool(options.get("prefix")) or not key or key.endswith("/")
        if prefix_mode:
            return self._fetch_prefix(client, bucket, key, options)
        return [self._fetch_object(client, bucket, key, options)]

    def can_handle(self, source: str) -> bool:
        return (source or "").strip().lower().startswith("s3://")

    # ------------------------------------------------------------- internals
    @staticmethod
    def _parse(source: str) -> tuple[str, str]:
        raw = (source or "").strip()
        if not raw.lower().startswith("s3://"):
            raise ConnectorError(f"Fuente S3 inválida: {source!r} (formato: s3://bucket/clave)")
        rest = raw[len("s3://") :]
        bucket, _, key = rest.partition("/")
        if not bucket:
            raise ConnectorError(f"Falta el bucket en la fuente S3: {source!r}")
        # La barra final se conserva: distingue prefijo (docs/) de objeto (docs/x.txt)
        return bucket, key.lstrip("/")

    @staticmethod
    def _boto_client(options: dict[str, Any]) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ConnectorError(
                "boto3 no está instalado. Instálalo con: pip install 'universal-rag[s3]'"
            ) from exc
        return boto3.client(
            "s3",
            endpoint_url=options.get("endpoint_url") or os.getenv("RAG_S3_ENDPOINT_URL"),
            region_name=options.get("region") or os.getenv("RAG_S3_REGION"),
            aws_access_key_id=options.get("aws_access_key_id"),
            aws_secret_access_key=options.get("aws_secret_access_key"),
        )

    def _fetch_prefix(
        self, client: Any, bucket: str, prefix: str, options: dict[str, Any]
    ) -> list[Document]:
        limit = max(1, int(options.get("limit", DEFAULT_LIMIT)))
        keys = self._list_keys(client, bucket, prefix, limit)
        if not keys:
            raise ConnectorError(f"Sin objetos soportados bajo s3://{bucket}/{prefix}")
        documents: list[Document] = []
        errors: list[str] = []
        for key in keys:
            try:
                documents.append(self._fetch_object(client, bucket, key, options))
            except Exception as exc:  # noqa: BLE001 — un objeto roto no para el lote
                logger.warning("Objeto S3 fallido %s: %s", key, exc)
                errors.append(f"{key}: {type(exc).__name__}: {exc}")
        if not documents:
            raise ConnectorError(
                f"Ningún objeto bajo s3://{bucket}/{prefix} pudo ingestarse: {errors or 'sin detalles'}"
            )
        return documents

    @staticmethod
    def _list_keys(client: Any, bucket: str, prefix: str, limit: int) -> list[str]:
        """Claves soportadas bajo un prefijo (paginado manual, sin depender del paginator)."""
        supported = FileConnector.extensions
        keys: list[str] = []
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        while len(keys) < limit:
            response = client.list_objects_v2(**kwargs)
            for item in response.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key or key.endswith("/"):
                    continue  # "directorio" sintético
                if Path(key).suffix.lower() not in supported:
                    continue
                keys.append(key)
                if len(keys) >= limit:
                    break
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if not token:
                break
            kwargs["ContinuationToken"] = token
        return keys

    def _fetch_object(
        self, client: Any, bucket: str, key: str, options: dict[str, Any]
    ) -> Document:
        suffix = Path(key).suffix.lower()
        if suffix not in FileConnector.extensions:
            raise ConnectorError(
                f"Extensión no soportada para s3://{bucket}/{key}: '{suffix or '(sin extensión)'}'"
            )
        try:
            response = client.get_object(Bucket=bucket, Key=key)
            body: bytes = response["Body"].read()
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError(f"No se pudo descargar s3://{bucket}/{key}: {exc}") from exc

        # La extracción (PDF/DOCX/…) corre local sobre un archivo temporal con
        # la extensión original; luego se reconstruye el documento con URI S3.
        file_connector = FileConnector()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(body)
            tmp_path = Path(handle.name)
        try:
            extracted = file_connector.fetch(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
        if not extracted:
            raise ConnectorError(f"Sin contenido extraíble en s3://{bucket}/{key}")

        source_doc = extracted[0]
        metadata = {
            key_: value
            for key_, value in source_doc.metadata.items()
            if key_ not in ("file_path", "file_size", "modified_at", "file_name")
        }
        metadata.update(
            {
                "bucket": bucket,
                "key": key,
                "etag": str(response.get("ETag", "")).strip('"'),
                "object_size": len(body),
            }
        )
        return self.build_document(
            content=source_doc.content,
            source=self.name,
            uri=f"s3://{bucket}/{key}",
            title=Path(key).stem,
            metadata=metadata,
        )
