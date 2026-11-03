"""Conector ``file``: archivos locales — TXT, MD, CSV, PDF y DOCX."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from universal_rag.core.connectors import BaseConnector
from universal_rag.core.models import Document
from universal_rag.core.utils import get_logger

logger = get_logger("connectors.file")

_MAX_CSV_ROWS = 5000


class FileConnector(BaseConnector):
    """Extrae texto de archivos locales y lo normaliza a Document(s).

    Soporta patrones glob (``docs/*.md``) para ingesta masiva.
    """

    name: ClassVar[str] = "file"
    description: ClassVar[str] = (
        "Ingesta de archivos locales: .txt, .md, .csv, .pdf y .docx (acepta globs como docs/*.pdf)."
    )
    schemes: ClassVar[tuple[str, ...]] = ("file",)
    extensions: ClassVar[tuple[str, ...]] = (".txt", ".md", ".markdown", ".csv", ".pdf", ".docx")

    def can_handle(self, source: str) -> bool:
        source = (source or "").strip()
        if source.startswith("file://"):
            return True
        if any(ch in source for ch in "*?[") and Path(source).parent.exists():
            return True
        return super().can_handle(source)

    def fetch(self, source: str, **options: Any) -> list[Document]:
        raw = source.strip()
        if raw.startswith("file://"):
            raw = raw[len("file://") :]
        path = Path(raw).expanduser()
        encoding = str(options.get("encoding", "utf-8"))
        title = options.get("title")
        extra_metadata = options.get("metadata") or {}

        if any(ch in str(path) for ch in "*?[") and path.parent.exists():
            matches = sorted(p for p in path.parent.glob(path.name) if p.is_file())
        elif path.exists():
            matches = [path if path.is_file() else None] if path.is_file() else self._dir_files(path)
        else:
            from universal_rag.core.connectors import ConnectorError

            raise ConnectorError(f"El archivo no existe: {path}")

        documents: list[Document] = []
        for file_path in matches:
            if file_path is None:
                continue
            if file_path.suffix.lower() not in self.extensions:
                logger.warning("Extensión no soportada, se omite: %s", file_path)
                continue
            documents.append(self._read_file(file_path, encoding, extra_metadata))
        if title and len(documents) == 1:
            documents[0].title = title
        return documents

    # ------------------------------------------------------------- extraction
    def _dir_files(self, directory: Path) -> list[Path]:
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in self.extensions)

    def _read_file(self, path: Path, encoding: str, extra_metadata: dict[str, Any]) -> Document:
        suffix = path.suffix.lower()
        stat = path.stat()
        metadata = {
            **extra_metadata,
            "file_path": str(path.resolve()),
            "file_name": path.name,
            "extension": suffix,
            "file_size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
        created_at: datetime | None = None

        if suffix in (".txt", ".md", ".markdown"):
            content = path.read_text(encoding=encoding, errors="replace")
            if suffix == ".txt":
                metadata["format"] = "plain-text"
            else:
                metadata["format"] = "markdown"
        elif suffix == ".csv":
            content = self._read_csv(path, encoding)
            metadata["format"] = "csv"
        elif suffix == ".pdf":
            content, page_count = self._read_pdf(path)
            metadata["format"] = "pdf"
            metadata["pages"] = page_count
        elif suffix == ".docx":
            content, paragraph_count = self._read_docx(path)
            metadata["format"] = "docx"
            metadata["paragraphs"] = paragraph_count
        else:  # pragma: no cover — filtrado por can_handle
            raise ValueError(f"Formato no soportado: {suffix}")

        return self.build_document(
            content=content,
            source=self.name,
            uri=path.resolve().as_uri(),
            title=path.stem,
            metadata=metadata,
            created_at=created_at,
        )

    @staticmethod
    def _read_csv(path: Path, encoding: str) -> str:
        """Convierte CSV a texto plano fila-a-fila (cabecera incluida)."""
        lines: list[str] = []
        with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            header: list[str] | None = None
            for row_index, row in enumerate(reader):
                if row_index >= _MAX_CSV_ROWS:
                    lines.append(f"… (truncado a {_MAX_CSV_ROWS} filas)")
                    break
                if not row:
                    continue
                if header is None:
                    header = row
                    lines.append(" | ".join(f"{cell.strip()}" for cell in row if cell.strip()))
                    continue
                cells = [
                    f"{(header[i] if i < len(header) else f'col{i}')}: {cell.strip()}"
                    for i, cell in enumerate(row)
                    if str(cell).strip()
                ]
                if cells:
                    lines.append(" | ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _read_pdf(path: Path) -> tuple[str, int]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pypdf no está instalado (pip install universal-rag)") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(page for page in pages if page.strip())
        return text, len(reader.pages)

    @staticmethod
    def _read_docx(path: Path) -> tuple[str, int]:
        try:
            import docx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("python-docx no está instalado (pip install universal-rag)") from exc
        document = docx.Document(str(path))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts), len(parts)
