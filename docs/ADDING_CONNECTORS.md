# Cómo agregar un conector (la guía de 5 minutos)

El objetivo de Universal RAG es que agregar una fuente nueva (Google Drive, Notion, S3, un MP4, una base de datos…) sea **soltar una carpeta**, sin tocar el núcleo.

## 1. La convención

```
mi_conectores/
└── drive/
    └── connector.py      ← suficiente
```

O dentro del repo: `src/universal_rag/connectors/drive/connector.py` (los builtins se autodescubren igual).

## 2. El código mínimo

```python
# mi_conectores/drive/connector.py
from __future__ import annotations

from typing import Any, ClassVar

from universal_rag.core.connectors import BaseConnector
from universal_rag.core.models import Document


class DriveConnector(BaseConnector):
    name: ClassVar[str] = "drive"
    description: ClassVar[str] = "Ingesta de documentos de Google Drive (exportados a texto)."
    schemes: ClassVar[tuple[str, ...]] = ("drive",)

    def can_handle(self, source: str) -> bool:
        return source.lower().startswith("drive://")

    def fetch(self, source: str, **options: Any) -> list[Document]:
        file_id = source.removeprefix("drive://")
        # ... tu lógica: API de Drive, descarga, extracción de texto ...
        text = "Contenido extraído del archivo de Drive"
        return [
            self.build_document(
                content=text,
                source=self.name,
                uri=source,
                title="Mi documento de Drive",
                metadata={"drive_file_id": file_id},
                # created_at=<datetime de creación del archivo>  ← útil para audio/video (Fase 2)
            )
        ]
```

Reglas:

1. Hereda de `BaseConnector` y define `name`, `description`, `schemes` (y/o `extensions`).
2. Implementa `can_handle()` y `fetch()`.
3. **Normaliza a `Document`** con el helper `build_document()` (normaliza espacios y calcula el ID estable por URI).
4. Una clase por módulo; el registro detecta cualquier subclase no abstracta de `BaseConnector` en el módulo.

## 3. Usarlo

```bash
# por directorio externo
export RAG_CONNECTOR_DIRS=/ruta/a/mi_conectores
universal-rag ingest "drive://abc123"

# o registro manual en Python
from universal_rag.core.connectors import ConnectorRegistry
from universal_rag.core.pipeline import RAGPipeline

registry = ConnectorRegistry.discover(["/ruta/a/mi_conectores"])
rag = RAGPipeline(registry=registry)
rag.ingest("drive://abc123")
```

La resolución es automática: `resolve("drive://abc123")` devuelve el primer conector cuyo `can_handle()` acepte la fuente. También puedes forzarlo: `rag.ingest("…", connector="drive")`.

## 4. Checklist de calidad

- [ ] `can_handle()` es **estricto** (no aceptes fuentes que no puedas procesar).
- [ ] `fetch()` lanza excepciones claras ante fallos (el pipeline las captura en `IngestResult.errors` y sigue).
- [ ] Metadatos útiles en el `Document` (rutas, IDs externos, fechas) — terminan en el índice y permiten filtros `where`.
- [ ] Sin efectos globales en el import (las dependencias pesadas se importan dentro de `fetch()`).
- [ ] Si la fuente es media (audio/video), rellena `created_at`: las citas por minuto de la Fase 2 lo usarán.

## 5. Ejemplo completo en el repo

Mira `examples/external_connector/` — un conector `memo://` de juguete que se carga desde un directorio externo y se testea en la suite.
