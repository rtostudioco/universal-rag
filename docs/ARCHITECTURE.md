# Arquitectura de Universal RAG

Universal RAG sigue la misma filosofía que Universal MCP: **núcleo pequeño y estable, capacidades plugables por convención de carpetas, aislamiento de fallos**.

## 1. Vista general del pipeline

```
┌─────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────┐
│   fuente     │─▶│ ConnectorRegistry │──▶│ Document │──▶│ Chunker  │──▶│  Embedder    │──▶│ VectorStore│
│ (pdf/url/…) │   │ (auto-discovery)  │   │ (modelo) │   │ (troceo) │   │ (local)     │   │ (Chroma)  │
└─────────────┘   └──────────────────┘   └──────────┘   └──────────┘   └──────────────┘   └─────┬─────┘
                                                                                                 │
   pregunta ────────────────────────────────▶ Retriever ◀────────────────────────────────────────┘
                                                 │ RetrievedChunks (con scores)
                                                 ▼
                                        Generator (LLM plugable)
                                                 │
                                                 ▼
                                     Answer + Citations [1][2]…
```

Todo lo anterior está orquestado por `RAGPipeline` (`core/pipeline.py`), la única pieza que consumen la API, el CLI y el servidor MCP.

## 2. Modelo de datos (core/models.py)

| Modelo | Rol |
|---|---|
| `Document` | Unidad universal de conocimiento: `id`, `content`, `source`, `uri`, `title`, `metadata`, `created_at` (fecha de origen, p. ej. de un MP3 en Fase 2), `ingested_at`. |
| `Chunk` | Fragmento vectorizable: texto, posición (`index`/`total`), metadatos heredados del documento. |
| `RetrievedChunk` | Chunk + score de similitud (coseno; mayor = mejor). |
| `Citation` | Referencia verificable: `uri`, `source`, `title`, `quote`, `score`. |
| `Answer` | Respuesta final: texto, citas, modelo usado, latencia, nº de contextos. |
| `IngestResult` | Resumen de ingesta: documentos, chunks, omitidos, errores. |

**Invariante clave**: cualquier conector, sin importar el origen, produce `Document`. El resto del pipeline ignora por completo de dónde vino el contenido.

## 3. Conectores (core/connectors.py)

```python
class BaseConnector(ABC):
    name: ClassVar[str]           # "file", "url", "audio", …
    description: ClassVar[str]
    schemes: ClassVar[tuple[str, ...]]     # "http", "file", "drive", …
    extensions: ClassVar[tuple[str, ...]]  # ".pdf", ".mp3", …

    def can_handle(self, source: str) -> bool: ...
    @abstractmethod
    def fetch(self, source: str, **options) -> list[Document]: ...
```

### Auto-descubrimiento

- **Builtins**: `src/universal_rag/connectors/<nombre>/connector.py` (misma convención que los módulos de Universal MCP).
- **Externos**: `RAG_CONNECTOR_DIRS=/ruta` — cada `<dir>/<nombre>/connector.py` se importa dinámicamente.
- El `ConnectorRegistry` escanea, instancia y registra; **un conector roto se reporta y queda aislado** (`records[].error`), sin afectar al resto.
- Routing: `registry.resolve(source)` devuelve el primer conector cuyo `can_handle()` acepte la fuente (esquema URI o extensión).

### Semántica de re-ingesta

El `id` del documento es `sha256("doc|" + uri)`: ingerir la misma URI **reemplaza** la versión anterior (se borran sus chunks y se insertan los nuevos). Re-ingerir un PDF actualizado actualiza el índice, no lo duplica.

## 4. Chunker (core/chunker.py)

Estrategia determinista:

1. Texto → párrafos; los párrafos > `size` se parten en frases; las frases enormes en palabras (nunca a mitad de palabra salvo palabras patológicas > size).
2. Empaquetado greedy hasta `chunk_size`.
3. Al abrir fragmento nuevo se antepone la **cola** del anterior (~`chunk_overlap` chars, cortada en límite de palabra) para conservar contexto entre fragmentos.

## 5. Embeddings (core/embeddings.py)

| Implementación | Uso |
|---|---|
| `SentenceTransformerEmbedder` | Embeddings **locales y privados** (default en `auto` si está instalado). Modelo multilingüe con buen rendimiento en español. Carga perezosa: importar la librería nunca descarga modelos. |
| `HashEmbedder` | Determinista, cero dependencias (hashing trick con signo, L2 normalizado). Para tests, CI y modo offline. |

`create_embedder(settings)` resuelve: `hash` fijo → HashEmbedder; `auto` → sentence-transformers si está instalable, si no hash con warning; `sentence-transformers` → error claro si falta la lib (`pip install 'universal-rag[local]'`).

## 6. Almacenes vectoriales (core/store.py)

Contrato `BaseVectorStore`: `add`, `query(embedding, k, where)`, `delete_document`, `count_chunks`, `list_documents`, `clear`.

- **`ChromaVectorStore`** (default): ChromaDB embebido y persistente (`RAG_CHROMA_PATH`), colección con espacio **coseno**. Metadatos aplanados a primitivos (dicts/listas → JSON string).
- **`InMemoryVectorStore`**: coseno puro en RAM. Para tests y demos sin persistencia.

Ambos devuelven scores como **similitud** (Chroma entrega distancias; se convierte `1 − dist`).

## 7. Generación (core/generator.py)

Contrato: `generate(question, contexts) -> (texto, modelo)`; el pipeline convierte los marcadores `[n]` del texto en `Citation` estructuradas (si el LLM no citó nada, se cita el top-1 para trazabilidad).

- **`OpenAICompatGenerator`**: `POST {base_url}/chat/completions` — funciona con OpenAI, z-ai, Ollama (`/v1`), LM Studio, vLLM, OpenRouter. Prompt de sistema en español con regla de citar solo el contexto.
- **`EchoGenerator`**: extractivo, sin LLM — offline, determinista. Útil para probar el pipeline end-to-end sin API keys.

## 8. Interfaces

| Interfaz | Archivo | Notas |
|---|---|---|
| API REST | `api/app.py` (FastAPI) | `/health`, `/stats`, `/ingest`, `/ingest/text`, `/query`, `/documents`, `/documents/{id}` |
| CLI | `cli.py` (Typer) | `ingest`, `text`, `query`, `documents`, `delete`, `clear`, `stats`, `serve`, `mcp` |
| MCP | `mcp_server.py` (FastMCP, SDK oficial) | 6 tools `rag_*`; stdio o streamable-http |

Las tres comparten el mismo `RAGPipeline`; no hay lógica duplicada.

## 9. Aislamiento de fallos

| Nivel | Garantía |
|---|---|
| Carga de conector | Un `connector.py` con error no derriba el registro (queda en `records` con `error`). |
| Routing | `can_handle()` que lanza excepción simplemente no participa en el routing. |
| Ingesta | Fallo de un documento no aborta el lote: se acumula en `IngestResult.errors`. |
| Query sin contexto | Respuesta honesta "No encontré información relevante…" en lugar de alucinación. |

## 10. Decisiones de diseño

- **`Settings` con prefijo `RAG_`** (pydantic-settings): todo configurable sin tocar código; `.env` soportado.
- **IDs deterministas** (`sha256` truncado): re-ingesta idempotente, chunk IDs estables, deduplicación natural.
- **Embeddings por defecto locales**: privacidad total del corpus; el LLM es opcional (modo extractivo).
- **ChromaDB embebido**: cero infraestructura (sin servidor de vectores), persistencia en disco, API estable.
