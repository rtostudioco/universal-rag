# Arquitectura de Universal RAG

Universal RAG sigue la misma filosofía que Universal MCP: **núcleo pequeño y estable, capacidades plugables por convención de carpetas, aislamiento de fallos**.

## 1. Vista general del pipeline

```
┌─────────────┐   ┌──────────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐
│   fuente     │─▶│ ConnectorRegistry │──▶│ Document │──▶│ Chunker  │──▶│ Embedder │──▶│VectorStore │
│ (pdf/url/…) │   │ (auto-discovery)  │   │ (modelo) │   │ (troceo) │   │ (local)  │   │  (Chroma)  │
└─────────────┘   └──────────────────┘   └──────────┘   └──────────┘   └──────────┘   └─────┬──────┘
                                                                                              │
   pregunta ──▶ Retriever ◀───────────────────────────────────────────────────────────────────┘
                   │  vectorial (coseno)  ──o── híbrido: + BM25, fusión RRF/weighted (F3)
                   │ RetrievedChunks (candidate_k)
                   ▼
                Reranker (F3, opcional: cross-encoder local)
                   │ top-k reordenado
                   ▼
                Generator (LLM plugable)
                   │
                   ▼
            Answer + Citations [1][2]… [mm:ss–mm:ss]
```

Todo lo anterior está orquestado por `RAGPipeline` (`core/pipeline.py`), la única pieza que consumen la API, el CLI y el servidor MCP.

## 2. Modelo de datos (core/models.py)

| Modelo | Rol |
|---|---|
| `Document` | Unidad universal de conocimiento: `id`, `content`, `source`, `uri`, `title`, `metadata`, `created_at` (fecha de origen, p. ej. de un MP3 en Fase 2), `ingested_at`. |
| `Chunk` | Fragmento vectorizable: texto, posición (`index`/`total`), metadatos heredados del documento. |
| `RetrievedChunk` | Chunk + score de similitud (coseno; mayor = mejor). En modo híbrido, score de fusión (RRF o ponderada). |
| `Citation` | Referencia verificable: `uri`, `source`, `title`, `quote`, `score` — y en media, `start_time`/`end_time`/`timestamp` `[mm:ss–mm:ss]`. |
| `Answer` | Respuesta final: texto, citas, modelo usado, latencia, nº de contextos y `retrieval` (modo de búsqueda activo). |
| `IngestResult` | Resumen de ingesta: documentos, chunks, omitidos, **sin cambios** (`unchanged`, Fase 4), errores. |

**Invariante clave**: cualquier conector, sin importar el origen, produce `Document`. El resto del pipeline ignora por completo de dónde vino el contenido.

## 3. Conectores (core/connectors.py)

```python
class BaseConnector(ABC):
    name: ClassVar[str]           # "file", "url", "audio", …
    description: ClassVar[str]
    schemes: ClassVar[tuple[str, ...]]     # "http", "file", "drive", …
    extensions: ClassVar[tuple[str, ...]]  # ".pdf", ".mp3", …
    priority: ClassVar[int] = 0            # mayor = se consulta antes en resolve()

    def can_handle(self, source: str) -> bool: ...
    @abstractmethod
    def fetch(self, source: str, **options) -> list[Document]: ...
```

### Catálogo

| Conector | Fase | Origen | Notas |
|---|---|---|---|
| `text` | 1 | texto inline (`text://`) | usado también por `pipeline.ingest_text()` |
| `file` | 1 | PDF, DOCX, TXT, MD, CSV + globs | extracción con pypdf/python-docx |
| `url` | 1 | páginas web | trafilatura sobre HTML limpiado + fallback bs4 |
| `audio` | 2 | MP3/WAV/M4A/OGG/FLAC… | ASR local faster-whisper + minutos |
| `video` | 2 | MP4/MKV/MOV/WEBM… | decodifica pista de audio (PyAV) + ASR |
| `youtube` | 2 | URLs de YouTube | yt-dlp (solo audio) → ASR; `priority=20` |
| `drive` | 2 | Google Drive público/compartido | gdown → ruta según formato; `priority=15` |

La **prioridad** resuelve colisiones: para `https://youtube.com/watch…`, `youtube` (20) se evalúa antes que `url` (0); para enlaces de Drive, `drive` (15) antes que `url`.

### Auto-descubrimiento

- **Builtins**: `src/universal_rag/connectors/<nombre>/connector.py` (misma convención que los módulos de Universal MCP).
- **Externos**: `RAG_CONNECTOR_DIRS=/ruta` — cada `<dir>/<nombre>/connector.py` se importa dinámicamente.
- El `ConnectorRegistry` escanea, instancia y registra; **un conector roto se reporta y queda aislado** (`records[].error`), sin afectar al resto.
- Routing: `registry.resolve(source)` devuelve el primer conector cuyo `can_handle()` acepte la fuente (esquema URI o extensión).

### Semántica de re-ingesta e incremental (Fase 4)

El `id` del documento es `sha256("doc|" + uri)`: ingerir la misma URI **reemplaza** la versión anterior (se borran sus chunks y se insertan los nuevos). Re-ingerir un PDF actualizado actualiza el índice, no lo duplica.

Además, cada documento registra una **huella de contenido** (`content_hash` en los metadatos de sus chunks): si se re-ingesta el mismo URI con contenido idéntico, el pipeline lo salta (`unchanged`) sin re-vectorizar — ingesta incremental apta para cron. `force=True` la ignora.

## 4. Chunker (core/chunker.py)

Estrategia determinista con **dos rutas**:

**Documentos de texto** (Fase 1):
1. Texto → párrafos; los párrafos > `size` se parten en frases; las frases enormes en palabras (nunca a mitad de palabra salvo palabras patológicas > size).
2. Empaquetado greedy hasta `chunk_size`.
3. Al abrir fragmento nuevo se antepone la **cola** del anterior (~`chunk_overlap` chars, cortada en límite de palabra) para conservar contexto entre fragmentos.

**Transcripciones** (Fase 2 — audio/video): si el `Document` trae `metadata["segments"]` (`start`, `end`, `text`), se agrupan segmentos respetando el presupuesto de caracteres pero con cortes solo en fronteras de segmento; cada chunk lleva `start_time`/`end_time`. Las citas de esos chunks muestran `[mm:ss–mm:ss]` (citas por minuto).

## 4b. Transcripciones locales (core/transcript.py, Fase 2)

- `Transcriber` envuelve **faster-whisper** (CTranslate2, CPU int8 por defecto): modelo perezoso, configuración `RAG_ASR_MODEL/LANGUAGE/…` u opciones por ingesta (`asr_model`, `language`).
- Decodificación robusta: intenta `decode_audio` de faster-whisper y, si la versión de PyAV instalada es incompatible, cae a un decodificador propio (PyAV → float32 mono 16 kHz) y le pasa el array a Whisper — mismo resultado, independiente del pin de PyAV.
- Los segmentos viajan en `Document.metadata["segments"]`; `build_transcript_document()` (connectors/_media.py) es el helper común de audio/video/youtube/drive.

## 5. Embeddings (core/embeddings.py)

| Implementación | Uso |
|---|---|
| `SentenceTransformerEmbedder` | Embeddings **locales y privados** (default en `auto` si está instalado). Modelo multilingüe con buen rendimiento en español. Carga perezosa: importar la librería nunca descarga modelos. |
| `HashEmbedder` | Determinista, cero dependencias (hashing trick con signo, L2 normalizado). Para tests, CI y modo offline. |

`create_embedder(settings)` resuelve: `hash` fijo → HashEmbedder; `auto` → sentence-transformers si está instalable, si no hash con warning; `sentence-transformers` → error claro si falta la lib (`pip install 'universal-rag[local]'`).

## 6. Almacenes vectoriales (core/store.py)

Contrato `BaseVectorStore`: `add`, `query(embedding, k, where)`, `all_chunks(where)` (Fase 3: alimenta el índice BM25), `delete_document`, `count_chunks`, `list_documents`, `clear`.

- **`ChromaVectorStore`** (default): ChromaDB embebido y persistente (`RAG_CHROMA_PATH`), colección con espacio **coseno**. Metadatos aplanados a primitivos (dicts/listas → JSON string).
- **`InMemoryVectorStore`**: coseno puro en RAM. Para tests y demos sin persistencia.

Ambos devuelven scores como **similitud** (Chroma entrega distancias; se convierte `1 − dist`).

## 6b. Recuperación híbrida y reranking (Fase 3)

El `Retriever` (`core/retriever.py`) encadena tres etapas:

1. **Vectorial** (Fase 1): embeddings de la consulta vs `store.query()`.
2. **Híbrida** (opcional, `RAG_RETRIEVAL=hybrid`): un **BM25Okapi propio y sin dependencias** (`core/bm25.py`) se construye en memoria sobre `store.all_chunks()` (cacheado; se invalida tras cada ingesta/borrado/vaciado) y su ranking se fusiona con el vectorial (`core/hybrid.py`) vía **RRF** (`score = Σ 1/(k+rank)`, robusta a escalas distintas) o **suma ponderada** con min-max (`RAG_HYBRID_ALPHA` controla el peso vectorial). Brillante con códigos, nombres propios y términos exactos que los embeddings diluyen.
3. **Reranking** (opcional, `RAG_RERANKER=cross-encoder`): se recuperan `candidate_k` candidatos y un cross-encoder (`core/rerank.py`, `bge-reranker-base` multilingüe por defecto) los reordena con máxima precisión; el score pasa por sigmoide para mantener `[0, 1]` y monotonía. Carga perezosa: si falta `sentence-transformers`, solo falla al reranquear, con mensaje de instalación.

Con `RAG_RERANKER=none` y `RAG_RETRIEVAL=vector` (defaults) el comportamiento es idéntico a Fase 1/2 — cero regresiones.

## 6c. Evaluación de calidad (core/evaluation.py, Fase 3)

- Golden sets en JSONL: `{"query": …, "relevant": [ids/URIs/títulos]}` (match insensible a mayúsculas).
- Métricas sobre el top-k: **precision** (hits/recuperados), **recall** (hits/relevantes), **MRR** (rank del primer acierto) y **hit_rate**.
- Interfaces: función `evaluate()` (Python), comando `universal-rag eval golden.jsonl` (tabla o `--json`), ejemplo en `examples/golden.jsonl`.

## 7. Generación (core/generator.py)

Contrato: `generate(question, contexts) -> (texto, modelo)`; el pipeline convierte los marcadores `[n]` del texto en `Citation` estructuradas (si el LLM no citó nada, se cita el top-1 para trazabilidad).

- **`OpenAICompatGenerator`**: `POST {base_url}/chat/completions` — funciona con OpenAI, z-ai, Ollama (`/v1`), LM Studio, vLLM, OpenRouter. Prompt de sistema en español con regla de citar solo el contexto.
- **`EchoGenerator`**: extractivo, sin LLM — offline, determinista. Útil para probar el pipeline end-to-end sin API keys.

## 8. Interfaces

| Interfaz | Archivo | Notas |
|---|---|---|
| API REST | `api/app.py` (FastAPI) | `/` (workbench web, F6), `/chat` (mini chat, F3), `/health`, `/stats` (+ métricas y colecciones), `/ingest`, `/ingest/text`, `/ingest/async` (F4), `/jobs`, `/jobs/{id}` (F4), `/query`, `/query/stream` (SSE, F6), `/documents`, `/documents/{id}`, `/collections` (F4) |
| CLI | `cli.py` (Typer) | `ingest`, `text`, `query`, `chat` (F3), `eval` (F3), `collections` (F4), `documents`, `delete`, `clear`, `stats`, `serve`, `mcp` — todos aceptan `--collection` |
| MCP | `mcp_server.py` (FastMCP, SDK oficial) | 7 tools `rag_*` (incl. `rag_list_collections`, F4); stdio o streamable-http |

Las tres comparten el mismo `RAGPipeline` (o un `CollectionManager`); no hay lógica duplicada.

## 8b. Producción y escalado (Fase 4)

| Pieza | Archivo | Qué hace |
|---|---|---|
| `CollectionManager` | `core/collections.py` | Pipelines aislados por nombre (lazy, thread-safe). Comparte embedder/generator/registro entre colecciones. `default` mantiene la config original (retro-compat); las demás aíslan su persistencia en `<chroma_path>/<nombre>`. Nombres validados por regex (anti path-traversal). |
| `JobManager` / `JobInfo` | `core/jobs.py` | Ingesta asíncrona: pool de hilos (`RAG_JOBS_WORKERS`), estados `pending → running → done/error`, historial acotado en memoria (`RAG_JOBS_MAX_HISTORY`). Pensado para fuentes lentas (audio/video). |
| `Metrics` | `core/metrics.py` | Contadores thread-safe por proceso (requests, queries, ingests, errors, uptime). Middleware HTTP en la API + eventos de endpoints. Expuesto en `GET /stats` bajo `runtime`. |
| Auth por API key | `api/auth.py` | Si `RAG_API_KEY` está definida, dependency de FastAPI compara la clave (`X-API-Key` o `Bearer`) con `hmac.compare_digest`; `/` y `/health` abiertos. |
| Ingesta incremental | `core/pipeline.py` + `store.document_hash()` | `content_hash` (whitespace-insensible) en metadatos de chunks; salto de documentos idénticos. |
| Docker | `Dockerfile`, `docker-compose.yml` | Imagen slim con healthcheck, volumen `/app/data`; extras `[local]`/`[media]` opcionales. Guía en `docs/DEPLOYMENT.md`. |

## 8c. Streaming y caché semántica (Fase 6)

| Pieza | Archivo | Qué hace |
|---|---|---|
| Streaming SSE | `pipeline.stream_query()` + `api/app.py` (`POST /query/stream`) | Eventos `meta` (modelo, retrieval, contextos, preview de fuentes) → `token`* → `done` (Answer completo con citas). `EchoGenerator` trocea su salida (`tokenize_stream`, reconstrucción exacta); `OpenAICompatGenerator` consume el SSE del LLM (`stream=true`) con fallback si el endpoint no soporta streaming. |
| Caché semántica | `core/cache.py` (`SemanticCache`) | Respuestas indexadas por embedding de la pregunta; acierto si coseno ≥ `RAG_CACHE_THRESHOLD` (0.93). LRU (`RAG_CACHE_MAX_ENTRIES`) + TTL opcional (`RAG_CACHE_TTL`), thread-safe, por colección. Se invalida al ingestar/borrar/vaciar. Solo aplica con parámetros por defecto (`k`/`where` la evitan). Métricas en `stats()["cache"]`. |
| Workbench | `api/static/workbench.html` | SPA autocontenida en `/` (sin CDNs): Chat en streaming (fetch + ReadableStream), Documentos (listar/eliminar), Ingesta (texto/fuente) y Estado (stats, caché, colecciones, conectores, runtime). El mini chat F3 queda en `/chat`. |

## 9. Aislamiento de fallos

| Nivel | Garantía |
|---|---|
| Carga de conector | Un `connector.py` con error no derriba el registro (queda en `records` con `error`). |
| Routing | `can_handle()` que lanza excepción simplemente no participa en el routing. |
| Ingesta | Fallo de un documento no aborta el lote: se acumula en `IngestResult.errors`. |
| Colección | Un nombre inválido se rechaza con 422; las colecciones aíslan stores y fallos. |
| Jobs | El error de un job no derriba el pool; queda registrado en `JobInfo.error`. |
| Query sin contexto | Respuesta honesta "No encontré información relevante…" en lugar de alucinación. |

## 10. Decisiones de diseño

- **`Settings` con prefijo `RAG_`** (pydantic-settings): todo configurable sin tocar código; `.env` soportado.
- **IDs deterministas** (`sha256` truncado): re-ingesta idempotente, chunk IDs estables, deduplicación natural.
- **Embeddings por defecto locales**: privacidad total del corpus; el LLM es opcional (modo extractivo).
- **ChromaDB embebido**: cero infraestructura (sin servidor de vectores), persistencia en disco, API estable.
- **BM25 propio (Fase 3)**: ~80 líneas sin dependencias — el corpus de un RAG típico se reindexa en milisegundos, y evitamos sumar dependencias nativas.
- **Híbrido desactivado por defecto**: `RAG_RETRIEVAL=vector` preserva el comportamiento de versiones anteriores; el híbrido es opt-in pero recomendado.
- **Colecciones compartiendo embedder (Fase 4)**: cargar N veces el mismo modelo de embeddings desperdicia RAM; el manager crea el embedder una vez y lo reutiliza.
- **Jobs en memoria (Fase 4)**: suficiente para un proceso uvicorn y cero infraestructura; si se necesita multi-nodo, la interfaz `JobManager` es el punto de extensión natural (Redis/Postgres).
- **Auth opt-in (Fase 4)**: sin `RAG_API_KEY` el comportamiento es idéntico a v0.3.0; con clave, comparación en tiempo constante y rutas de salud abiertas para orquestadores.
