# Changelog de Universal RAG

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y versionado con [SemVer](https://semver.org/lang/es/).

## [0.7.0] — Fase 7: Conectores empresariales, memoria conversacional y sincronización

### Añadido
- **Conector `confluence`** — espacios (`confluence://space/KEY`), páginas
  (`confluence://page/ID`) y búsquedas **CQL** (`confluence://cql/…`) vía REST
  v1; URLs de `atlassian.net` y `/display/` detectadas automáticamente. Auth
  **Basic** (email + token, Cloud) o **Bearer** (PAT, Server/DC). El formato de
  almacenamiento XHTML se convierte a texto (encabezados, listas, tablas y
  macros de código).
- **Conector `sharepoint`** — Microsoft Graph con **client credentials**
  (Entra ID, `Sites.Read.All`): sitio → librería → listado paginado →
  descarga → extracción reutilizando las rutinas del conector `file`
  (TXT/MD/CSV/PDF/DOCX). Filtro de extensiones, carpetas opcionales
  (`folder=`) y token cacheado.
- **Memoria conversacional multi-turno** (`ConversationMemory`): doble LRU
  thread-safe (turnos por conversación + conversaciones vivas).
  `query(…, conversation_id=…)` en Python, `conversation_id` en `/query` y
  `/query/stream`, `--conversation` en CLI y **`chat` multi-turno por
  defecto**; el historial se inyecta al LLM como mensajes `user`/`assistant`
  (firma clásica de generadores de terceros intacta). Las consultas
  conversacionales **omiten la caché semántica**. Endpoints
  `GET/DELETE /conversations` (+ historial por ID).
- **Sincronización programada de fuentes** (`SyncScheduler`): `POST /sync`
  (o `sync add` en CLI) registra fuentes con intervalo; re-ingesta en un hilo
  de fondo opt-in (`RAG_SYNC_ENABLED=1`) o bajo demanda
  (`POST /sync/{id}/run`, `sync run`). Estado persistente opcional a JSON
  (`RAG_SYNC_STATE_PATH`) compartible entre CLI y API. La ingesta incremental
  (F4) hace que re-ingerir sin cambios sea gratuito.
- CLI: `sync add/list/run/remove`, `conversations [--show|--clear]`,
  `ingest -o clave=valor` (opciones de conector repetibles).

### Cambiado
- Compatibilidad total hacia atrás: los generadores custom con la firma
  clásica siguen funcionando.

### Tests y docs
- **87 tests nuevos (393 total)** con dobles de Confluence/Microsoft Graph
  vía `httpx.MockTransport`.
- Docs actualizadas: README (features, CLI, API, config), `.env.example`,
  `docs/ARCHITECTURE.md`.

## [0.6.0] — Fase 6: Streaming, caché semántica y workbench web

### Añadido
- **Streaming SSE** (`POST /query/stream`): eventos `meta` (modelo, modo de
  búsqueda, nº de contextos y **preview de fuentes**) → `token`* (texto
  progresivo) → `done` (Answer completo con citas). Funciona con o sin LLM:
  `EchoGenerator` trocea su salida y `OpenAICompatGenerator` hace **streaming
  real** del endpoint (`stream=true`, parseo SSE tolerante, fallback automático
  si el endpoint ignora `stream`). Cabeceras anti-buffering (`X-Accel-Buffering`).
- **`RAGPipeline.stream_query()`**: mismo contrato de eventos a nivel Python;
  en CLI con `universal-rag query "…" --stream` (imprime token a token y
  cierra con la tabla de fuentes).
- **Caché semántica de consultas** (`core/cache.py`, Fase 6): guarda respuestas
  indexadas por el embedding de la pregunta; ante una pregunta igual o
  parafraseada (coseno ≥ `RAG_CACHE_THRESHOLD`, default `0.93`) sirve la
  respuesta sin repetir recuperación ni LLM. **LRU** (`RAG_CACHE_MAX_ENTRIES`)
  + **TTL opcional** (`RAG_CACHE_TTL`), thread-safe, por colección, con
  métricas (`entries/hits/misses/hit_rate` en `/stats`). **Invalidación
  automática** al ingestar, borrar documento o vaciar. `Answer.cached=true`
  y `cache.enabled=false` con `RAG_CACHE_ENABLED=0`. La caché solo aplica a
  consultas con parámetros por defecto (`k`/`where` explícitos la evitan).
- **Workbench web** en `/` (Fase 6): consola completa autocontenida (sin
  CDNs) con 4 pestañas — **Chat** (streaming token a token vía `fetch` +
  ReadableStream, badges de caché/modelo/búsqueda y citas al final),
  **Documentos** (tabla con título/fuente/URI/fragmentos + eliminar),
  **Ingestar** (texto directo o fuente con conector automático) y **Estado**
  (stats, caché, colecciones con vaciado, conectores y métricas runtime).
  Selector de colección y API key compartidos entre pestañas.
- 41 tests nuevos (caché, streaming, SSE mock del LLM, workbench) —
  suite total: **306**.

### Cambiado
- `GET /` ahora sirve el **workbench**; el mini chat clásico de la Fase 3
  sigue disponible en `GET /chat`.
- `Metrics.snapshot()` expone el contador `queries_stream`.
- `OpenAICompatGenerator` acepta `client=` (httpx) inyectable para tests y
  reutiliza payload/headers entre `generate()` y `stream_generate()`.
- `RAGPipeline.stats()` incluye la clave `cache`.

## [0.5.0] — Fase 5: Ecosistema de fuentes y calidad

### Añadido
- **Conector `s3`** (Fase 5): objetos S3/MinIO/R2 (`s3://bucket/clave.pdf`) y
  **prefijos completos** (`s3://bucket/informes/` → ingesta en lote con `limit`).
  La extracción (PDF/DOCX/TXT/MD/CSV) corre local sobre el objeto descargado;
  el documento conserva `bucket`, `key`, `etag` y `object_size`. Requiere el
  extra ``[s3]`` (boto3); credenciales por cadena estándar de AWS o opciones
  `aws_access_key_id`/`aws_secret_access_key`; MinIO/R2/Spaces con
  `endpoint_url` (o `RAG_S3_ENDPOINT_URL`). Un objeto roto no detiene el lote.
- **Conector `notion`** (Fase 5): páginas, **bases de datos completas** (cada
  fila = un documento) y búsqueda (`notion://search/…`) vía **API REST oficial**
  — sin dependencias nuevas (httpx). Token por `RAG_NOTION_TOKEN` u opción
  `token=`. Renderiza encabezados, listas, to-dos, código, tablas (→ Markdown)
  y respeta paginación y throttling (~3 req/s, ajustable con `throttle=0`).
  También acepta URLs públicas `https://www.notion.so/…` (prioridad sobre el
  conector `url`).
- **Reranking con LLM** (`RAG_RERANKER=llm`): un endpoint compatible OpenAI
  puntúa la relevancia 0-10 de cada candidato y reordena el top. Modelo
  dedicado opcional (`RAG_RERANK_LLM_MODEL`, default `RAG_LLM_MODEL`).
  **Degradación elegante**: fallo de red, JSON inválido o cobertura parcial
  (<80%) conserva el orden del retriever.
- **Extracción de tablas a Markdown**: las tablas de **DOCX** (python-docx) y
  **HTML/URL** (bs4, sin duplicar el texto del artículo) se anexan como bloques
  ``### Tabla N`` indexables; en **PDF** vía ``pdfplumber`` (extra opcional
  ``[tables]``, ausencia tolerada). Los metadatos exponen `tables: N` y una
  consulta sobre celdas recupera la tabla completa.
- Extras opcionales `[s3]` (boto3) y `[tables]` (pdfplumber).
- 63 tests nuevos (tablas, Notion, S3, reranker LLM) — suite total: **265**.

### Cambiado
- `Settings.reranker` acepta `llm` (antes: `none` | `cross-encoder`).
- `create_reranker()` con `RAG_RERANKER=llm` sin `RAG_LLM_BASE_URL` cae a
  `NoopReranker` con advertencia (no rompe el arranque).
- El conector `url` extrae `<table>` a Markdown y pasa `include_tables=False`
  a trafilatura para no duplicar contenido.

## [0.4.0] — Fase 4: Producción y escalado

### Añadido
- **Colecciones múltiples** (`CollectionManager`): bases de conocimiento
  aisladas por nombre en API, CLI (`--collection`), MCP y Python. La colección
  `default` mantiene la configuración original (retro-compatibilidad con
  índices de Fases 1–3); las demás aíslan su persistencia en
  `<chroma_path>/<nombre>`. Endpoints `GET /collections` y
  `DELETE /collections/{name}`; comando CLI `collections`; tool MCP
  `rag_list_collections`.
- **Jobs de ingesta asíncrona** (`JobManager`): `POST /ingest/async` encola la
  ingesta en un pool de hilos (`RAG_JOBS_WORKERS`) y devuelve `job_id` para
  consultar en `GET /jobs/{id}`. Ideal para audio/video/lotes largos.
- **Autenticación opcional por API key**: `RAG_API_KEY` activa la exigencia de
  `X-API-Key` o `Authorization: Bearer` en toda la API (excepto `/` y
  `/health`), con comparación en tiempo constante. La mini chat UI pregunta la
  clave al recibir un 401.
- **Métricas de runtime** (`Metrics`): contadores thread-safe de requests,
  queries, ingestas, errores y uptime; middleware HTTP y bloque `runtime` en
  `GET /stats`, más un resumen de colecciones.
- **Ingesta incremental**: huella de contenido (`content_hash`, insensible a
  espacios) en los metadatos; re-ingerir lo idéntico devuelve `unchanged` sin
  re-vectorizar. `force=True` fuerza la re-indexación.
- **Docker**: `Dockerfile` (slim, healthcheck, volumen `/app/data`, extras
  `[local]`/`[media]` opcionales), `docker-compose.yml` y
  `docs/DEPLOYMENT.md`; job de CI que construye la imagen y verifica
  `/health` en el contenedor.
- Mini chat UI: selector de colecciones y soporte de API key.
- 40+ tests nuevos (colecciones, jobs, auth, métricas, incremental, API y
  MCP) — suite total: 202.

### Cambiado
- `RAGPipeline.ingest()`/`ingest_text()` aceptan `force=False` (keyword-only).
- `BaseVectorStore` incorpora `document_hash(document_id)`.
- `create_mcp_server(manager)` recibe un `CollectionManager` (compatible con
  `pipeline=` pre-construido).

## [0.3.0] — Fase 3: Búsqueda híbrida, reranking y evaluación

### Añadido
- **Búsqueda híbrida** (`RAG_RETRIEVAL=hybrid`): BM25 propio sin dependencias
  (`core/bm25.py`) fusionado con el ranking vectorial vía **RRF** o suma
  ponderada min-max (`core/hybrid.py`), con `candidate_k` candidatos.
- **Reranking** (`RAG_RERANKER=cross-encoder`): cross-encoder local
  multilingüe (`bge-reranker-base`) con carga perezosa y score sigmoide.
- **Evaluación de calidad** (`core/evaluation.py`): golden sets JSONL con
  precision/recall/MRR/hit-rate; comando `universal-rag eval` (tabla o
  `--json`) y ejemplo en `examples/golden.jsonl`.
- **Mini chat UI**: chat web autocontenido en `/` (sin CDNs) con citas,
  minutos y badges del modo de búsqueda.
- `Retriever` (`core/retriever.py`) unifica las tres etapas de recuperación.

## [0.2.0] — Fase 2: Audio, video, YouTube y Drive

### Añadido
- Conectores **`audio`** (MP3/WAV/M4A/OGG/FLAC…), **`video`** (MP4/MKV/MOV…),
  **`youtube`** (yt-dlp) y **`drive`** (gdown) con **ASR local
  faster-whisper** — todo corre en tu máquina.
- **Citas por minuto**: chunker temporal acotado por segmentos
  (`metadata["segments"]`) y citas `[mm:ss–mm:ss]` en `Citation.timestamp`.
- `core/transcript.py`: `Transcriber` (config global `RAG_ASR_*` u opciones
  por ingesta) y decodificación robusta de audio con fallback propio sobre
  PyAV.
- Extras opcionales `[media]` (faster-whisper, yt-dlp).

## [0.1.0] — Fase 1: MVP

### Añadido
- Núcleo del pipeline: `Document → Chunker → Embedder → VectorStore →
  Retriever → Generator → Answer + Citations [n]`.
- Conectores `text`, `file` (PDF/DOCX/TXT/MD/CSV + globs) y `url` (extracción
  legible con trafilatura/bs4), con auto-descubrimiento por convención de
  carpetas y conectores externos vía `RAG_CONNECTOR_DIRS`.
- Embeddings locales (`sentence-transformers` con fallback hash determinista)
  y ChromaDB embebido persistente (+ store en memoria para tests).
- Generación plugable: endpoints compatibles OpenAI o modo extractivo sin LLM.
- Tres interfaces: API REST (FastAPI), CLI (Typer + Rich) y servidor MCP
  (FastMCP).
- Respuestas con citas verificables y semántica de re-ingesta por URI.
