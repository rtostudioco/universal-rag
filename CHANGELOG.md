# Changelog de Universal RAG

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y versionado con [SemVer](https://semver.org/lang/es/).

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
