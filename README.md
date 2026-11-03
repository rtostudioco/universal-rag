# Universal RAG

> 🧩 **Pipeline plugable de Retrieval-Augmented Generation** — ingesta universal (texto, archivos, URLs, **audio, video, YouTube, Google Drive**), búsqueda **vectorial o híbrida (BM25 + vectores)** con reranking opcional, evaluación de calidad, **colecciones múltiples, jobs asíncronos, auth por API key** y respuestas con **citas verificables, incluidas citas por minuto** en audio/video.

Hermano de [Universal MCP](https://github.com/rtostudioco/universal-mcp): misma filosofía **"universal"** — núcleo estable, capacidades que se agregan soltando carpetas, sin tocar el servidor.

[![CI](https://github.com/rtostudioco/universal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/rtostudioco/universal-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)

---

## ✨ Características

| Pieza | Qué hace |
|---|---|
| 🔌 **Conectores plugables** | Fase 1: `text`, `file` (PDF/DOCX/TXT/MD/CSV + globs), `url` (extracción legible). Fase 2: `audio`, `video`, `youtube`, `drive`. Agrega los tuyos soltando una carpeta — un conector roto **no derriba el sistema**. |
| ✂️ **Chunker configurable** | División con conciencia de párrafos/frases/palabras y solapamiento; para audio/video, **troceado temporal** acotado por minuto. |
| 🧠 **Embeddings locales** | `sentence-transformers` (multilingüe, privado, sin APIs) con fallback determinista sin dependencias. |
| 🗄️ **ChromaDB embebido** | Persistente, sin servidor; intercambiable por otro store (incl. uno en memoria para tests). |
| 🔍 **Búsqueda semántica** | Recuperación con scores de similitud y filtros por metadatos (`where`). |
| ⚡ **Búsqueda híbrida (Fase 3)** | BM25 (léxico, cero dependencias) + vectores con **fusión RRF o ponderada** — atrapa códigos, nombres propios y términos exactos que los embeddings diluyen. |
| 🎯 **Reranking (Fase 3)** | Reranker plugable: cross-encoder local (`bge-reranker`, multilingüe) reordena los candidatos con máxima precisión. |
| 📊 **Evaluación (Fase 3)** | Golden sets JSONL con **precision/recall/MRR/hit-rate** — mide la calidad antes de tocar producción. |
| 🗨️ **Mini chat UI (Fase 3)** | Chat web incluido (`http://localhost:8100/`), autocontenido y sin CDNs, con citas y minutos visibles. |
| 📚 **Colecciones múltiples (Fase 4)** | Bases de conocimiento aisladas (`legal`, `faq`, `energy`…) sobre un mismo despliegue: ingesta/consulta por colección en API, CLI y MCP. |
| 🔄 **Jobs asíncronos (Fase 4)** | `POST /ingest/async` + polling en `GET /jobs/{id}` — ingesta de fuentes lentas (audio, video, lotes) sin bloquear al cliente. |
| 🔐 **Auth por API key (Fase 4)** | Protege la API con `RAG_API_KEY` (`X-API-Key` o `Bearer`); `/` y `/health` siguen abiertos para monitoreo. |
| 📈 **Métricas runtime (Fase 4)** | Contadores de peticiones/consultas/ingestas/errores y uptime en `GET /stats`, sin dependencias extra. |
| 🪶 **Ingesta incremental (Fase 4)** | Huella de contenido (`content_hash`): re-ingerir lo mismo es un no-op (`unchanged`); solo se re-vectoriza lo que cambió. |
| 🐳 **Docker incluido (Fase 4)** | `Dockerfile` + `docker-compose.yml` con healthcheck y volumen de datos — despliegue en un comando. |
| 💬 **Generación plugable** | Cualquier endpoint **compatible OpenAI** (OpenAI, z-ai, Ollama, LM Studio, vLLM…) o modo extractivo sin LLM. |
| 📎 **Citas verificables** | Cada respuesta referencia sus fuentes con `uri`, score y cita textual. |
| 🎙️ **ASR local (Fase 2)** | faster-whisper corre en tu máquina: MP3/WAV/M4A/MP4/MKV… → transcripción con **citas `[mm:ss–mm:ss]`**. |
| 🖥️ **3 interfaces** | **API REST** (FastAPI) + **CLI** (Typer) + **servidor MCP** (para Claude Desktop / Universal MCP). |

## 🚀 Inicio rápido

```bash
pip install universal-rag                 # núcleo (embeddings deterministas)
pip install "universal-rag[local]"        # + embeddings neuronales locales
pip install "universal-rag[media]"        # + audio/video/youtube (faster-whisper, yt-dlp)
```

### CLI

```bash
# Ingesta (el conector se resuelve automáticamente por esquema/extensión/dominio)
universal-rag ingest ./docs/manual.pdf
universal-rag ingest "https://es.wikipedia.org/wiki/Energía_solar"
universal-rag ingest "./informes/*.md"            # globs = ingesta masiva
universal-rag ingest ./podcast.mp3                # → transcripción local + minutos
universal-rag ingest ./clase.mp4                  # → pista de audio → ASR + minutos
universal-rag ingest "https://youtu.be/abc123"    # → YouTube → ASR + minutos
universal-rag ingest "drive://<fileId>"           # → Google Drive (público/compartido)

# Consulta con citas (en media, con minuto exacto)
universal-rag query "¿cada cuánto se limpian los paneles?"

# Colecciones (Fase 4): bases de conocimiento aisladas
universal-rag ingest ./contratos/2025.pdf --collection legal
universal-rag query "¿cláusula de penalización?" --collection legal
universal-rag collections                        # lista colecciones activas

# Chat interactivo en la terminal
universal-rag chat

# Evaluación de calidad con un golden set (precision/recall/MRR)
universal-rag eval examples/golden.jsonl --k 5
universal-rag eval examples/golden.jsonl --json      # reporte completo en JSON

# Administración
universal-rag documents     # lista lo ingestado
universal-rag stats         # documentos, fragmentos, conectores, modo de búsqueda
```

### API REST

```bash
universal-rag serve --port 8100
# → http://localhost:8100/    mini chat UI
# → http://localhost:8100/docs Swagger UI
```

```bash
curl -X POST localhost:8100/ingest -H 'Content-Type: application/json' \
     -d '{"source": "https://ejemplo.com/articulo"}'

curl -X POST localhost:8100/ingest -H 'Content-Type: application/json' \
     -d '{"source": "./entrevista.mp3", "options": {"language": "es", "asr_model": "small"}}'

curl -X POST localhost:8100/query -H 'Content-Type: application/json' \
     -d '{"question": "¿qué dijo sobre costos?", "k": 4}'

# Fase 4 — colecciones y jobs asíncronos
curl -X POST localhost:8100/ingest/text -H 'Content-Type: application/json' \
     -d '{"text": "…", "title": "Nota", "collection": "legal"}'

curl -X POST localhost:8100/ingest/async -H 'Content-Type: application/json' \
     -d '{"source": "./podcast-largo.mp3"}'          # → {"job_id": …}
curl localhost:8100/jobs/<job_id>                     # → estado del job

# Fase 4 — con autenticación (si RAG_API_KEY está definida)
curl localhost:8100/stats -H 'X-API-Key: <tu-key>'
```

Endpoints: `/` · `/health` · `/stats` · `/ingest` · `/ingest/text` · `/ingest/async` · `/jobs` · `/jobs/{id}` · `/query` · `/documents` · `/collections`.

### Como servidor MCP (Claude Desktop, Universal MCP…)

```bash
universal-rag mcp                             # stdio
universal-rag mcp --transport streamable-http --port 8200
```

Herramientas MCP: `rag_ingest`, `rag_ingest_text`, `rag_query`, `rag_stats`, `rag_list_documents`, `rag_delete_document`, `rag_list_collections` — todas aceptan `collection` (Fase 4).

### Docker (Fase 4)

```bash
docker compose up -d --build
curl http://localhost:8100/health
```

Detalles de despliegue (auth, volúmenes, extras `[local]`/`[media]`) en [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## 🧪 Python

```python
from universal_rag import CollectionManager, RAGPipeline

rag = RAGPipeline()
rag.ingest("./docs/contrato.pdf")
rag.ingest("./entrevista.mp3", language="es", asr_model="small")

answer = rag.query("¿cuál es la cláusula de penalización?")
print(answer.answer)
for citation in answer.citations:
    minute = f" {citation.timestamp}" if citation.timestamp else ""
    print(f"  [{citation.source}{minute}] {citation.title} → {citation.quote[:80]}")
#  [audio 03:12–04:05] Entrevista → ...sobre la cláusula de penalización...

# Fase 4: colecciones aisladas que comparten embedder/generator
manager = CollectionManager()
manager.get("legal").ingest("./docs/contrato.pdf")
manager.get("faq").ingest_text("¿Horario? De 9 a 17.", title="FAQ")
manager.get("faq").query("¿cuál es el horario?")
print(manager.list_collections())   # resumen por colección
```

## ⚙️ Configuración (variables `RAG_*`)

| Variable | Default | Descripción |
|---|---|---|
| `RAG_EMBEDDING_PROVIDER` | `auto` | `auto` \| `sentence-transformers` \| `hash` |
| `RAG_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Modelo local multilingüe |
| `RAG_STORE_PROVIDER` | `chroma` | `chroma` \| `memory` |
| `RAG_CHROMA_PATH` | `./data/chroma` | Persistencia del índice |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `900` / `150` | Troceado |
| `RAG_LLM_BASE_URL` | *(vacío → modo extractivo)* | Endpoint compatible OpenAI |
| `RAG_LLM_API_KEY` / `RAG_LLM_MODEL` | — / `gpt-4o-mini` | Credenciales y modelo |
| `RAG_TOP_K` | `5` | Fragmentos recuperados |
| `RAG_RETRIEVAL` | `vector` | `vector` \| `hybrid` (BM25 + vectores, Fase 3) |
| `RAG_HYBRID_FUSION` | `rrf` | `rrf` \| `weighted` (min-max + alpha) |
| `RAG_HYBRID_ALPHA` | `0.55` | Peso vectorial en `weighted` (0 = solo BM25) |
| `RAG_CANDIDATE_K` | `25` | Candidatos antes de rerank/top-k |
| `RAG_RERANKER` | `none` | `none` \| `cross-encoder` (requiere `[local]`) |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder multilingüe |
| `RAG_ASR_MODEL` | `base` | Whisper: `tiny` \| `base` \| `small` \| `medium` \| `large-v3` |
| `RAG_ASR_LANGUAGE` | *(auto)* | `es`, `en`, … (o `language=` por ingesta) |
| `RAG_CONNECTOR_DIRS` | — | Directorios con conectores externos |
| `RAG_DEFAULT_COLLECTION` | `default` | Colección activa si no se indica otra (Fase 4) |
| `RAG_API_KEY` | *(vacío = sin auth)* | Si se define, la API exige `X-API-Key`/`Bearer` (Fase 4) |
| `RAG_JOBS_WORKERS` | `2` | Hilos del pool de ingesta asíncrona (Fase 4) |
| `RAG_JOBS_MAX_HISTORY` | `200` | Jobs recordados en memoria (Fase 4) |

Ejemplo con Ollama local:

```bash
export RAG_LLM_BASE_URL=http://localhost:11434/v1
export RAG_LLM_MODEL=llama3.1
```

## 🏗️ Arquitectura

```
fuente ──▶ ConnectorRegistry (auto-descubrimiento + prioridad) ──▶ Document
              │ text/file/url (F1)      │ audio/video/youtube/drive (F2: ASR local)
              ▼                         ▼   segments (start, end, text)
                                        Chunker ──▶ Chunks ──▶ Embedder ──▶ VectorStore (Chroma)
                                        (texto: párrafos)    (media: por minuto [mm:ss])
                                                                     │
        pregunta ──▶ Retriever ◀──────────────────────────────────────┘
          │  │ vectorial (coseno)
          │  └ híbrido: + BM25 (F3) ── fusión RRF/weighted
          │ RetrievedChunks ──▶ Reranker (F3: cross-encoder, opcional)
                        ▼
                   Generator (OpenAI-compat / echo) ──▶ Answer + Citations [1][2]… [00:00–01:05]
```

Detalles en [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Búsqueda híbrida y evaluación en [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md) · Cómo agregar conectores en [`docs/ADDING_CONNECTORS.md`](docs/ADDING_CONNECTORS.md).

## 🗺️ Hoja de ruta

- **Fase 1 (✅ v0.1.0)** — núcleo + conectores `text`/`file`/`url` + ChromaDB + embeddings locales + API/CLI/MCP.
- **Fase 2 (✅ v0.2.0)** — conectores `audio` (MP3), `video` (MP4), `youtube` y `drive` con ASR **faster-whisper** local y **citas por minuto**.
- **Fase 3 (✅ v0.3.0)** — **búsqueda híbrida** (BM25 + vectores, fusión RRF/weighted), **reranking** (cross-encoder local), **evaluación** (precision/recall/MRR con golden sets) y **mini chat UI**.
- **Fase 4 (✅ v0.4.0)** — **producción y escalado**: colecciones múltiples, jobs asíncronos con polling, auth por API key, métricas runtime, ingesta incremental (skip sin cambios) y despliegue Docker.
- **Fase 5 (futuro)** — conectores S3/Notion, reranking con LLM y extracción de tablas.

> 💡 **Calidad del ASR**: el modelo `base` es rápido y ligero; para español u otros idiomas no ingleses recomendamos `RAG_ASR_MODEL=small` (o `medium`) — la precisión mejora notablemente. Audio con música de fondo o voces sintéticas degrada la precisión de cualquier Whisper.

## 🧰 Desarrollo

```bash
git clone https://github.com/rtostudioco/universal-rag && cd universal-rag
pip install -e ".[dev]"
pytest                # 202 tests (sin red, sin modelos pesados)
ruff check .
```

## 📄 Licencia

[MIT](LICENSE) — Universal RAG Contributors.
