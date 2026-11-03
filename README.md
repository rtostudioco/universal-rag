# Universal RAG

> 🧩 **Pipeline plugable de Retrieval-Augmented Generation** — ingesta universal (texto, archivos, URLs, **audio, video, YouTube, Google Drive**), búsqueda semántica local y respuestas con **citas verificables, incluidas citas por minuto** en audio/video.

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

# Administración
universal-rag documents     # lista lo ingestado
universal-rag stats         # documentos, fragmentos, conectores
```

### API REST

```bash
universal-rag serve --port 8100
```

```bash
curl -X POST localhost:8100/ingest -H 'Content-Type: application/json' \
     -d '{"source": "https://ejemplo.com/articulo"}'

curl -X POST localhost:8100/ingest -H 'Content-Type: application/json' \
     -d '{"source": "./entrevista.mp3", "options": {"language": "es", "asr_model": "small"}}'

curl -X POST localhost:8100/query -H 'Content-Type: application/json' \
     -d '{"question": "¿qué dijo sobre costos?", "k": 4}'
```

### Como servidor MCP (Claude Desktop, Universal MCP…)

```bash
universal-rag mcp                             # stdio
universal-rag mcp --transport streamable-http --port 8200
```

Herramientas MCP: `rag_ingest`, `rag_ingest_text`, `rag_query`, `rag_stats`, `rag_list_documents`, `rag_delete_document`.

## 🧪 Python

```python
from universal_rag import RAGPipeline

rag = RAGPipeline()
rag.ingest("./docs/contrato.pdf")
rag.ingest("./entrevista.mp3", language="es", asr_model="small")

answer = rag.query("¿cuál es la cláusula de penalización?")
print(answer.answer)
for citation in answer.citations:
    minute = f" {citation.timestamp}" if citation.timestamp else ""
    print(f"  [{citation.source}{minute}] {citation.title} → {citation.quote[:80]}")
#  [audio 03:12–04:05] Entrevista → ...sobre la cláusula de penalización...
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
| `RAG_ASR_MODEL` | `base` | Whisper: `tiny` \| `base` \| `small` \| `medium` \| `large-v3` |
| `RAG_ASR_LANGUAGE` | *(auto)* | `es`, `en`, … (o `language=` por ingesta) |
| `RAG_CONNECTOR_DIRS` | — | Directorios con conectores externos |

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
                        │ RetrievedChunks
                        ▼
                   Generator (OpenAI-compat / echo) ──▶ Answer + Citations [1][2]… [00:00–01:05]
```

Detalles en [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Cómo agregar conectores en [`docs/ADDING_CONNECTORS.md`](docs/ADDING_CONNECTORS.md).

## 🗺️ Hoja de ruta

- **Fase 1 (✅ v0.1.0)** — núcleo + conectores `text`/`file`/`url` + ChromaDB + embeddings locales + API/CLI/MCP.
- **Fase 2 (✅ v0.2.0)** — conectores `audio` (MP3), `video` (MP4), `youtube` y `drive` con ASR **faster-whisper** local y **citas por minuto**.
- **Fase 3 (próxima)** — búsqueda híbrida (BM25 + vectores), reranking, evaluación de calidad y mini UI de chat.

> 💡 **Calidad del ASR**: el modelo `base` es rápido y ligero; para español u otros idiomas no ingleses recomendamos `RAG_ASR_MODEL=small` (o `medium`) — la precisión mejora notablemente. Audio con música de fondo o voces sintéticas degrada la precisión de cualquier Whisper.

## 🧰 Desarrollo

```bash
git clone https://github.com/rtostudioco/universal-rag && cd universal-rag
pip install -e ".[dev]"
pytest                # suite completa (sin red, sin modelos pesados)
ruff check .
```

## 📄 Licencia

[MIT](LICENSE) — Universal RAG Contributors.
