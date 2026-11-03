# Universal RAG

> 🧩 **Pipeline plugable de Retrieval-Augmented Generation** — ingesta universal (texto, archivos, URLs; audio/video/Drive en fases siguientes), búsqueda semántica local y respuestas con **citas verificables**.

Hermano de [Universal MCP](https://github.com/rtostudioco/universal-mcp): misma filosofía **"universal"** — núcleo estable, capacidades que se agregan soltando carpetas, sin tocar el servidor.

[![CI](https://github.com/rtostudioco/universal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/rtostudioco/universal-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)

---

## ✨ Características (Fase 1 — MVP)

| Pieza | Qué hace |
|---|---|
| 🔌 **Conectores plugables** | `text`, `file` (PDF/DOCX/TXT/MD/CSV + globs) y `url` (extracción legible). Agrega los tuyos soltando una carpeta — un conector roto **no derriba el sistema**. |
| ✂️ **Chunker configurable** | División con conciencia de párrafos/frases/palabras y solapamiento ajustable. |
| 🧠 **Embeddings locales** | `sentence-transformers` (multilingüe, privado, sin APIs) con fallback determinista sin dependencias. |
| 🗄️ **ChromaDB embebido** | Persistente, sin servidor; intercambiable por otro store (incl. uno en memoria para tests). |
| 🔍 **Búsqueda semántica** | Recuperación con scores de similitud y filtros por metadatos (`where`). |
| 💬 **Generación plugable** | Cualquier endpoint **compatible OpenAI** (OpenAI, z-ai, Ollama, LM Studio, vLLM…) o modo extractivo sin LLM. |
| 📎 **Citas verificables** | Cada respuesta referencia sus fuentes con `uri`, score y cita textual. |
| 🖥️ **3 interfaces** | **API REST** (FastAPI) + **CLI** (Typer) + **servidor MCP** (para Claude Desktop / Universal MCP). |

## 🚀 Inicio rápido

```bash
pip install universal-rag                 # núcleo (embeddings deterministas)
pip install "universal-rag[local]"        # + embeddings neuronales locales (recomendado)
```

### CLI

```bash
# Ingesta (el conector se resuelve automáticamente)
universal-rag ingest ./docs/manual.pdf
universal-rag ingest "https://es.wikipedia.org/wiki/Energía_solar"
universal-rag ingest "./informes/*.md"          # globs = ingesta masiva

# Consulta con citas
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

curl -X POST localhost:8100/query -H 'Content-Type: application/json' \
     -d '{"question": "¿qué dice el artículo sobre costos?", "k": 4}'
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
rag.ingest("https://blog.ejemplo.com/novedades")

answer = rag.query("¿cuál es la cláusula de penalización?")
print(answer.answer)
for citation in answer.citations:
    print(f"  [{citation.source}] {citation.title} → {citation.quote[:80]}")
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
| `RAG_CONNECTOR_DIRS` | — | Directorios con conectores externos |

Ejemplo con Ollama local:

```bash
export RAG_LLM_BASE_URL=http://localhost:11434/v1
export RAG_LLM_MODEL=llama3.1
```

## 🏗️ Arquitectura

```
fuente ──▶ ConnectorRegistry (auto-descubrimiento) ──▶ Document
                                                        │
                                        Chunker ──▶ Chunks ──▶ Embedder ──▶ VectorStore (Chroma)
                                                                              │
        pregunta ──▶ Retriever ◀──────────────────────────────────────────────┘
                        │ RetrievedChunks
                        ▼
                   Generator (OpenAI-compat / echo) ──▶ Answer + Citations [1][2]…
```

Detalles en [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Cómo agregar conectores en [`docs/ADDING_CONNECTORS.md`](docs/ADDING_CONNECTORS.md).

## 🗺️ Hoja de ruta

- **Fase 1 (✅ esta versión)** — núcleo + conectores `text`/`file`/`url` + ChromaDB + embeddings locales + API/CLI/MCP.
- **Fase 2** — conectores `audio` (MP3), `video` (MP4) y `youtube` con ASR *faster-whisper* y citas por minuto + `drive`.
- **Fase 3** — búsqueda híbrida (BM25 + vectores), reranking, evaluación de calidad y mini UI de chat.

## 🧰 Desarrollo

```bash
git clone https://github.com/rtostudioco/universal-rag && cd universal-rag
pip install -e ".[dev]"
pytest                # suite completa (sin red, sin modelos pesados)
ruff check .
```

## 📄 Licencia

[MIT](LICENSE) — Universal RAG Contributors.
