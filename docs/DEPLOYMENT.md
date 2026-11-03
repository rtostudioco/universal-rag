# Despliegue de Universal RAG (Fase 4)

Guía para llevar Universal RAG a producción con Docker. Para desarrollo local
ver el [README](../README.md).

## 1. Rápido: docker compose

```bash
git clone https://github.com/rtostudioco/universal-rag && cd universal-rag
cp .env.example .env            # edita lo que necesites (RAG_API_KEY recomendado)
docker compose up -d --build

curl http://localhost:8100/health
# {"status":"ok","version":"0.4.0"}
```

Qué incluye el compose:

| Aspecto | Valor |
|---|---|
| Puerto | `8100` (API + mini chat UI en `/`) |
| Persistencia | `./data:/app/data` — el índice Chroma sobrevive a reinicios |
| Healthcheck | `curl /health` cada 30 s (con `start_period` de 10 s) |
| Reinicio | `unless-stopped` |

## 2. Dockerfile sin compose

```bash
docker build -t universal-rag .
docker run -d --name rag -p 8100:8100 -v "$PWD/data:/app/data" \
    -e RAG_API_KEY=mi-secreto universal-rag
```

> **Nota**: la imagen base instala solo el stack ligero (embeddings hash +
> ChromaDB). Para embeddings neuronales o ASR local, descomenta las líneas
> `pip install ".[local]"` / `".[media]"` del Dockerfile (imagen más grande:
> +2 GB con `[local]`, +3 GB con `[media]`).

## 3. Autenticación (Fase 4)

Define `RAG_API_KEY` y todos los endpoints (excepto `/` y `/health`) exigirán:

```bash
curl localhost:8100/stats -H 'X-API-Key: mi-secreto'
# o equivalente:
curl localhost:8100/stats -H 'Authorization: Bearer mi-secreto'
```

Sin la clave → `401`. La mini chat UI detecta el 401 y pregunta la clave una
vez (la guarda en `localStorage`). Usa HTTPS delante (reverse proxy, Caddy,
nginx) si la API queda expuesta a internet: las API keys viajan en headers.

## 4. Colecciones múltiples (Fase 4)

Un solo despliegue puede servir varias bases de conocimiento aisladas:

```bash
# ingesta por colección
curl -X POST localhost:8100/ingest/text -H 'Content-Type: application/json' \
     -H 'X-API-Key: mi-secreto' \
     -d '{"text": "…", "title": "Nota", "collection": "legal"}'

# consulta por colección
curl -X POST localhost:8100/query -H 'Content-Type: application/json' \
     -H 'X-API-Key: mi-secreto' \
     -d '{"question": "…", "collection": "legal"}'

# listar y vaciar
curl localhost:8100/collections -H 'X-API-Key: mi-secreto'
curl -X DELETE localhost:8100/collections/legal -H 'X-API-Key: mi-secreto'
```

Con ChromaDB cada colección persiste en `<RAG_CHROMA_PATH>/<nombre>`; la
colección `default` mantiene su ruta original (retro-compatible con índices
de Fases 1–3).

## 5. Jobs asíncronos (Fase 4)

Para fuentes lentas (audio/video largos, lotes) usa la ingesta en background:

```bash
JOB=$(curl -s -X POST localhost:8100/ingest/async \
      -H 'Content-Type: application/json' -H 'X-API-Key: mi-secreto' \
      -d '{"source": "./podcast-3h.mp3"}' | jq -r .job_id)

curl localhost:8100/jobs/$JOB -H 'X-API-Key: mi-secreto'
# {"status":"running"…} → {"status":"done","result":{…}}
```

Los jobs viven **en memoria del proceso** (no sobreviven reinicios); ajusta
`RAG_JOBS_WORKERS` (hilos) y `RAG_JOBS_MAX_HISTORY` según carga.

## 6. Ingesta incremental (Fase 4)

Re-ingerir el mismo archivo/URL **no re-vectoriza nada** si el contenido no
cambió (la respuesta reporta `unchanged: 1`). Ideal para cron jobs:

```bash
*/15 * * * * curl -s -X POST localhost:8100/ingest \
    -H 'Content-Type: application/json' -H 'X-API-Key: mi-secreto' \
    -d '{"source": "/datos/carpeta/*.md"}' > /dev/null
```

Para forzar re-indexación usa `"force": true`.

## 7. Métricas y monitoreo (Fase 4)

`GET /stats` expone, además del estado del pipeline, un bloque `runtime`:

```json
{
  "runtime": {
    "uptime_seconds": 3600.2,
    "requests": 1520,
    "queries": 940,
    "ingests": 37,
    "ingests_async": 4,
    "errors": 2,
    "documents_deleted": 1
  },
  "collections": [{"name": "legal", "documents": 12, "chunks": 340, "store": "chroma", "retrieval": "vector"}]
}
```

El healthcheck de Docker/compose y `/health` no requieren autenticación, así
que orquestadores (Compose, Kubernetes, Fly.io…) pueden sondear sin claves.

## 8. Checklist de producción

- [ ] `RAG_API_KEY` definida y tráfico por HTTPS (reverse proxy).
- [ ] Volumen montado en `RAG_CHROMA_PATH` (backup de esa carpeta = backup del índice).
- [ ] `RAG_EMBEDDING_PROVIDER=sentence-transformers` (o imagen con `[local]`) para búsqueda semántica real.
- [ ] Opcional: `RAG_RETRIEVAL=hybrid` + `RAG_RERANKER=cross-encoder` para máxima calidad.
- [ ] Cron de re-ingesta incremental para fuentes que cambian.
- [ ] `RAG_LLM_*` apuntando a tu LLM (o sin LLM: el modo extractivo funciona offline).
