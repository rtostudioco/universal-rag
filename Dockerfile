# Universal RAG — imagen de producción (Fase 4)
# Incluye el stack base (ChromaDB + FastAPI). Los extras pesados —
# sentence-transformers ([local]) y faster-whisper/yt-dlp ([media]) — se pueden
# añadir descomentando las líneas marcadas abajo.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias del sistema mínimas (trafilatura/lxml no las necesitan en slim;
# se deja curl para el healthcheck manual si se desea).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Extras [local] (embeddings neuronales) y [media] (ASR) son opcionales:
RUN pip install --no-cache-dir .
# Con embeddings neuronales locales:
# RUN pip install --no-cache-dir ".[local]"
# Con ingesta de audio/video/YouTube:
# RUN pip install --no-cache-dir ".[local,media]"

# Datos persistentes (montar un volumen aquí)
ENV RAG_CHROMA_PATH=/app/data/chroma
RUN mkdir -p /app/data/chroma
VOLUME ["/app/data"]

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://127.0.0.1:8100/health || exit 1

CMD ["uvicorn", "universal_rag.api.app:app", "--host", "0.0.0.0", "--port", "8100"]
