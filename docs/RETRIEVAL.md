# Búsqueda híbrida, reranking y evaluación (Fase 3)

Guía práctica para sacar el máximo a la recuperación de Universal RAG. Los tres componentes son **opt-in e independientes**: puedes activar solo el híbrido, solo el reranker, o ambos.

---

## 1. ¿Por qué híbrido?

| Canal | Fortaleza | Debilidad |
|---|---|---|
| **Vectorial** (embeddings) | Sinónimos, paráfrasis, intención semántica | Diluye códigos, nombres propios, referencias exactas |
| **BM25** (léxico) | Términos exactos: `RTX-4090`, `ISO-9001`, `Quimbaya` | No entiende sinónimos ni paráfrasis |

La búsqueda **híbrida** ejecuta ambos y **funde** los rankings: cada canal rescata lo que el otro pierde. Caso típico: *"¿cuánto cuesta el modelo XYZ-2000?"* — el embedder puede no dar relevancia máxima al chunk con `XYZ-2000`, pero BM25 lo pone primero con seguridad.

## 2. Activarlo

```bash
export RAG_RETRIEVAL=hybrid
universal-rag query "¿cuánto cuesta el modelo XYZ-2000?"
```

o en Python:

```python
from universal_rag.core.config import Settings
from universal_rag.core.pipeline import RAGPipeline

rag = RAGPipeline(settings=Settings(retrieval="hybrid"))
answer = rag.query("¿cuánto cuesta el modelo XYZ-2000?")
print(answer.retrieval)  # "hybrid"
```

> El índice BM25 se construye en memoria sobre el corpus, se **cachea** y se **invalida automáticamente** tras cada ingesta/borrado/vaciado. Cero dependencias nuevas (BM25Okapi propio, ~80 líneas).

## 3. Estrategias de fusión

| `RAG_HYBRID_FUSION` | Cómo funciona | Cuándo usarla |
|---|---|---|
| `rrf` (default) | Reciprocal Rank Fusion: `score = Σ 1/(k + rank)` con `k=60`. Solo mira **posiciones**, no valores de score. | Default robusta: inmune a las escalas dispares de coseno vs BM25. |
| `weighted` | Normaliza cada lista a `[0,1]` (min-max) y suma `alpha·vector + (1−alpha)·bm25`. | Cuando quieres afinar el balance: `RAG_HYBRID_ALPHA=0.3` favorece términos exactos; `0.8`, la semántica. |

## 4. Reranking (precisión máxima)

El flujo completo con reranker: se recuperan `RAG_CANDIDATE_K` (default 25) candidatos → un **cross-encoder** puntúa cada par `(pregunta, fragmento)` leyéndolos juntos (mucho más fino que el bi-encoder de la búsqueda) → se devuelve el top-k reordenado.

```bash
pip install "universal-rag[local]"        # sentence-transformers
export RAG_RERANKER=cross-encoder
export RAG_RERANK_MODEL=BAAI/bge-reranker-base   # multilingüe (buen español), ~1 GB
universal-rag query "¿cuál es la cláusula de penalización?"
```

Notas:
- El score del cross-encoder pasa por sigmoide → citas con scores en `[0, 1]` monótonos con el orden.
- Primera ejecución descarga el modelo (luego corre 100% local).
- El reranker también se aplica en modo `vector`; es independiente del híbrido.
- Latencia: O(candidatos) por consulta; para latencia mínima baja `RAG_CANDIDATE_K` a 10–15.

## 5. Evaluación de calidad

Define un **golden set** JSONL — una línea por caso:

```jsonl
{"query": "¿cada cuánto tiempo hay que limpiar los paneles solares?", "relevant": ["Guía de energía solar"]}
{"query": "¿cuántas familias abastece un aerogenerador de 3 MW?", "relevant": ["Manual de energía eólica"]}
```

`relevant` acepta **IDs de documento, URIs o títulos** (comparación insensible a mayúsculas). Ejecuta:

```bash
universal-rag eval examples/golden.jsonl --k 5
universal-rag eval examples/golden.jsonl --k 5 --json   # reporte completo
```

Métricas agregadas y por caso:

| Métrica | Significado | Buen valor |
|---|---|---|
| `precision` | fracción de lo recuperado que era relevante | alto en corpus limpios |
| `recall` | fracción de lo relevante que fue recuperado | **la métrica clave** en RAG |
| `mrr` | 1/posición del primer acierto (media) | > 0.8 si el top-1 suele acertar |
| `hit_rate` | % de consultas con al menos un acierto | cerca de 1.0 |

Flujo recomendado: fija el golden set → mide baseline (`vector`) → activa `hybrid` → compara → añade reranker → compara. Sin medir, estás adivinando.

En Python:

```python
from universal_rag.core.evaluation import evaluate, load_golden

report = evaluate(rag, load_golden("golden.jsonl"), k=5)
print(report["recall"], report["mrr"])
```

## 6. Chat incluido

```bash
universal-rag serve --port 8100
# → http://localhost:8100/   mini chat UI (autocontenida, sin CDNs)
```

La UI muestra modelo, latencia, modo de búsqueda, número de fragmentos y las citas con sus minutos `[mm:ss–mm:ss]` cuando la fuente es audio/video.

## 7. Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `RAG_RETRIEVAL` | `vector` | `vector` \| `hybrid` |
| `RAG_HYBRID_FUSION` | `rrf` | `rrf` \| `weighted` |
| `RAG_HYBRID_RRF_K` | `60` | Constante k de RRF |
| `RAG_HYBRID_ALPHA` | `0.55` | Peso vectorial en `weighted` |
| `RAG_CANDIDATE_K` | `25` | Candidatos antes de rerank/top-k |
| `RAG_RERANKER` | `none` | `none` \| `cross-encoder` |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-base` | Modelo del cross-encoder |
