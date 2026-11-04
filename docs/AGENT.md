# Agente multi-hop (Fase 8)

El **agente RAG** responde preguntas complejas en varios pasos: en lugar de una
única búsqueda, descompone la pregunta en **sub-consultas**, busca cada una,
**fusiona** los fragmentos recuperados (dedupe por `chunk_id` conservando el
mejor score), **refina** las sub-consultas sin resultados y **sintetiza** una
respuesta final con citas `[n]` verificables sobre el ranking combinado.
Cada paso queda trazado en la respuesta (`AgentAnswer.steps`).

```
Pregunta  ──►  PLAN (LLM o heurística)  ──►  sub-consultas [s1, s2, …]
                    │
                    ▼
              SEARCH s1 … sn   (retriever; refinamiento si 0 resultados)
                    │
                    ▼
        FUSIÓN (dedupe chunk_id, mejor score, top-k)
                    │
                    ▼
        SÍNTESIS (generador sobre contextos fusionados) ──► AgentAnswer
                    (answer + citations + steps + sub_queries)
```

## Por qué multi-hop

Una pregunta como *«¿Qué mantenimiento necesita un panel solar y cuánto dura
el inversor?»* mezcla dos temas distintos: una sola búsqueda suele recuperar
fragmentos de solo uno. El agente la separa en dos búsquedas y sintetiza sobre
ambas — con citas que apuntan al fragmento exacto de cada sub-pregunta.

## Planificación

| Modo (`RAG_AGENT_PLANNING`) | Comportamiento |
|---|---|
| `auto` (default) | LLM si el generador soporta `chat()` (OpenAI-compat); si no, heurística |
| `heuristic` | Determinista, sin LLM: separadores (`?`, `;`, saltos, fin de frase) y conjunción « y »; fusiona fragmentos cortos |
| `llm` | Fuerza planificación LLM (JSON array de sub-consultas) con fallback heurístico si falla |

La **reformulación** de sub-consultas sin resultados sigue la misma regla:
LLM si está disponible, si no extracción de palabras de contenido
(stopwords es/en incluidas, cero dependencias).

## API

```bash
# Respuesta multi-hop con plan y pasos
curl -X POST http://localhost:8100/query/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué mantenimiento necesita un panel solar y cuánto dura el inversor?"}'

# Streaming SSE: plan → step* → synthesize → token* → done
curl -N -X POST http://localhost:8100/query/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "…", "max_steps": 3, "planning": "heuristic"}'
```

`AgentAnswer` (recortado):

```json
{
  "question": "…",
  "answer": "… [1][2]",
  "citations": [{"chunk_id": "…", "quote": "…", "score": 0.71}],
  "model": "gpt-4o-mini",
  "planner": "gpt-4o-mini",
  "sub_queries": ["mantenimiento panel solar", "vida útil inversor"],
  "steps": [
    {"step": 0, "action": "plan", "note": "gpt-4o-mini: 2 sub-consulta(s)"},
    {"step": 1, "action": "search", "query": "mantenimiento panel solar", "found": 4},
    {"step": 2, "action": "search", "query": "vida útil inversor", "found": 3},
    {"step": 3, "action": "synthesize", "found": 5, "note": "5 contexto(s) fusionado(s)"}
  ],
  "context_count": 5,
  "retrieval": "hybrid",
  "latency_ms": 812.4
}
```

## CLI

```bash
universal-rag ask "¿Qué mantenimiento necesita un panel solar y cuánto dura el inversor?"
universal-rag ask "…" --max-steps 3 --planning heuristic --quiet -C legal
```

Imprime la tabla del plan (paso, consulta, hallazgos, nota), la respuesta en un
panel y la tabla de fuentes citadas (`--quiet` oculta el plan).

## MCP

```
rag_ask(question, k=None, collection=None, max_steps=None, planning=None)
```

Herramienta pensada para preguntas complejas desde Claude Desktop / Universal
MCP; `rag_query` sigue siendo la opción ligera para preguntas simples.

## Workbench

En la pestaña **Chat**, activa el interruptor **«agente»**: la conversación usa
`/query/agent/stream` y muestra badges con el planificador, cada sub-consulta
con sus hallazgos y el número de contextos fusionados antes del texto final.

## Configuración

| Variable | Default | Descripción |
|---|---|---|
| `RAG_AGENT_MAX_STEPS` | `3` | Máximo de sub-consultas por pregunta (1–10) |
| `RAG_AGENT_PLANNING` | `auto` | `auto` \| `heuristic` \| `llm` |

## Python

```python
from universal_rag import RAGPipeline, Settings

pipeline = RAGPipeline(Settings())
pipeline.ingest_text("…")

result = pipeline.agent_ask("¿A y B?")     # AgentAnswer
for step in result.steps:
    print(step.step, step.action, step.query, step.found, step.note)
print(result.answer, result.citations)

for event in pipeline.stream_agent_ask("¿A y B?"):   # streaming
    if event["type"] == "plan":
        print("plan:", event["sub_queries"], "vía", event["planner"])
```

## Notas de diseño

- **Stateless**: el agente vive sobre el pipeline; no guarda estado entre
  preguntas (la memoria conversacional de Fase 7 es ortogonal y el modo agente
  no la usa).
- **Caché semántica**: las preguntas del agente no pasan por la caché de
  respuestas (el plan depende de la pregunta; el coste extra es la búsqueda,
  no el LLM de síntesis).
- **Límites acotados**: `max_steps` acota el número de búsquedas y la
  reformulación es de un solo intento por sub-consulta — sin bucles abiertos.
- **Cero dependencias nuevas**: la heurística es regex propia; la planificación
  LLM reutiliza el endpoint `/chat/completions` ya configurado.
