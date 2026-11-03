"""Evaluación de calidad de recuperación (Fase 3).

Un *golden set* es un JSONL (una línea = un caso) con::

    {"query": "¿cada cuánto se limpian los paneles?", "relevant": ["Guía de energía solar"]}

``relevant`` acepta IDs de documento, URIs o títulos (comparación insensible a
mayúsculas); también un array JSON ``[{...}, ...]``. Métricas sobre el top-k
recuperado:

* ``precision`` — fracción de lo recuperado que es relevante (hits / recuperados).
* ``recall``    — fracción de lo relevante que fue recuperado (hits / relevantes).
* ``mrr``       — Reciprocal Rank medio del primer acierto (0 si no hay acierto).
* ``hit_rate``  — fracción de consultas con al menos un acierto.

Uso::

    from universal_rag.core.evaluation import evaluate, load_golden
    report = evaluate(pipeline, load_golden("golden.jsonl"), k=5)
    # o por CLI: universal-rag eval golden.jsonl --k 5
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from universal_rag.core.utils import get_logger

if TYPE_CHECKING:
    from universal_rag.core.pipeline import RAGPipeline

logger = get_logger("evaluation")


@dataclass
class EvalCase:
    """Caso de evaluación: consulta + claves de documentos relevantes."""

    query: str
    relevant: list[str] = field(default_factory=list)


@dataclass
class EvalCaseResult:
    """Resultado de un caso individual."""

    query: str
    relevant: list[str]
    retrieved: list[str]  # claves recuperadas (id/uri/título) para inspección
    hits: int
    precision: float
    recall: float
    mrr: float


def load_golden(path: str | Path) -> list[EvalCase]:
    """Carga un golden set en JSONL (una línea = un caso) o JSON (lista)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    cases: list[EvalCase] = []
    if path.suffix.lower() == ".jsonl" or text.lstrip().startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                EvalCase(query=str(data["query"]), relevant=[str(item) for item in data.get("relevant", [])])
            )
    else:
        for item in json.loads(text):
            cases.append(
                EvalCase(query=str(item["query"]), relevant=[str(sub) for sub in item.get("relevant", [])])
            )
    valid = [case for case in cases if case.query.strip()]
    logger.info("Golden set cargado: %d caso(s) desde %s", len(valid), path)
    return valid


def _keys_of(citation: Any) -> set[str]:  # noqa: ANN401 — Citation desacoplado para tests
    data = citation if isinstance(citation, dict) else citation.model_dump()
    raw = {str(data.get("uri", "")), str(data.get("title", "")), str(data.get("document_id", ""))}
    return {key.strip().casefold() for key in raw if key and key.strip()}


def evaluate(pipeline: RAGPipeline, cases: list[EvalCase], k: int = 5) -> dict[str, Any]:
    """Ejecuta los casos contra el pipeline y agrega métricas de recuperación."""
    per_case: list[EvalCaseResult] = []
    for case in cases:
        answer = pipeline.query(case.query, k=k)
        wanted = {item.strip().casefold() for item in case.relevant if item and item.strip()}
        hits = 0
        first_rank = 0
        retrieved_keys: list[str] = []
        for rank, citation in enumerate(answer.citations, start=1):
            keys = _keys_of(citation)
            retrieved_keys.append(next(iter(keys)) if keys else "?")
            if wanted & keys:
                hits += 1
                if not first_rank:
                    first_rank = rank
        retrieved_total = len(answer.citations)
        per_case.append(
            EvalCaseResult(
                query=case.query,
                relevant=case.relevant,
                retrieved=retrieved_keys,
                hits=hits,
                precision=round(hits / retrieved_total, 4) if retrieved_total else 0.0,
                recall=round(hits / len(wanted), 4) if wanted else 0.0,
                mrr=round(1.0 / first_rank, 4) if first_rank else 0.0,
            )
        )
    total = len(per_case)

    def _mean(attr: str) -> float:
        return round(sum(getattr(case, attr) for case in per_case) / total, 4) if total else 0.0

    report = {
        "k": k,
        "cases": total,
        "precision": _mean("precision"),
        "recall": _mean("recall"),
        "mrr": _mean("mrr"),
        "hit_rate": round(sum(1 for case in per_case if case.hits > 0) / total, 4) if total else 0.0,
        "per_case": [asdict(case) for case in per_case],
    }
    logger.info(
        "Evaluación k=%d · %d casos · precision=%s recall=%s mrr=%s hit_rate=%s",
        k,
        total,
        report["precision"],
        report["recall"],
        report["mrr"],
        report["hit_rate"],
    )
    return report
