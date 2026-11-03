"""Fusión de rankings para búsqueda híbrida (Fase 3).

Dos estrategias intercambiables:

* ``rrf``      — **Reciprocal Rank Fusion** (por defecto): suma
                 ``1 / (k + rank)`` para cada lista en la que aparece el
                 documento. Robusta a escalas de score distintas (coseno
                 vectorial vs BM25) y estándar en la literatura.
* ``weighted`` — suma ponderada tras normalizar cada lista a ``[0, 1]``
                 (min-max): ``alpha·vector + (1−alpha)·bm25``.
"""

from __future__ import annotations


def rrf_fuse(rankings: list[list[int]], k: int = 60, top_n: int = 10) -> list[tuple[int, float]]:
    """Fusiona rankings (listas de índices ordenados por relevancia) con RRF.

    Devuelve ``[(índice, score_rrf), …]`` ordenado descendente, hasta ``top_n``.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ordered[: max(0, top_n)]


def _min_max(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not pairs:
        return []
    values = [score for _, score in pairs]
    low, high = min(values), max(values)
    span = high - low
    if span <= 0:
        return [(idx, 1.0) for idx, _ in pairs]
    return [(idx, (score - low) / span) for idx, score in pairs]


def weighted_fuse(
    vector_pairs: list[tuple[int, float]],
    bm25_pairs: list[tuple[int, float]],
    alpha: float = 0.55,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Suma ponderada: ``alpha·vector + (1−alpha)·bm25`` con scores min-max.

    ``alpha = 1`` es búsqueda puramente vectorial; ``alpha = 0`` puramente léxica.
    """
    total: dict[int, float] = {}
    for idx, score in _min_max(vector_pairs):
        total[idx] = total.get(idx, 0.0) + alpha * score
    for idx, score in _min_max(bm25_pairs):
        total[idx] = total.get(idx, 0.0) + (1.0 - alpha) * score
    ordered = sorted(total.items(), key=lambda pair: pair[1], reverse=True)
    return ordered[: max(0, top_n)]
