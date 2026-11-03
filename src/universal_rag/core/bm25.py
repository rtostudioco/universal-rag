"""BM25 (Okapi) — búsqueda léxica sparse sin dependencias externas (Fase 3).

Complementa la búsqueda vectorial: mientras los embeddings capturan similitud
semántica, BM25 captura coincidencia léxica exacta (nombres propios, códigos,
referencias, términos raros) que los embeddings suelen diluir.

La fusión de ambos rankings vive en :mod:`universal_rag.core.hybrid` y el
:class:`~universal_rag.core.retriever.Retriever` orquesta todo.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenización simple multilingüe: minúsculas + palabras Unicode."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Índice BM25Okapi en memoria sobre un corpus de textos.

    ``k1`` controla la saturación de la frecuencia de término y ``b`` la
    normalización por longitud de documento (valores estándar de la literatura).
    Cero dependencias externas: se reconstruye en milisegundos para corpus de
    hasta decenas de miles de fragmentos.
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: list[Counter[str]] = [Counter(tokenize(doc)) for doc in corpus]
        self._len = [sum(counter.values()) for counter in self._tf]
        self.n = len(corpus)
        self._avgdl = (sum(self._len) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for counter in self._tf:
            df.update(counter.keys())
        # IDF variante "+1" (Robertson/Sparck-Jones suavizada): nunca negativa.
        self._idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """Score BM25 de cada documento del corpus para la consulta."""
        query_tokens = tokenize(query)
        if not query_tokens or not self.n:
            return [0.0] * self.n
        scores: list[float] = []
        for counter, doc_len in zip(self._tf, self._len, strict=False):
            if not counter:
                scores.append(0.0)
                continue
            norm = self.k1 * (1.0 - self.b + self.b * (doc_len / (self._avgdl or 1.0)))
            score = 0.0
            for term in query_tokens:
                freq = counter.get(term, 0)
                if not freq:
                    continue
                score += self._idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + norm)
            scores.append(score)
        return scores

    def top(self, query: str, n: int = 10) -> list[tuple[int, float]]:
        """``[(índice, score), …]`` de los ``n`` documentos con mejor score (> 0)."""
        ranked = sorted(
            ((idx, score) for idx, score in enumerate(self.scores(query)) if score > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[: max(0, n)]
