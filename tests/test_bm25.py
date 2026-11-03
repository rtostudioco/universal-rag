"""Tests del índice BM25 (búsqueda léxica — Fase 3)."""

from universal_rag.core.bm25 import BM25Index, tokenize

CORPUS = [
    "La energía solar fotovoltaica convierte la luz del sol en electricidad.",
    "El aerogenerador transforma la energía cinética del viento en electricidad.",
    "Mantenimiento preventivo: limpieza de paneles cada tres meses y revisión del inversor.",
]


def test_tokenize_minusculas_y_unicode():
    tokens = tokenize("La Energía SOLAR ¡fotovoltaica!")
    assert tokens == ["la", "energía", "solar", "fotovoltaica"]


def test_tokenize_vacio():
    assert tokenize("") == []
    assert tokenize(".,;:!?") == []


def test_bm25_ranking_por_termino_exacto():
    index = BM25Index(CORPUS)
    top = index.top("aerogenerador viento", n=3)
    assert len(top) == 1  # solo el documento eólico contiene ambos términos
    assert top[0][0] == 1


def test_bm25_ranking_varios_documentos():
    index = BM25Index(CORPUS)
    top = index.top("electricidad", n=3)
    assert {idx for idx, _ in top} == {0, 1}
    assert top[0][1] > top[1][1]


def test_bm25_todos_los_scores_cero_sin_coincidencia():
    index = BM25Index(CORPUS)
    assert all(score == 0.0 for score in index.scores("zzz término inexistente"))


def test_bm25_termino_comun_idf_positivo():
    """'electricidad' aparece en 2 de 3 docs: IDF variante +1 nunca es negativa."""
    index = BM25Index(CORPUS)
    scores = index.scores("electricidad")
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0.0


def test_bm25_corpus_vacio():
    index = BM25Index([])
    assert index.scores("cualquier cosa") == []
    assert index.top("cualquier cosa") == []


def test_bm25_top_n_limita():
    index = BM25Index(CORPUS)
    assert len(index.top("energía electricidad", n=1)) == 1
    assert len(index.top("energía electricidad", n=10)) <= 3


def test_bm25_documento_largo_no_domina_por_longitud():
    """La normalización por longitud (b=0.75) penaliza documentos inflados."""
    corpus = [
        "paneles " + "palabras ".join(f"relleno{i}" for i in range(200)) + " paneles",
        "paneles limpios rinden más",
    ]
    index = BM25Index(corpus)
    top = index.top("paneles", n=2)
    assert top[0][0] == 1  # el corto y al grano gana
