"""Tests de evaluación de calidad de recuperación (Fase 3)."""

import json

import pytest

from universal_rag.core.evaluation import EvalCase, evaluate, load_golden

GOLDEN = (
    '{"query": "¿cada cuánto se limpian los paneles solares?", "relevant": ["Guía de energía solar"]}\n'
    '{"query": "¿cuántas familias abastece un aerogenerador?", "relevant": ["Manual de energía eólica"]}\n'
    '{"query": "¿qué dice la base sobre alpinismo extremo?", "relevant": ["Documento inexistente"]}\n'
)


@pytest.fixture()
def golden_path(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text(GOLDEN, encoding="utf-8")
    return path


def test_load_golden_jsonl(golden_path):
    cases = load_golden(golden_path)
    assert len(cases) == 3
    assert cases[0].query.startswith("¿cada cuánto")
    assert cases[0].relevant == ["Guía de energía solar"]


def test_load_golden_json_array(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{"query": "q1", "relevant": ["a"]}, {"query": "q2"}]), encoding="utf-8")
    cases = load_golden(path)
    assert len(cases) == 2
    assert cases[1].relevant == []


def test_load_golden_ignora_lineas_vacias(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text('\n{"query": "q", "relevant": []}\n\n', encoding="utf-8")
    assert len(load_golden(path)) == 1


def test_evaluar_con_golden_set(pipeline_con_datos, golden_path):
    cases = load_golden(golden_path)
    report = evaluate(pipeline_con_datos, cases, k=5)

    assert report["cases"] == 3
    assert report["k"] == 5
    per_case = report["per_case"]
    # Caso 1 y 2: relevantes por título → acierto.
    assert per_case[0]["hits"] >= 1
    assert per_case[0]["mrr"] > 0
    assert per_case[1]["hits"] >= 1
    # Caso 3: documento inexistente → sin acierto.
    assert per_case[2]["hits"] == 0
    assert per_case[2]["mrr"] == 0.0
    assert report["hit_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"] <= 1.0


def test_evaluar_relevante_por_document_id(pipeline_con_datos):
    result = pipeline_con_datos.ingest_text("Contenido relevante de prueba.", title="Doc prueba")
    doc_id = result.document_ids[0]
    report = evaluate(pipeline_con_datos, [EvalCase(query="contenido relevante de prueba", relevant=[doc_id])], k=3)
    assert report["per_case"][0]["hits"] >= 1
    assert report["mrr"] == 1.0


def test_evaluar_sin_relevantes_recalled_metrics_cero(pipeline):
    report = evaluate(pipeline, [EvalCase(query="cualquier cosa", relevant=["nada"])], k=3)
    assert report["cases"] == 1
    assert report["per_case"][0]["hits"] == 0
    assert report["mrr"] == 0.0
    assert report["recall"] == 0.0


def test_evaluar_match_insensible_a_mayusculas(pipeline_con_datos):
    report = evaluate(
        pipeline_con_datos,
        [EvalCase(query="energía solar", relevant=["guía de energía solar"])],  # minúsculas
        k=3,
    )
    assert report["per_case"][0]["hits"] >= 1
