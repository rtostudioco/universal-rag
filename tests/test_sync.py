"""Tests del planificador de sincronización (Fase 7) — pipeline falso, sin red."""

from __future__ import annotations

import json
import time

import pytest

from universal_rag.core.models import IngestResult
from universal_rag.core.sync import SyncScheduler


class FakePipeline:
    """Doble de prueba con la superficie mínima que usa el planificador."""

    def __init__(self, fail: bool = False, documents: int = 1) -> None:
        self.calls: list[tuple[str, str | None, dict]] = []
        self.fail = fail
        self.documents = documents

    def ingest(self, source: str, connector: str | None = None, **options) -> IngestResult:
        self.calls.append((source, connector, options))
        if self.fail:
            return IngestResult(connector=connector or "?", errors=[f"boom en {source}"])
        return IngestResult(connector=connector or "text", documents=self.documents, chunks=2)


@pytest.fixture()
def fake_pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture()
def scheduler(fake_pipeline) -> SyncScheduler:
    return SyncScheduler(fake_pipeline, tick=0.5)


class TestRegistro:
    def test_register_devuelve_job_listo_para_ejecutar(self, scheduler):
        job = scheduler.register("text://hola", interval=60)
        assert job.source == "text://hola"
        assert job.interval == 60
        assert job.next_run is not None  # primera ejecución inmediata
        assert job.runs == 0

    def test_id_estable_re_registro_actualiza(self, scheduler):
        first = scheduler.register("text://hola", interval=60)
        second = scheduler.register("text://hola", interval=120)
        assert first.id == second.id
        assert scheduler.list() == [second]
        assert second.interval == 120

    def test_intervalo_invalido(self, scheduler):
        with pytest.raises(ValueError, match="intervalo"):
            scheduler.register("text://hola", interval=0.5)

    def test_fuente_vacia(self, scheduler):
        with pytest.raises(ValueError, match="vacía"):
            scheduler.register("   ", interval=60)

    def test_remove(self, scheduler):
        job = scheduler.register("text://hola", interval=60)
        assert scheduler.remove(job.id) is True
        assert scheduler.remove(job.id) is False
        assert scheduler.list() == []

    def test_get(self, scheduler):
        job = scheduler.register("text://hola", interval=60)
        assert scheduler.get(job.id) is not None
        assert scheduler.get("otro") is None


class TestEjecucion:
    def test_run_now_ejecuta_y_actualiza(self, scheduler, fake_pipeline):
        job = scheduler.register("text://hola", interval=60, connector="text", options={"title": "X"})
        outcome = scheduler.run_now(job.id)
        assert outcome["job"]["last_status"] == "ok"
        assert outcome["result"]["documents"] == 1
        assert fake_pipeline.calls == [("text://hola", "text", {"title": "X"})]
        updated = scheduler.get(job.id)
        assert updated is not None
        assert updated.runs == 1
        assert updated.last_run is not None
        assert updated.next_run > updated.last_run  # reprogramado al futuro
        assert updated.last_documents == 1
        assert updated.last_chunks == 2

    def test_run_now_con_fallo_marca_error(self, fake_pipeline):
        scheduler = SyncScheduler(FakePipeline(fail=True), tick=0.5)
        job = scheduler.register("text://malo", interval=60)
        outcome = scheduler.run_now(job.id)
        assert outcome["job"]["last_status"] == "error"
        assert "boom" in (outcome["job"]["last_error"] or "")

    def test_run_now_excepcion_del_pipeline(self):
        class ExplosivePipeline:
            def ingest(self, source, connector=None, **options):
                raise RuntimeError("red caída")

        scheduler = SyncScheduler(ExplosivePipeline(), tick=0.5)
        job = scheduler.register("text://x", interval=60)
        outcome = scheduler.run_now(job.id)
        assert outcome["job"]["last_status"] == "error"
        assert "RuntimeError" in (outcome["job"]["last_error"] or "")
        assert outcome["result"] is None

    def test_run_now_desconocido(self, scheduler):
        with pytest.raises(KeyError):
            scheduler.run_now("fantasma")

    def test_tick_solo_ejecuta_vencidos(self, scheduler, fake_pipeline):
        due = scheduler.register("text://vencido", interval=60)
        future = scheduler.register("text://futuro", interval=3600)
        scheduler.run_now(future.id)  # ejecución manual: queda reprogramado al futuro
        before = len(fake_pipeline.calls)
        results = scheduler.tick()
        assert [r["job"]["id"] for r in results] == [due.id]
        assert [call[0] for call in fake_pipeline.calls[before:]] == ["text://vencido"]

    def test_tick_sin_pendientes(self, scheduler):
        assert scheduler.tick() == []


class TestPersistencia:
    def test_roundtrip_json(self, fake_pipeline, tmp_path):
        state = tmp_path / "sync.json"
        scheduler = SyncScheduler(fake_pipeline, tick=0.5, state_path=state)
        job = scheduler.register("text://hola", interval=600, options={"a": 1})
        assert state.exists()
        restored = SyncScheduler(fake_pipeline, tick=0.5, state_path=state)
        loaded = restored.get(job.id)
        assert loaded is not None
        assert loaded.source == "text://hola"
        assert loaded.interval == 600
        assert loaded.options == {"a": 1}
        assert loaded.next_run is not None  # vence pronto: se ejecutará al primer tick

    def test_estado_corrupto_no_rompe(self, fake_pipeline, tmp_path):
        state = tmp_path / "sync.json"
        state.write_text("{no soy json", encoding="utf-8")
        scheduler = SyncScheduler(fake_pipeline, tick=0.5, state_path=state)
        assert scheduler.list() == []

    def test_json_contenido_legible(self, fake_pipeline, tmp_path):
        state = tmp_path / "sync.json"
        scheduler = SyncScheduler(fake_pipeline, tick=0.5, state_path=state)
        job = scheduler.register("text://hola", interval=600)
        data = json.loads(state.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["jobs"][0]["id"] == job.id


class TestHiloFondo:
    def test_start_stop_ejecuta_jobs(self, fake_pipeline):
        scheduler = SyncScheduler(fake_pipeline, tick=0.5)
        scheduler.register("text://hola", interval=1.0)
        scheduler.start()
        try:
            time.sleep(1.4)
        finally:
            scheduler.stop()
        assert scheduler.executions >= 1
        assert scheduler.stats()["running"] is False

    def test_start_es_idempotente(self, scheduler):
        scheduler.start()
        try:
            thread_first = scheduler._thread
            scheduler.start()
            assert scheduler._thread is thread_first
        finally:
            scheduler.stop()

    def test_stats(self, scheduler):
        scheduler.register("text://hola", interval=60)
        stats = scheduler.stats()
        assert stats["jobs"] == 1
        assert stats["running"] is False
        assert stats["executions"] == 0
