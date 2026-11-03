"""Tests de la memoria conversacional multi-turno (Fase 7)."""

from __future__ import annotations

from universal_rag.core.memory import ConversationMemory


class TestAppendYHistory:
    def test_history_roles_alternados(self):
        m = ConversationMemory()
        m.append("c1", "¿qué es RAG?", "RAG es búsqueda aumentada.")
        m.append("c1", "¿y en Colombia?", "Se usa en banca.")
        history = m.history("c1")
        assert [msg["role"] for msg in history] == ["user", "assistant", "user", "assistant"]
        assert history[0]["content"] == "¿qué es RAG?"
        assert history[1]["content"] == "RAG es búsqueda aumentada."
        assert history[3]["content"] == "Se usa en banca."

    def test_conversacion_inexistente(self):
        m = ConversationMemory()
        assert m.history("nope") == []
        assert m.turns("nope") == 0

    def test_id_vacio_se_ignora(self):
        m = ConversationMemory()
        m.append("", "q", "a")
        m.append("   ", "q", "a")
        assert m.conversations() == []
        assert m.stats()["conversations"] == 0

    def test_max_turns_recorta_los_mas_viejos(self):
        m = ConversationMemory(max_turns=2)
        for i in range(5):
            m.append("c1", f"p{i}", f"r{i}")
        history = m.history("c1")
        assert history[0]["content"] == "p3"
        assert len(history) == 4  # 2 turnos × 2 mensajes
        assert m.turns("c1") == 2

    def test_max_conversations_lru(self):
        m = ConversationMemory(max_conversations=2)
        m.append("a", "q", "r")
        m.append("b", "q", "r")
        m.append("c", "q", "r")
        ids = {row["id"] for row in m.conversations()}
        assert ids == {"b", "c"}  # 'a' era la menos reciente

    def test_acceso_refresca_lru(self):
        m = ConversationMemory(max_conversations=2)
        m.append("a", "q", "r")
        m.append("b", "q", "r")
        m.history("a")  # tocar 'a' la marca como reciente
        m.append("c", "q", "r")
        ids = {row["id"] for row in m.conversations()}
        assert "a" in ids and "b" not in ids

    def test_conversations_incluye_contadores(self):
        m = ConversationMemory()
        m.append("c1", "q1", "r1")
        m.append("c1", "q2", "r2")
        rows = m.conversations()
        assert len(rows) == 1
        assert rows[0]["id"] == "c1"
        assert rows[0]["turns"] == 2
        assert rows[0]["last_at"] is not None


class TestClearYStats:
    def test_clear_una(self):
        m = ConversationMemory()
        m.append("a", "q", "r")
        m.append("b", "q", "r")
        assert m.clear("a") == 1
        assert m.turns("a") == 0
        assert m.turns("b") == 1

    def test_clear_inexistente_devuelve_cero(self):
        assert ConversationMemory().clear("fantasma") == 0

    def test_clear_todas(self):
        m = ConversationMemory()
        m.append("a", "q", "r")
        m.append("b", "q", "r")
        assert m.clear() == 2
        assert m.conversations() == []

    def test_stats_refleja_estado(self):
        m = ConversationMemory(max_turns=3, max_conversations=7)
        m.append("c1", "q", "r")
        m.append("c1", "q2", "r2")
        stats = m.stats()
        assert stats["enabled"] is True
        assert stats["conversations"] == 1
        assert stats["turns"] == 2
        assert stats["max_turns"] == 3
        assert stats["max_conversations"] == 7
        assert stats["appends"] == 2
