"""Memoria conversacional multi-turno (Fase 7).

Guarda el historial de preguntas y respuestas por ``conversation_id`` para que
las consultas de seguimiento ("¿y en Colombia?") tengan contexto. El pipeline
inyecta el historial en el generador como mensajes de chat (``user``/``assistant``).

* En memoria (por proceso / pipeline), thread-safe.
* LRU: se descarta la conversación menos usada al superar ``max_conversations``
  y el turno más viejo al superar ``max_turns``.
* Interacción con la caché semántica: las consultas con ``conversation_id``
  la omiten (la respuesta depende del historial, no solo de la pregunta).
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from typing import Any

from universal_rag.core.utils import get_logger, utcnow

logger = get_logger("memory")


class ConversationMemory:
    """Historial de conversaciones con límites LRU (turnos y conversaciones)."""

    def __init__(self, max_turns: int = 8, max_conversations: int = 200) -> None:
        self.max_turns = max(1, int(max_turns))
        self.max_conversations = max(1, int(max_conversations))
        self._lock = threading.Lock()
        # conversation_id -> deque de turnos {"question", "answer", "at"}
        self._conversations: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()
        self._appends = 0

    # ------------------------------------------------------------------ API
    def append(self, conversation_id: str, question: str, answer: str) -> None:
        """Registra un turno (pregunta + respuesta) en la conversación."""
        cid = (conversation_id or "").strip()
        if not cid:
            return
        turn = {"question": question, "answer": answer, "at": utcnow().isoformat()}
        with self._lock:
            self._appends += 1
            turns = self._conversations.get(cid)
            if turns is None:
                turns = deque()
                self._conversations[cid] = turns
            self._conversations.move_to_end(cid)
            turns.append(turn)
            while len(turns) > self.max_turns:
                turns.popleft()
            while len(self._conversations) > self.max_conversations:
                evicted, _ = self._conversations.popitem(last=False)
                logger.debug("Conversación desalojada por LRU: %s", evicted)

    def history(self, conversation_id: str) -> list[dict[str, str]]:
        """Mensajes en formato chat ``{"role", "content"}`` del más viejo al más nuevo."""
        cid = (conversation_id or "").strip()
        if not cid:
            return []
        with self._lock:
            turns = self._conversations.get(cid)
            if not turns:
                return []
            self._conversations.move_to_end(cid)  # acceso refresca el LRU
            messages: list[dict[str, str]] = []
            for turn in turns:
                messages.append({"role": "user", "content": str(turn["question"])})
                messages.append({"role": "assistant", "content": str(turn["answer"])})
            return messages

    def turns(self, conversation_id: str) -> int:
        """Número de turnos recordados de una conversación."""
        cid = (conversation_id or "").strip()
        with self._lock:
            turns = self._conversations.get(cid)
            return len(turns) if turns else 0

    def conversations(self) -> list[dict[str, Any]]:
        """Lista de conversaciones vivas: ``[{"id", "turns", "last_at"}]``."""
        with self._lock:
            return [
                {"id": cid, "turns": len(turns), "last_at": turns[-1]["at"] if turns else None}
                for cid, turns in self._conversations.items()
            ]

    def clear(self, conversation_id: str | None = None) -> int:
        """Borra una conversación (o todas); devuelve cuántas se eliminaron."""
        with self._lock:
            if conversation_id:
                cid = conversation_id.strip()
                if cid in self._conversations:
                    del self._conversations[cid]
                    return 1
                return 0
            cleared = len(self._conversations)
            self._conversations.clear()
            return cleared

    def stats(self) -> dict[str, Any]:
        """Resumen para ``/stats``."""
        with self._lock:
            total_turns = sum(len(turns) for turns in self._conversations.values())
            return {
                "enabled": True,
                "conversations": len(self._conversations),
                "turns": total_turns,
                "appends": self._appends,
                "max_turns": self.max_turns,
                "max_conversations": self.max_conversations,
            }
