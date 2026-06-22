"""Conversational memory for multi-turn RAG sessions.

Реализует паттерн скользящего окна (Lehman 2026, LogRocket):
- max_turns хранит только последние N ходов диалога
- TTL expiration освобождает неактивные сессии
- Query rewriting превращает follow-up вопросы в standalone retrieval-запросы:
  "когда это применяется?" → "когда применяется политика удалённой работы?"
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ConversationTurn:
    """Один ход диалога: вопрос пользователя + ответ системы."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class MemoryConfig:
    """Параметры памяти сессии диалога."""

    max_turns: int = 10
    ttl_seconds: float = 3600.0
    context_turns: int = 3  # число предыдущих ходов для query rewriting


@dataclass
class SessionStats:
    """Статистика сессии диалога для /memory/sessions."""

    session_id: str
    n_turns: int
    created_at: str
    last_active: str
    is_expired: bool


class ConversationSession:
    """Сессия диалога с историей ходов (скользящее окно)."""

    def __init__(self, session_id: str, config: MemoryConfig) -> None:
        self.session_id = session_id
        self.config = config
        self.turns: deque[ConversationTurn] = deque(maxlen=config.max_turns)
        self.created_at: datetime = datetime.now(UTC)
        self.last_active: datetime = self.created_at

    def add_turn(
        self, question: str, answer: str, sources: list[str] | None = None
    ) -> ConversationTurn:
        """Добавить ход диалога, обновить last_active."""
        turn = ConversationTurn(question=question, answer=answer, sources=sources or [])
        self.turns.append(turn)
        self.last_active = datetime.now(UTC)
        return turn

    def get_history(self, last_n: int | None = None) -> list[ConversationTurn]:
        """Получить историю ходов, опционально ограниченную last_n."""
        turns = list(self.turns)
        if last_n is not None:
            turns = turns[-last_n:]
        return turns

    def is_expired(self, ttl_seconds: float) -> bool:
        elapsed = (datetime.now(UTC) - self.last_active).total_seconds()
        return elapsed > ttl_seconds

    def stats(self, ttl_seconds: float | None = None) -> SessionStats:
        _ttl = ttl_seconds if ttl_seconds is not None else self.config.ttl_seconds
        return SessionStats(
            session_id=self.session_id,
            n_turns=len(self.turns),
            created_at=self.created_at.isoformat(),
            last_active=self.last_active.isoformat(),
            is_expired=self.is_expired(_ttl),
        )


class ConversationMemory:
    """Менеджер памяти для multi-turn RAG диалогов.

    Ключевые возможности:
    - Скользящее окно ходов (max_turns) — ограничение памяти
    - TTL expiration — автоматическое освобождение неактивных сессий
    - Query rewriting — follow-up вопросы превращаются в standalone retrieval-запросы

    Источники:
    - Rackauckas 2024 (Conversational RAG, arxiv:2402.03367)
    - LogRocket 2026 "LLM context problem: sliding window"
    - Confident AI 2026 "Multi-turn LLM evaluation"
    """

    # Слова-ссылки, указывающие что вопрос зависит от предыдущего контекста
    _CONTEXT_SIGNALS: frozenset[str] = frozenset(
        [
            "it",
            "this",
            "that",
            "they",
            "them",
            "their",
            "these",
            "those",
            "such",
            "same",
            "above",
            "mentioned",
            "described",
            "said",
            "previous",
            "aforementioned",
        ]
    )

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self.config = config or MemoryConfig()
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self) -> str:
        """Создать новую сессию. Возвращает session_id (UUID4)."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = ConversationSession(session_id, self.config)
        return session_id

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Получить существующую сессию или создать новую.

        Истёкшие сессии (TTL) автоматически сбрасываются.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id, self.config)
        session = self._sessions[session_id]
        if session.is_expired(self.config.ttl_seconds):
            # Сбрасываем истёкшую сессию вместо отказа — seamless UX
            self._sessions[session_id] = ConversationSession(session_id, self.config)
        return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[str] | None = None,
    ) -> ConversationTurn:
        """Записать ход диалога в сессию."""
        session = self.get_or_create_session(session_id)
        return session.add_turn(question, answer, sources)

    def get_history(self, session_id: str, last_n: int | None = None) -> list[ConversationTurn]:
        """Получить историю ходов сессии (или пустой список если нет сессии)."""
        if session_id not in self._sessions:
            return []
        return self._sessions[session_id].get_history(last_n)

    def rewrite_query(self, session_id: str, question: str) -> str:
        """Переписать follow-up вопрос в standalone поисковый запрос.

        Алгоритм:
        1. Если нет истории → запрос без изменений.
        2. Если вопрос короткий (≤5 слов) или содержит слова-ссылки
           (it / this / that / they ...) → добавляем контекст предыдущих ходов.
        3. Иначе → запрос самодостаточен, изменений нет.

        Это позволяет retrieval-компоненту найти релевантные чанки
        без знания истории диалога.

        Примеры:
          "когда это применяется?" → "когда это применяется? [Context: Q: ... A: ...]"
          "What is the vacation policy?" → без изменений (самодостаточен)
        """
        if session_id not in self._sessions:
            return question

        session = self._sessions[session_id]
        if not session.turns:
            return question

        words = set(question.lower().split())
        is_short = len(question.split()) <= 5
        has_reference = bool(words & self._CONTEXT_SIGNALS)

        if not is_short and not has_reference:
            return question

        recent = session.get_history(last_n=self.config.context_turns)
        parts: list[str] = []
        for turn in recent:
            parts.append(f"Q: {turn.question}")
            if turn.answer:
                # Первые 120 символов первого предложения — достаточно для контекста
                preview = turn.answer.split(".")[0].strip()[:120]
                parts.append(f"A: {preview}")

        context_str = " | ".join(parts)
        return f"{question} [Context: {context_str}]"

    def reset_session(self, session_id: str) -> bool:
        """Удалить историю сессии. Возвращает True если сессия существовала."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[str]:
        """Список активных (не истёкших) session_id."""
        return [
            sid for sid, s in self._sessions.items() if not s.is_expired(self.config.ttl_seconds)
        ]

    def get_session_stats(self, session_id: str) -> SessionStats | None:
        """Статистика конкретной сессии или None если не найдена."""
        if session_id not in self._sessions:
            return None
        return self._sessions[session_id].stats()

    def purge_expired(self) -> int:
        """Удалить все истёкшие сессии. Возвращает число удалённых."""
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired(self.config.ttl_seconds)
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)
