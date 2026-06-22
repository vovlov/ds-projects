"""Tests for Conversational Memory (multi-turn RAG sessions)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.memory.conversation_memory import (
    ConversationMemory,
    ConversationTurn,
    MemoryConfig,
    SessionStats,
)

# ---------------------------------------------------------------------------
# TestConversationMemory
# ---------------------------------------------------------------------------


class TestConversationMemory:
    def setup_method(self):
        self.mem = ConversationMemory()

    def test_create_session_returns_string(self):
        sid = self.mem.create_session()
        assert isinstance(sid, str) and len(sid) > 0

    def test_create_session_unique_ids(self):
        ids = {self.mem.create_session() for _ in range(10)}
        assert len(ids) == 10

    def test_get_or_create_new_session(self):
        sid = "test-session-001"
        session = self.mem.get_or_create_session(sid)
        assert session.session_id == sid

    def test_get_or_create_returns_same(self):
        sid = self.mem.create_session()
        s1 = self.mem.get_or_create_session(sid)
        s2 = self.mem.get_or_create_session(sid)
        assert s1 is s2

    def test_add_turn_returns_conversation_turn(self):
        sid = self.mem.create_session()
        turn = self.mem.add_turn(sid, "Hello?", "Hi there!", ["doc.txt"])
        assert isinstance(turn, ConversationTurn)
        assert turn.question == "Hello?"
        assert turn.answer == "Hi there!"
        assert turn.sources == ["doc.txt"]

    def test_get_history_empty_unknown_session(self):
        history = self.mem.get_history("nonexistent")
        assert history == []

    def test_get_history_with_turns(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "Q1", "A1")
        self.mem.add_turn(sid, "Q2", "A2")
        history = self.mem.get_history(sid)
        assert len(history) == 2
        assert history[0].question == "Q1"
        assert history[1].question == "Q2"

    def test_get_history_last_n(self):
        sid = self.mem.create_session()
        for i in range(5):
            self.mem.add_turn(sid, f"Q{i}", f"A{i}")
        history = self.mem.get_history(sid, last_n=2)
        assert len(history) == 2
        assert history[0].question == "Q3"
        assert history[1].question == "Q4"

    def test_sliding_window_max_turns(self):
        config = MemoryConfig(max_turns=3)
        mem = ConversationMemory(config)
        sid = mem.create_session()
        for i in range(5):
            mem.add_turn(sid, f"Q{i}", f"A{i}")
        history = mem.get_history(sid)
        # Только последние 3 хода
        assert len(history) == 3
        assert history[0].question == "Q2"

    def test_rewrite_query_no_session(self):
        result = self.mem.rewrite_query("unknown-session", "What is this?")
        assert result == "What is this?"

    def test_rewrite_query_no_history(self):
        sid = self.mem.create_session()
        result = self.mem.rewrite_query(sid, "What is the policy?")
        assert result == "What is the policy?"

    def test_rewrite_query_standalone_unchanged(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "What is the vacation policy?", "Employees get 20 days.")
        # Самодостаточный вопрос — не требует контекста
        result = self.mem.rewrite_query(sid, "How many sick days are allowed per year?")
        assert result == "How many sick days are allowed per year?"

    def test_rewrite_query_short_question_adds_context(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "What is the vacation policy?", "Employees get 20 days.")
        # Короткий вопрос (≤5 слов) → добавляем контекст
        result = self.mem.rewrite_query(sid, "When does it start?")
        assert "[Context:" in result
        assert "vacation policy" in result.lower() or "Q:" in result

    def test_rewrite_query_pronoun_adds_context(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "What is the remote work policy?", "Work from home 3 days.")
        # Вопрос с местоимением → добавляем контекст
        result = self.mem.rewrite_query(sid, "Does this apply to contractors?")
        assert "[Context:" in result

    def test_rewrite_query_preserves_original_question(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "Q1", "A1")
        rewritten = self.mem.rewrite_query(sid, "it")
        assert rewritten.startswith("it")

    def test_reset_session_true_if_exists(self):
        sid = self.mem.create_session()
        assert self.mem.reset_session(sid) is True

    def test_reset_session_false_if_not_exists(self):
        assert self.mem.reset_session("ghost-session") is False

    def test_reset_removes_history(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "Q1", "A1")
        self.mem.reset_session(sid)
        assert self.mem.get_history(sid) == []

    def test_list_sessions_active(self):
        sid1 = self.mem.create_session()
        sid2 = self.mem.create_session()
        active = self.mem.list_sessions()
        assert sid1 in active
        assert sid2 in active

    def test_list_sessions_after_reset(self):
        sid = self.mem.create_session()
        self.mem.reset_session(sid)
        assert sid not in self.mem.list_sessions()

    def test_get_session_stats_none_for_unknown(self):
        stats = self.mem.get_session_stats("unknown")
        assert stats is None

    def test_get_session_stats_returns_stats(self):
        sid = self.mem.create_session()
        self.mem.add_turn(sid, "Q1", "A1")
        stats = self.mem.get_session_stats(sid)
        assert isinstance(stats, SessionStats)
        assert stats.session_id == sid
        assert stats.n_turns == 1
        assert not stats.is_expired

    def test_purge_expired_returns_count(self):
        # TTL=0 — все сессии истёкшие сразу
        config = MemoryConfig(ttl_seconds=0.0)
        mem = ConversationMemory(config)
        mem.create_session()
        mem.create_session()
        # Форсируем TTL проверку через purge
        import time

        time.sleep(0.01)
        purged = mem.purge_expired()
        assert purged >= 2

    def test_add_turn_to_unknown_session_creates_it(self):
        turn = self.mem.add_turn("brand-new", "Q?", "A.")
        assert isinstance(turn, ConversationTurn)
        assert len(self.mem.get_history("brand-new")) == 1


# ---------------------------------------------------------------------------
# TestConversationalRAGAPI
# ---------------------------------------------------------------------------


class TestConversationalRAGAPI:
    """Тесты API-эндпоинтов памяти диалога."""

    @pytest.fixture(autouse=True)
    def client(self):
        from fastapi.testclient import TestClient
        from rag.api.app import _reset_cache, _reset_memory, app

        _reset_memory()
        _reset_cache()
        return TestClient(app)

    def test_memory_create_session_200(self, client):
        resp = client.post("/memory/session")
        assert resp.status_code == 200

    def test_memory_create_session_structure(self, client):
        resp = client.post("/memory/session")
        data = resp.json()
        assert "session_id" in data
        assert "max_turns" in data
        assert "ttl_seconds" in data
        assert "context_turns" in data

    def test_memory_create_session_unique(self, client):
        sid1 = client.post("/memory/session").json()["session_id"]
        sid2 = client.post("/memory/session").json()["session_id"]
        assert sid1 != sid2

    def test_memory_history_empty_200(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        resp = client.get(f"/memory/history/{sid}")
        assert resp.status_code == 200

    def test_memory_history_structure(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        resp = client.get(f"/memory/history/{sid}")
        data = resp.json()
        assert "session_id" in data
        assert "turns" in data
        assert "n_turns" in data
        assert data["n_turns"] == 0

    def test_memory_reset_200(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        resp = client.post(f"/memory/reset/{sid}")
        assert resp.status_code == 200

    def test_memory_reset_structure(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        resp = client.post(f"/memory/reset/{sid}")
        data = resp.json()
        assert data["cleared"] is True
        assert data["session_id"] == sid

    def test_memory_reset_unknown_session(self, client):
        resp = client.post("/memory/reset/nonexistent-session-xyz")
        data = resp.json()
        assert data["cleared"] is False

    def test_memory_sessions_200(self, client):
        resp = client.get("/memory/sessions")
        assert resp.status_code == 200

    def test_memory_sessions_structure(self, client):
        resp = client.get("/memory/sessions")
        data = resp.json()
        assert "active_sessions" in data
        assert "count" in data

    def test_memory_session_appears_in_list(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        active = client.get("/memory/sessions").json()["active_sessions"]
        assert sid in active

    def test_query_response_has_session_id_field(self, client):
        resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data

    def test_query_without_session_id_none(self, client):
        resp = client.post("/query", json={"question": "test"})
        assert resp.json()["session_id"] is None

    def test_query_with_session_id_echoed(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        resp = client.post("/query", json={"question": "test", "session_id": sid})
        assert resp.json()["session_id"] == sid

    def test_query_adds_turn_to_history(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        client.post("/query", json={"question": "What is the policy?", "session_id": sid})
        history = client.get(f"/memory/history/{sid}").json()
        assert history["n_turns"] == 1
        assert history["turns"][0]["question"] == "What is the policy?"

    def test_query_multiple_turns_stored(self, client):
        sid = client.post("/memory/session").json()["session_id"]
        client.post("/query", json={"question": "Q1", "session_id": sid})
        client.post("/query", json={"question": "Q2", "session_id": sid})
        history = client.get(f"/memory/history/{sid}").json()
        assert history["n_turns"] == 2

    def test_memory_custom_config(self, client):
        resp = client.post(
            "/memory/session",
            json={"max_turns": 5, "ttl_seconds": 7200.0, "context_turns": 2},
        )
        data = resp.json()
        assert data["max_turns"] == 5
        assert data["ttl_seconds"] == 7200.0
        assert data["context_turns"] == 2
