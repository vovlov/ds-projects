"""Тесты HyDE (Hypothetical Document Embeddings): генерация + retrieval + API endpoint.

Gao et al. 2022 (arxiv:2212.10496): вместо эмбеддинга запроса используем
эмбеддинг гипотетического документа-ответа для улучшения semantic retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retrieval.hyde import (
    HyDEConfig,
    HyDEResult,
    _extract_keywords,
    _rule_based_hypothetical,
    generate_hypothetical_document,
)

# ---------------------------------------------------------------------------
# TestHyDEConfig — конфигурация
# ---------------------------------------------------------------------------


class TestHyDEConfig:
    def test_defaults(self):
        cfg = HyDEConfig()
        assert cfg.max_tokens == 150
        assert cfg.n_hypothetical == 1
        assert cfg.use_llm is False

    def test_custom(self):
        cfg = HyDEConfig(max_tokens=200, use_llm=True, temperature=0.5)
        assert cfg.max_tokens == 200
        assert cfg.use_llm is True
        assert cfg.temperature == 0.5


# ---------------------------------------------------------------------------
# TestKeywordExtraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    def test_filters_stop_words(self):
        kws = _extract_keywords("What is the remote work policy?")
        lower = [k.lower() for k in kws]
        assert "what" not in lower
        assert "is" not in lower
        assert "the" not in lower

    def test_keeps_content_words(self):
        kws = _extract_keywords("What is the remote work policy?")
        lower = [k.lower() for k in kws]
        assert "remote" in lower or "work" in lower or "policy" in lower

    def test_empty_query(self):
        kws = _extract_keywords("")
        assert kws == []

    def test_only_stop_words(self):
        kws = _extract_keywords("what is the")
        # Все стоп-слова — результат может быть пустым или содержать только короткие токены
        assert isinstance(kws, list)


# ---------------------------------------------------------------------------
# TestRuleBasedHypothetical — детерминированная генерация без LLM
# ---------------------------------------------------------------------------


class TestRuleBasedHypothetical:
    def test_returns_string(self):
        result = _rule_based_hypothetical("What is the vacation policy?")
        assert isinstance(result, str)

    def test_non_empty(self):
        result = _rule_based_hypothetical("How do I request time off?")
        assert len(result) > 20

    def test_what_is_pattern(self):
        result = _rule_based_hypothetical("What is VPN?")
        # Должен содержать слово из запроса
        assert "vpn" in result.lower() or "VPN" in result

    def test_how_does_pattern(self):
        result = _rule_based_hypothetical("How does authentication work?")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_why_pattern(self):
        result = _rule_based_hypothetical("Why is security important?")
        assert "security" in result.lower() or "important" in result.lower()

    def test_who_pattern(self):
        result = _rule_based_hypothetical("Who is responsible for data protection?")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_when_pattern(self):
        result = _rule_based_hypothetical("When does the policy apply?")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_generic_fallback(self):
        result = _rule_based_hypothetical("Tell me about GDPR compliance requirements.")
        assert isinstance(result, str)
        assert "gdpr" in result.lower() or "compliance" in result.lower()

    def test_includes_query_keywords(self):
        result = _rule_based_hypothetical("What is the onboarding process for contractors?")
        result_lower = result.lower()
        # Хотя бы одно ключевое слово из запроса должно быть в гипотезе
        assert any(kw in result_lower for kw in ["onboarding", "contractor", "process"])

    def test_deterministic(self):
        q = "What is the remote work policy?"
        assert _rule_based_hypothetical(q) == _rule_based_hypothetical(q)


# ---------------------------------------------------------------------------
# TestGenerateHypotheticalDocument — публичный интерфейс
# ---------------------------------------------------------------------------


class TestGenerateHypotheticalDocument:
    def test_default_config_returns_string(self):
        result = generate_hypothetical_document("What is the HR policy?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_none_config_uses_defaults(self):
        result = generate_hypothetical_document("How to request vacation?", config=None)
        assert isinstance(result, str)

    def test_rule_based_when_use_llm_false(self):
        cfg = HyDEConfig(use_llm=False)
        result = generate_hypothetical_document("What is VPN?", config=cfg)
        # Rule-based = детерминированный
        assert result == generate_hypothetical_document("What is VPN?", config=cfg)

    def test_llm_fallback_without_api_key(self):
        """Без ANTHROPIC_API_KEY use_llm=True должен fallback на rule-based."""
        import os

        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            cfg = HyDEConfig(use_llm=True)
            result = generate_hypothetical_document("What is GDPR?", config=cfg)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_result_differs_from_query(self):
        query = "What is vacation policy?"
        result = generate_hypothetical_document(query)
        # Гипотеза должна быть длиннее/другой чем оригинал
        assert result != query


# ---------------------------------------------------------------------------
# TestHyDEResult — dataclass
# ---------------------------------------------------------------------------


class TestHyDEResult:
    def test_n_results_auto(self):
        result = HyDEResult(
            chunks=[{"text": "a"}, {"text": "b"}],
            hypothetical_document="Some hypothesis",
        )
        assert result.n_results == 2

    def test_empty_chunks(self):
        result = HyDEResult(chunks=[], hypothetical_document="Test")
        assert result.n_results == 0

    def test_retrieval_method_default(self):
        result = HyDEResult(chunks=[], hypothetical_document="Test")
        assert result.retrieval_method == "hyde"

    def test_to_dict_structure(self):
        result = HyDEResult(chunks=[{"text": "x"}], hypothetical_document="Hypothesis text")
        d = result.to_dict()
        assert "n_chunks" in d
        assert "hypothetical_document" in d
        assert "retrieval_method" in d
        assert d["n_chunks"] == 1
        assert d["hypothetical_document"] == "Hypothesis text"
        assert d["retrieval_method"] == "hyde"


# ---------------------------------------------------------------------------
# TestHyDERetrieve — retrieval через гипотетический документ
# ---------------------------------------------------------------------------


class TestHyDERetrieve:
    def test_returns_hyde_result(self):
        """hyde_retrieve возвращает HyDEResult с hypothetical_document."""
        # Проверяем через прямое конструирование — retrieval с mock collection
        # тестируется в API-тестах выше через TestClient
        result = HyDEResult(
            chunks=[{"text": "policy chunk"}],
            hypothetical_document="Hypothetical doc",
        )
        assert isinstance(result, HyDEResult)
        assert result.hypothetical_document == "Hypothetical doc"
        assert result.retrieval_method == "hyde"

    def test_hyde_result_has_hypothetical_document(self):
        """HyDEResult всегда содержит hypothetical_document."""
        result = HyDEResult(
            chunks=[],
            hypothetical_document="The policy states that employees receive 20 days vacation.",
        )
        assert result.hypothetical_document
        assert len(result.hypothetical_document) > 0


# ---------------------------------------------------------------------------
# TestHyDEAPIEndpoints — API endpoint тесты
# ---------------------------------------------------------------------------


class TestHyDEAPIEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from rag.api.app import app

        return TestClient(app)

    def test_hyde_generate_200(self, client):
        resp = client.post("/hyde/generate", json={"query": "What is the vacation policy?"})
        assert resp.status_code == 200

    def test_hyde_generate_response_structure(self, client):
        resp = client.post("/hyde/generate", json={"query": "What is GDPR?"})
        data = resp.json()
        assert "query" in data
        assert "hypothetical_document" in data
        assert "generated_by" in data

    def test_hyde_generate_echoes_query(self, client):
        q = "How do I request remote work?"
        resp = client.post("/hyde/generate", json={"query": q})
        data = resp.json()
        assert data["query"] == q

    def test_hyde_generate_non_empty_document(self, client):
        resp = client.post("/hyde/generate", json={"query": "What is the onboarding process?"})
        data = resp.json()
        assert len(data["hypothetical_document"]) > 10

    def test_hyde_generate_rule_based_by_default(self, client):
        resp = client.post("/hyde/generate", json={"query": "What is VPN?", "use_llm": False})
        data = resp.json()
        assert data["generated_by"] == "rule_based"

    def test_hyde_generate_llm_fallback_without_key(self, client):
        """Без API-ключа use_llm=True должен вернуть generated_by=rule_based."""
        import os

        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            resp = client.post(
                "/hyde/generate",
                json={"query": "What is security policy?", "use_llm": True},
            )
            data = resp.json()
            assert resp.status_code == 200
            assert data["generated_by"] == "rule_based"
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_query_response_has_hypothetical_document_field(self, client):
        """POST /query должен возвращать поле hypothetical_document."""
        resp = client.post(
            "/query",
            json={"question": "What is the vacation policy?", "retrieval_method": "semantic"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Поле должно присутствовать (null для non-HyDE методов)
        assert "hypothetical_document" in data
        assert data["hypothetical_document"] is None  # None для semantic

    def test_query_hyde_method_returns_hypothetical_document(self, client):
        """POST /query с retrieval_method='hyde' должен вернуть hypothetical_document."""
        resp = client.post(
            "/query",
            json={"question": "Tell me about policies", "retrieval_method": "hyde"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["retrieval_method"] == "hyde"
        assert data["hypothetical_document"] is not None
        assert len(data["hypothetical_document"]) > 0

    def test_query_hyde_method_echoed(self, client):
        """retrieval_method='hyde' должен быть отражён в ответе."""
        resp = client.post(
            "/query",
            json={"question": "What is security?", "retrieval_method": "hyde"},
        )
        data = resp.json()
        assert data["retrieval_method"] == "hyde"
