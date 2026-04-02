"""Tests for RAG Enterprise pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import os

from rag.generation.chain import build_prompt
from rag.ingestion.loader import chunk_documents, load_documents
from rag.retrieval.store import get_client, get_or_create_collection, index_chunks, search

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"


class TestIngestion:
    def test_load_documents(self):
        docs = load_documents(DATA_DIR)
        assert len(docs) >= 2
        assert all("text" in d and "metadata" in d for d in docs)

    def test_load_documents_have_source(self):
        docs = load_documents(DATA_DIR)
        for doc in docs:
            assert "source" in doc["metadata"]
            assert doc["metadata"]["source"].endswith(".txt")

    def test_chunk_documents_default(self):
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        assert len(chunks) > len(docs)  # Should produce more chunks than docs
        assert all("text" in c for c in chunks)

    def test_chunk_documents_preserves_metadata(self):
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "source" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
            assert "chunk_total" in chunk["metadata"]

    def test_chunk_size_respected(self):
        docs = load_documents(DATA_DIR)
        chunk_size = 256
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=32)
        for chunk in chunks:
            # Allow some tolerance for splitting
            assert len(chunk["text"]) <= chunk_size + 50


class TestRetrieval:
    @pytest.fixture
    def indexed_collection(self):
        client = get_client(Path("/tmp/test_chroma_rag"))
        with contextlib.suppress(Exception):
            client.delete_collection("test_docs")
        collection = get_or_create_collection(client, name="test_docs")
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        index_chunks(chunks, collection)
        return collection

    def test_index_chunks(self, indexed_collection):
        assert indexed_collection.count() > 0

    def test_search_returns_results(self, indexed_collection):
        results = search("remote work policy", indexed_collection, n_results=3)
        assert len(results) == 3
        assert all("text" in r for r in results)

    def test_search_relevance(self, indexed_collection):
        results = search("VPN security requirements", indexed_collection, n_results=3)
        # At least one result should mention VPN or security
        texts = " ".join(r["text"].lower() for r in results)
        assert "vpn" in texts or "security" in texts

    def test_search_onboarding(self, indexed_collection):
        results = search("first week onboarding schedule", indexed_collection, n_results=3)
        texts = " ".join(r["text"].lower() for r in results)
        assert "week" in texts or "onboarding" in texts or "day" in texts

    def test_search_code_review(self, indexed_collection):
        results = search("code review process", indexed_collection, n_results=3)
        texts = " ".join(r["text"].lower() for r in results)
        assert "review" in texts or "code" in texts

    def test_search_data_governance(self, indexed_collection):
        results = search("GDPR data retention policy", indexed_collection, n_results=3)
        texts = " ".join(r["text"].lower() for r in results)
        assert "gdpr" in texts or "retention" in texts or "data" in texts


class TestGeneration:
    def test_build_prompt_includes_context(self):
        chunks = [
            {"text": "Policy text here", "metadata": {"source": "policy.txt"}},
            {"text": "Another doc", "metadata": {"source": "guide.txt"}},
        ]
        prompt = build_prompt("What is the policy?", chunks)
        assert "Policy text here" in prompt
        assert "Another doc" in prompt
        assert "What is the policy?" in prompt

    def test_build_prompt_includes_sources(self):
        chunks = [{"text": "Some text", "metadata": {"source": "doc.txt"}}]
        prompt = build_prompt("Question?", chunks)
        assert "doc.txt" in prompt

    def test_generate_answer_without_api_key(self):
        """Without ANTHROPIC_API_KEY, should return error message, not crash."""
        import os

        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from rag.generation.chain import generate_answer

            result = generate_answer("Test?", [{"text": "ctx", "metadata": {"source": "t"}}])
            assert "Error" in result["answer"] or "ANTHROPIC_API_KEY" in result["answer"]
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from rag.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


class TestEvaluation:
    """Тесты RAGAS-метрик для оценки качества RAG-ответов.

    Все тесты работают без внешних API — используются лексические метрики.
    """

    # Тестовый сэмпл: вопрос о VPN-политике
    QUESTION = "What are the VPN requirements for remote work?"
    CONTEXTS = [
        "All employees must use company VPN when working remotely."
        " VPN is required for accessing internal systems.",
        "Remote work policy: employees may work from home up to 3 days"
        " per week with manager approval.",
        "Security requirements include two-factor authentication"
        " and VPN usage for all remote connections.",
    ]
    ANSWER = (
        "According to the policy, all employees must use VPN when working remotely. "
        "VPN is required for accessing internal systems and remote connections."
    )
    IRRELEVANT_ANSWER = "The sky is blue and water is wet."

    def test_context_precision_high_for_relevant_query(self):
        from rag.evaluation.ragas_eval import context_precision

        score = context_precision(self.QUESTION, self.CONTEXTS)
        # Все 3 чанка содержат слова VPN/remote — ожидаем высокую точность
        assert score >= 0.8, f"Expected precision ≥ 0.8, got {score}"

    def test_context_precision_zero_for_empty_contexts(self):
        from rag.evaluation.ragas_eval import context_precision

        assert context_precision(self.QUESTION, []) == 0.0

    def test_context_precision_partial(self):
        from rag.evaluation.ragas_eval import context_precision

        mixed_contexts = [
            "VPN is required for remote work.",  # relevant
            "The cafeteria serves lunch at noon.",  # irrelevant
        ]
        score = context_precision("VPN requirements", mixed_contexts)
        # 1 из 2 релевантен → 0.5
        assert 0.3 <= score <= 0.7

    def test_context_recall_high_when_answer_matches_context(self):
        from rag.evaluation.ragas_eval import context_recall

        score = context_recall(self.ANSWER, self.CONTEXTS)
        # Ответ написан на основе контекста → recall должен быть высоким
        assert score >= 0.7, f"Expected recall ≥ 0.7, got {score}"

    def test_context_recall_zero_for_empty_inputs(self):
        from rag.evaluation.ragas_eval import context_recall

        assert context_recall("", self.CONTEXTS) == 0.0
        assert context_recall(self.ANSWER, []) == 0.0

    def test_answer_relevance_high_for_on_topic_answer(self):
        from rag.evaluation.ragas_eval import answer_relevance

        score = answer_relevance(self.QUESTION, self.ANSWER)
        assert score > 0.0, "On-topic answer should have positive relevance"

    def test_answer_relevance_lower_for_off_topic(self):
        from rag.evaluation.ragas_eval import answer_relevance

        on_topic = answer_relevance(self.QUESTION, self.ANSWER)
        off_topic = answer_relevance(self.QUESTION, self.IRRELEVANT_ANSWER)
        assert on_topic > off_topic, "On-topic answer should score higher than irrelevant one"

    def test_faithfulness_high_for_grounded_answer(self):
        from rag.evaluation.ragas_eval import faithfulness

        score = faithfulness(self.ANSWER, self.CONTEXTS)
        assert score >= 0.5, f"Expected faithfulness ≥ 0.5, got {score}"

    def test_evaluate_sample_returns_ragas_result(self):
        from rag.evaluation.ragas_eval import RAGASResult, evaluate_sample

        result = evaluate_sample(self.QUESTION, self.ANSWER, self.CONTEXTS)
        assert isinstance(result, RAGASResult)
        assert 0.0 <= result.context_precision <= 1.0
        assert 0.0 <= result.context_recall <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.overall <= 1.0

    def test_evaluate_sample_as_dict(self):
        from rag.evaluation.ragas_eval import evaluate_sample

        result = evaluate_sample(self.QUESTION, self.ANSWER, self.CONTEXTS)
        d = result.as_dict()
        assert set(d.keys()) == {
            "context_precision",
            "context_recall",
            "answer_relevance",
            "faithfulness",
            "overall",
        }
        assert all(isinstance(v, float) for v in d.values())

    def test_evaluate_sample_with_metadata(self):
        from rag.evaluation.ragas_eval import evaluate_sample

        result = evaluate_sample(
            self.QUESTION,
            self.ANSWER,
            self.CONTEXTS,
            metadata={"sample_id": "test-001", "model": "claude"},
        )
        assert result.metadata["sample_id"] == "test-001"

    def test_evaluate_dataset_returns_averages(self):
        from rag.evaluation.ragas_eval import evaluate_dataset

        samples = [
            {"question": self.QUESTION, "answer": self.ANSWER, "contexts": self.CONTEXTS},
            {
                "question": "What is the remote work policy?",
                "answer": "Employees may work from home up to 3 days per week.",
                "contexts": self.CONTEXTS,
            },
        ]
        metrics = evaluate_dataset(samples)
        assert metrics["num_samples"] == 2
        assert 0.0 <= metrics["overall"] <= 1.0
        required_keys = ["context_precision", "context_recall", "answer_relevance", "faithfulness"]
        assert all(k in metrics for k in required_keys)

    def test_evaluate_dataset_empty(self):
        from rag.evaluation.ragas_eval import evaluate_dataset

        metrics = evaluate_dataset([])
        assert metrics["num_samples"] == 0
        assert metrics["overall"] == 0.0

    def test_overall_score_is_mean_of_four_metrics(self):
        from rag.evaluation.ragas_eval import RAGASResult

        r = RAGASResult(
            context_precision=0.8,
            context_recall=0.6,
            answer_relevance=0.4,
            faithfulness=0.6,
        )
        assert abs(r.overall - 0.6) < 1e-9


class TestFaithfulnessGate:
    """Тесты Agentic RAG faithfulness gate.

    Все тесты работают в lexical-режиме (без ANTHROPIC_API_KEY).
    Проверяют корректность проверки верности ответа retrieved-чанкам.
    """

    ANSWER_GROUNDED = (
        "All employees must use VPN when working remotely. "
        "VPN is required for accessing internal systems and remote connections."
    )
    # Полностью нерелевантный ответ — нет общих слов с контекстом о VPN/политиках
    ANSWER_HALLUCINATED = (
        "The capital of France is Paris. "
        "The Eiffel Tower was constructed in 1889 and stands 330 meters tall."
    )
    CONTEXTS = [
        "All employees must use company VPN when working remotely."
        " VPN is required for accessing internal systems.",
        "Remote work policy: employees may work from home up to 3 days per week.",
        "Security requirements include two-factor authentication and VPN usage.",
    ]

    def test_check_faithfulness_returns_result(self):
        from rag.generation.faithfulness_gate import FaithfulnessResult, check_faithfulness

        result = check_faithfulness(self.ANSWER_GROUNDED, self.CONTEXTS)
        assert isinstance(result, FaithfulnessResult)
        assert 0.0 <= result.score <= 1.0
        assert result.verdict in ("FAITHFUL", "UNFAITHFUL")
        assert result.method == "lexical"  # нет API-ключа в CI

    def test_grounded_answer_scores_higher(self):
        from rag.generation.faithfulness_gate import check_faithfulness

        grounded = check_faithfulness(self.ANSWER_GROUNDED, self.CONTEXTS)
        hallucinated = check_faithfulness(self.ANSWER_HALLUCINATED, self.CONTEXTS)
        assert grounded.score > hallucinated.score, (
            f"Grounded answer should score higher: {grounded.score} vs {hallucinated.score}"
        )

    def test_empty_answer_returns_unfaithful(self):
        from rag.generation.faithfulness_gate import check_faithfulness

        result = check_faithfulness("", self.CONTEXTS)
        assert result.is_faithful is False
        assert result.score == 0.0

    def test_empty_contexts_returns_unfaithful(self):
        from rag.generation.faithfulness_gate import check_faithfulness

        result = check_faithfulness(self.ANSWER_GROUNDED, [])
        assert result.is_faithful is False
        assert result.score == 0.0

    def test_threshold_controls_is_faithful(self):
        from rag.generation.faithfulness_gate import check_faithfulness

        check_faithfulness(self.ANSWER_GROUNDED, self.CONTEXTS, threshold=0.99)
        result_lenient = check_faithfulness(self.ANSWER_GROUNDED, self.CONTEXTS, threshold=0.0)
        # С нулевым порогом — всегда проходит
        assert result_lenient.is_faithful is True

    def test_parse_judge_response_valid(self):
        from rag.generation.faithfulness_gate import _parse_judge_response

        raw = "VERDICT: FAITHFUL\nSCORE: 0.92\nREASON: All claims are supported."
        result = _parse_judge_response(raw, threshold=0.5)
        assert result.verdict == "FAITHFUL"
        assert result.score == 0.92
        assert result.is_faithful is True
        assert result.method == "llm"
        assert "supported" in result.reason.lower()

    def test_parse_judge_response_unfaithful(self):
        from rag.generation.faithfulness_gate import _parse_judge_response

        raw = "VERDICT: UNFAITHFUL\nSCORE: 0.15\nREASON: Bonus claim not in context."
        result = _parse_judge_response(raw, threshold=0.5)
        assert result.verdict == "UNFAITHFUL"
        assert result.score == 0.15
        assert result.is_faithful is False

    def test_parse_judge_response_malformed_falls_back_safe(self):
        from rag.generation.faithfulness_gate import _parse_judge_response

        # Консервативный фallback: при ошибке парсинга → unfaithful, score=0
        result = _parse_judge_response("This is not parseable at all", threshold=0.5)
        assert result.verdict == "UNFAITHFUL"
        assert result.score == 0.0
        assert result.is_faithful is False

    def test_parse_judge_response_score_clamped(self):
        from rag.generation.faithfulness_gate import _parse_judge_response

        raw = "VERDICT: FAITHFUL\nSCORE: 1.99\nREASON: Over limit."
        result = _parse_judge_response(raw, threshold=0.5)
        assert result.score <= 1.0

    def test_generate_answer_with_gate_lexical_mode(self):
        """generate_answer_with_gate возвращает confidence_score без API-ключа."""
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from rag.generation.chain import generate_answer_with_gate

            chunks = [
                {"text": "VPN is required for remote work.", "metadata": {"source": "policy.txt"}},
                {"text": "Employees may work 3 days from home.", "metadata": {"source": "hr.txt"}},
            ]
            # generate_answer вернёт ошибку без ключа, но gate должен отработать
            result = generate_answer_with_gate("What is the VPN policy?", chunks)
            assert "confidence_score" in result
            assert "is_faithful" in result
            assert "faithfulness_method" in result
            assert 0.0 <= result["confidence_score"] <= 1.0
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_query_endpoint_returns_confidence(self):
        """POST /query возвращает confidence_score в ответе.

        ChromaDB создаёт singleton-клиент, поэтому мокируем _get_collection,
        чтобы не конфликтовать с другими тестами, использующими разные persist dirs.
        """
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient
        from rag.api.app import app

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # Пустая коллекция → пустой ответ

        with patch("rag.api.app._get_collection", return_value=mock_collection):
            client = TestClient(app)
            resp = client.post(
                "/query",
                json={"question": "What is the VPN policy?", "check_faithfulness": True},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "confidence_score" in data
        assert "is_faithful" in data
        assert "faithfulness_method" in data
        assert isinstance(data["confidence_score"], float)
