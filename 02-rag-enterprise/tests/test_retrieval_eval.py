"""
Tests for Retrieval Evaluation Framework.

TestIRMetrics           — unit tests: P@K, R@K, MRR, MAP, NDCG@K
TestQueryMetrics        — compute_query_metrics и граничные случаи
TestAggregateMetrics    — aggregate_metrics (macro-averaging)
TestGoldenQueryDataset  — golden query dataset структура и фильтрация
TestChunkRelevance      — chunk_is_relevant и find_relevant_chunks
TestRetrievalEvalAPI    — API endpoints /evaluation/retrieval и /evaluation/golden-queries
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.evaluation.golden_queries import (
    GOLDEN_QUERIES,
    GoldenQuery,
    chunk_is_relevant,
    find_relevant_chunks,
    get_dataset_stats,
    get_golden_queries,
)
from rag.evaluation.retrieval_metrics import (
    aggregate_metrics,
    average_precision,
    compute_query_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ─────────────────────────────────────────────────────────────────────────────
# TestIRMetrics — единичные функции метрик
# ─────────────────────────────────────────────────────────────────────────────


class TestIRMetrics:
    def test_precision_at_k_perfect(self):
        """P@K=1.0 когда все retrieved релевантны."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_zero(self):
        """P@K=0.0 когда нет совпадений."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_precision_at_k_partial(self):
        """P@3 = 1/3 при одном совпадении в первых 3."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert abs(precision_at_k(retrieved, relevant, k=3) - 1 / 3) < 1e-9

    def test_precision_at_k_truncation(self):
        """P@K обрезает retrieved до K — результаты после K не влияют."""
        retrieved = ["x", "a", "b"]  # a, b релевантны, но после позиции 1
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, k=1) == 0.0
        assert precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_precision_at_k_empty_relevant(self):
        """Пустой relevant → P@K = 0."""
        assert precision_at_k(["a", "b"], set(), k=2) == 0.0

    def test_precision_at_k_empty_retrieved(self):
        """Пустой retrieved → P@K = 0."""
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_recall_at_k_perfect(self):
        """R@K=1.0 когда все релевантные найдены в top-K."""
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_recall_at_k_zero(self):
        """R@K=0.0 при отсутствии совпадений."""
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_recall_at_k_partial(self):
        """R@K < 1.0 когда часть релевантных найдена."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b", "c"}  # 3 релевантных, найден только 1
        assert abs(recall_at_k(retrieved, relevant, k=3) - 1 / 3) < 1e-9

    def test_recall_increases_with_k(self):
        """Recall монотонно не убывает с увеличением K."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        recalls = [recall_at_k(retrieved, relevant, k=i) for i in range(1, 6)]
        assert recalls == sorted(recalls)

    def test_reciprocal_rank_first_hit(self):
        """RR=1.0 когда первый результат релевантен."""
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_reciprocal_rank_second_hit(self):
        """RR=0.5 когда релевантен второй результат."""
        assert abs(reciprocal_rank(["x", "a", "b"], {"a"}) - 0.5) < 1e-9

    def test_reciprocal_rank_no_hit(self):
        """RR=0 если нет ни одного релевантного."""
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_reciprocal_rank_empty(self):
        """Пустые списки → RR=0."""
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_average_precision_perfect(self):
        """AP=1.0 при идеальном ранжировании: все релевантные первыми."""
        retrieved = ["a", "b", "c", "x"]
        relevant = {"a", "b", "c"}
        assert abs(average_precision(retrieved, relevant) - 1.0) < 1e-9

    def test_average_precision_all_miss(self):
        """AP=0 если ни один не релевантен."""
        assert average_precision(["x", "y"], {"a", "b"}) == 0.0

    def test_average_precision_position_penalty(self):
        """Более низкое AP когда релевантный на 3-й позиции."""
        # [rel, x, x] vs [x, x, rel]: первый должен иметь большее AP
        ap_early = average_precision(["a", "x", "y"], {"a"})
        ap_late = average_precision(["x", "y", "a"], {"a"})
        assert ap_early > ap_late

    def test_ndcg_at_k_perfect(self):
        """NDCG@K=1.0 при идеальном ранжировании."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert abs(ndcg_at_k(retrieved, relevant, k=3) - 1.0) < 1e-9

    def test_ndcg_at_k_zero(self):
        """NDCG@K=0.0 если нет совпадений."""
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_ndcg_normalized_range(self):
        """NDCG@K ∈ [0, 1] всегда."""
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        val = ndcg_at_k(retrieved, relevant, k=4)
        assert 0.0 <= val <= 1.0

    def test_ndcg_position_matters(self):
        """NDCG выше когда релевантный стоит раньше в списке."""
        relevant = {"a"}
        high = ndcg_at_k(["a", "x", "x"], relevant, k=3)
        low = ndcg_at_k(["x", "x", "a"], relevant, k=3)
        assert high > low


# ─────────────────────────────────────────────────────────────────────────────
# TestQueryMetrics
# ─────────────────────────────────────────────────────────────────────────────


class TestQueryMetrics:
    K_VALUES = [1, 3, 5]

    def test_compute_returns_all_k_values(self):
        """compute_query_metrics возвращает dict со всеми k_values."""
        qm = compute_query_metrics("test", ["a", "b"], {"a"}, self.K_VALUES)
        assert set(qm.precision_at_k.keys()) == {1, 3, 5}
        assert set(qm.recall_at_k.keys()) == {1, 3, 5}
        assert set(qm.ndcg_at_k.keys()) == {1, 3, 5}

    def test_compute_mrr_first_hit(self):
        """MRR=1.0 когда первый результат релевантен."""
        qm = compute_query_metrics("test", ["a", "b"], {"a"}, [1])
        assert qm.mrr == 1.0

    def test_compute_n_relevant(self):
        """n_relevant корректно считает ground truth."""
        qm = compute_query_metrics("test", ["a"], {"a", "b", "c"}, [1])
        assert qm.n_relevant == 3

    def test_compute_n_retrieved(self):
        """n_retrieved корректно считает полученные результаты."""
        qm = compute_query_metrics("test", ["a", "b", "c"], {"a"}, [1])
        assert qm.n_retrieved == 3

    def test_to_dict_structure(self):
        """to_dict() возвращает dict с обязательными полями."""
        qm = compute_query_metrics("q", ["a"], {"a"}, [1, 3])
        d = qm.to_dict()
        assert "query" in d
        assert "precision_at_k" in d
        assert "recall_at_k" in d
        assert "mrr" in d
        assert "average_precision" in d
        assert "ndcg_at_k" in d


# ─────────────────────────────────────────────────────────────────────────────
# TestAggregateMetrics
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateMetrics:
    K_VALUES = [1, 3, 5]

    def _make_qm(self, query: str, p: float) -> object:
        """Синтетический QueryMetrics с заданным P@1."""
        retrieved = ["a"] if p == 1.0 else ["x"]
        relevant = {"a"}
        return compute_query_metrics(query, retrieved, relevant, self.K_VALUES)

    def test_aggregate_empty(self):
        """Пустой список → нулевой отчёт."""
        report = aggregate_metrics([], self.K_VALUES)
        assert report.n_queries == 0
        assert report.mrr == 0.0
        assert report.map_score == 0.0

    def test_aggregate_single_query(self):
        """Один запрос → MRR = RR одного запроса."""
        qm = compute_query_metrics("q", ["a", "b"], {"a"}, self.K_VALUES)
        report = aggregate_metrics([qm], self.K_VALUES)
        assert report.n_queries == 1
        assert report.mrr == qm.mrr

    def test_aggregate_macro_averaging(self):
        """macro-averaging: mean P@1 = (P1 + P2) / 2."""
        qm1 = compute_query_metrics("q1", ["a"], {"a"}, self.K_VALUES)  # P@1=1
        qm2 = compute_query_metrics("q2", ["x"], {"a"}, self.K_VALUES)  # P@1=0
        report = aggregate_metrics([qm1, qm2], self.K_VALUES)
        assert abs(report.mean_precision_at_k[1] - 0.5) < 1e-4

    def test_aggregate_to_dict_structure(self):
        """to_dict() содержит все нужные поля."""
        qm = compute_query_metrics("q", ["a"], {"a"}, self.K_VALUES)
        report = aggregate_metrics([qm], self.K_VALUES)
        d = report.to_dict()
        assert "n_queries" in d
        assert "mean_precision_at_k" in d
        assert "mean_recall_at_k" in d
        assert "mrr" in d
        assert "map" in d
        assert "mean_ndcg_at_k" in d
        assert "per_query" in d

    def test_aggregate_perfect_retrieval(self):
        """Идеальный retrieval → все метрики = 1.0."""
        qm = compute_query_metrics("q", ["a", "b"], {"a", "b"}, [1, 2])
        report = aggregate_metrics([qm], [1, 2])
        assert abs(report.mrr - 1.0) < 1e-4
        assert abs(report.map_score - 1.0) < 1e-4
        assert abs(report.mean_ndcg_at_k[2] - 1.0) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# TestGoldenQueryDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestGoldenQueryDataset:
    def test_n_queries_20(self):
        """Golden dataset содержит ровно 20 запросов."""
        assert len(GOLDEN_QUERIES) == 20

    def test_all_queries_have_keywords(self):
        """Каждый запрос имеет хотя бы 3 relevant_keywords."""
        for q in GOLDEN_QUERIES:
            assert len(q.relevant_keywords) >= 3, f"Not enough keywords for: {q.query}"

    def test_categories_valid(self):
        """Все категории из допустимого множества."""
        valid = {"governance", "engineering", "remote", "onboarding", "product"}
        for q in GOLDEN_QUERIES:
            assert q.category in valid, f"Unknown category: {q.category}"

    def test_difficulties_valid(self):
        """Все уровни сложности из допустимого множества."""
        valid = {"easy", "medium", "hard"}
        for q in GOLDEN_QUERIES:
            assert q.difficulty in valid, f"Unknown difficulty: {q.difficulty}"

    def test_all_categories_represented(self):
        """Все 5 категорий присутствуют."""
        cats = {q.category for q in GOLDEN_QUERIES}
        assert "governance" in cats
        assert "engineering" in cats
        assert "remote" in cats
        assert "onboarding" in cats
        assert "product" in cats

    def test_all_difficulties_represented(self):
        """Все 3 уровня сложности присутствуют."""
        diffs = {q.difficulty for q in GOLDEN_QUERIES}
        assert "easy" in diffs
        assert "medium" in diffs
        # hard может отсутствовать в минимальной конфигурации

    def test_filter_by_category(self):
        """get_golden_queries(category=...) фильтрует корректно."""
        remote_queries = get_golden_queries(category="remote")
        assert all(q.category == "remote" for q in remote_queries)
        assert len(remote_queries) >= 1

    def test_filter_by_difficulty(self):
        """get_golden_queries(difficulty=...) фильтрует корректно."""
        easy_queries = get_golden_queries(difficulty="easy")
        assert all(q.difficulty == "easy" for q in easy_queries)
        assert len(easy_queries) >= 1

    def test_filter_combined(self):
        """Комбинированный фильтр возвращает пересечение."""
        result = get_golden_queries(category="governance", difficulty="easy")
        for q in result:
            assert q.category == "governance"
            assert q.difficulty == "easy"

    def test_get_dataset_stats_structure(self):
        """get_dataset_stats() возвращает dict с обязательными полями."""
        stats = get_dataset_stats()
        assert stats["n_queries"] == 20
        assert isinstance(stats["categories"], dict)
        assert isinstance(stats["difficulties"], dict)
        assert "categories_list" in stats

    def test_keyword_set_property(self):
        """keyword_set возвращает frozenset нижнего регистра."""
        q = GoldenQuery(
            query="test",
            relevant_keywords=("VPN", "Two-Factor", "Security"),
            category="remote",
            difficulty="easy",
        )
        ks = q.keyword_set
        assert "vpn" in ks
        assert "two-factor" in ks
        assert "security" in ks


# ─────────────────────────────────────────────────────────────────────────────
# TestChunkRelevance
# ─────────────────────────────────────────────────────────────────────────────


class TestChunkRelevance:
    VPN_QUERY = GoldenQuery(
        query="What security requirements apply when working remotely?",
        relevant_keywords=("VPN", "two-factor", "authentication", "security", "company data"),
        category="remote",
        difficulty="medium",
    )

    def test_relevant_chunk_detected(self):
        """Чанк с ≥2 ключевыми словами считается релевантным."""
        text = "All employees must connect via VPN. Two-factor authentication is required."
        assert chunk_is_relevant(text, self.VPN_QUERY, min_keywords=2) is True

    def test_irrelevant_chunk_rejected(self):
        """Чанк без ключевых слов — нерелевантен."""
        text = "The company cafeteria serves lunch from 12:00 to 14:00."
        assert chunk_is_relevant(text, self.VPN_QUERY, min_keywords=2) is False

    def test_single_keyword_insufficient(self):
        """Одного ключевого слова недостаточно при min_keywords=2."""
        # "sensitive" != "security" — только VPN совпадает (1 keyword), min_keywords=2 → False
        text = "Please use VPN when accessing sensitive files."
        assert chunk_is_relevant(text, self.VPN_QUERY, min_keywords=2) is False
        # Полностью нерелевантный чанк
        text2 = "Please use the encrypted connection for external access."
        assert chunk_is_relevant(text2, self.VPN_QUERY, min_keywords=2) is False

    def test_case_insensitive(self):
        """Поиск нечувствителен к регистру."""
        text = "vpn and TWO-FACTOR authentication required."
        assert chunk_is_relevant(text, self.VPN_QUERY, min_keywords=2) is True

    def test_find_relevant_chunks_returns_ids(self):
        """find_relevant_chunks возвращает set chunk IDs."""
        chunks = [
            {"text": "Use VPN and two-factor authentication for remote access.", "metadata": {}},
            {"text": "The cafeteria serves lunch from 12:00 to 14:00.", "metadata": {}},
            {"text": "Company security data requires VPN authentication.", "metadata": {}},
        ]
        relevant = find_relevant_chunks(chunks, self.VPN_QUERY, min_keywords=2)
        assert "chunk_0" in relevant
        assert "chunk_1" not in relevant
        assert "chunk_2" in relevant

    def test_find_relevant_chunks_empty_corpus(self):
        """Пустой корпус → пустое множество."""
        assert find_relevant_chunks([], self.VPN_QUERY) == set()

    def test_find_relevant_chunks_none_relevant(self):
        """Чанки без ключевых слов → пустое множество."""
        chunks = [{"text": "Hello world", "metadata": {}}, {"text": "Random text", "metadata": {}}]
        assert find_relevant_chunks(chunks, self.VPN_QUERY) == set()


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrievalEvalAPI — HTTP endpoints
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrievalEvalAPI:
    """API-тесты: создаём тестовый клиент без реального ChromaDB."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from rag.api.app import _reset_cache, _reset_memory, app

        _reset_cache()
        _reset_memory()
        return TestClient(app)

    def test_golden_queries_endpoint_200(self, client):
        """GET /evaluation/golden-queries возвращает 200."""
        resp = client.get("/evaluation/golden-queries")
        assert resp.status_code == 200

    def test_golden_queries_structure(self, client):
        """Ответ содержит n_queries, queries, dataset_stats."""
        resp = client.get("/evaluation/golden-queries")
        data = resp.json()
        assert "n_queries" in data
        assert "queries" in data
        assert "dataset_stats" in data

    def test_golden_queries_total_count(self, client):
        """Без фильтров возвращает все 20 запросов."""
        resp = client.get("/evaluation/golden-queries")
        assert resp.json()["n_queries"] == 20

    def test_golden_queries_filter_category(self, client):
        """Фильтр по категории сокращает результаты."""
        resp = client.get("/evaluation/golden-queries?category=remote")
        data = resp.json()
        assert data["n_queries"] < 20
        for q in data["queries"]:
            assert q["category"] == "remote"

    def test_golden_queries_filter_difficulty(self, client):
        """Фильтр по сложности сокращает результаты."""
        resp = client.get("/evaluation/golden-queries?difficulty=easy")
        data = resp.json()
        for q in data["queries"]:
            assert q["difficulty"] == "easy"

    def test_golden_queries_query_has_keywords(self, client):
        """Каждый запрос в ответе имеет relevant_keywords."""
        resp = client.get("/evaluation/golden-queries")
        for q in resp.json()["queries"]:
            assert "relevant_keywords" in q
            assert len(q["relevant_keywords"]) >= 3

    def test_retrieval_eval_200_with_corpus(self, client):
        """POST /evaluation/retrieval возвращает 200 (с документами или без)."""
        resp = client.post("/evaluation/retrieval", json={"retrieval_method": "semantic"})
        assert resp.status_code == 200
        data = resp.json()
        assert "mrr" in data
        assert "map" in data
        # Метрики всегда в допустимом диапазоне
        assert 0.0 <= data["mrr"] <= 1.0
        assert 0.0 <= data["map"] <= 1.0

    def test_retrieval_eval_response_structure(self, client):
        """POST /evaluation/retrieval возвращает все обязательные поля."""
        resp = client.post("/evaluation/retrieval", json={})
        data = resp.json()
        assert "n_queries" in data
        assert "k_values" in data
        assert "retrieval_method" in data
        assert "mean_precision_at_k" in data
        assert "mean_recall_at_k" in data
        assert "mrr" in data
        assert "map" in data
        assert "mean_ndcg_at_k" in data
        assert "per_query" in data

    def test_retrieval_eval_echoes_method(self, client):
        """Метод retrieval echoes в ответе."""
        resp = client.post("/evaluation/retrieval", json={"retrieval_method": "semantic"})
        assert resp.json()["retrieval_method"] == "semantic"

    def test_retrieval_eval_echoes_k_values(self, client):
        """k_values echoes в ответе."""
        resp = client.post("/evaluation/retrieval", json={"k_values": [1, 5]})
        assert resp.json()["k_values"] == [1, 5]

    def test_retrieval_eval_invalid_filter(self, client):
        """Несуществующая категория → 422 (нет совпадений в golden)."""
        resp = client.post("/evaluation/retrieval", json={"category": "nonexistent_category_xyz"})
        assert resp.status_code == 422

    def test_retrieval_eval_n_queries_with_filter(self, client):
        """Фильтр по сложности уменьшает n_queries в отчёте (0 документов → per_query=[])."""
        resp = client.post(
            "/evaluation/retrieval",
            json={"difficulty": "easy", "k_values": [1, 3]},
        )
        data = resp.json()
        # n_queries = число golden queries для данного фильтра
        easy_count = len([q for q in GOLDEN_QUERIES if q.difficulty == "easy"])
        assert data["n_queries"] == easy_count

    def test_retrieval_eval_metrics_range(self, client):
        """Все метрики ∈ [0, 1] при нулевом corpus."""
        resp = client.post("/evaluation/retrieval", json={})
        data = resp.json()
        assert 0.0 <= data["mrr"] <= 1.0
        assert 0.0 <= data["map"] <= 1.0
        for v in data["mean_precision_at_k"].values():
            assert 0.0 <= v <= 1.0

    def test_golden_queries_stats_all_categories(self, client):
        """dataset_stats содержит все 5 категорий."""
        resp = client.get("/evaluation/golden-queries")
        cats = resp.json()["dataset_stats"]["categories"]
        assert "governance" in cats
        assert "engineering" in cats
        assert "remote" in cats
        assert "onboarding" in cats
        assert "product" in cats
