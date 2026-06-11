"""Тесты Multi-Query Retrieval: варианты запроса + RRF-агрегация + API endpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retrieval.multi_query import (
    MultiQueryConfig,
    MultiQueryResult,
    _rule_based_variants,
    compute_consistency_score,
    generate_query_variants,
    multi_query_retrieve,
)

# ---------------------------------------------------------------------------
# TestQueryVariants — генерация вариантов запроса
# ---------------------------------------------------------------------------


class TestQueryVariants:
    def test_returns_list(self):
        variants = generate_query_variants("What is VPN?", n=3)
        assert isinstance(variants, list)

    def test_first_variant_is_original(self):
        q = "How does remote work policy apply?"
        variants = generate_query_variants(q, n=3)
        assert variants[0] == q

    def test_returns_n_variants(self):
        variants = generate_query_variants("What is machine learning?", n=3)
        assert len(variants) == 3

    def test_returns_one_when_n_equals_one(self):
        variants = generate_query_variants("Some question", n=1)
        assert len(variants) == 1
        assert variants[0] == "Some question"

    def test_no_empty_variants(self):
        for q in ["VPN policy", "What is the onboarding process?", "GDPR requirements"]:
            for v in generate_query_variants(q, n=3):
                assert v.strip(), f"Empty variant from: {q!r}"

    def test_keyword_variant_uses_content_words(self):
        """Keyword variant должен содержать смысловые слова из запроса."""
        q = "What is the VPN authentication policy?"
        variants = generate_query_variants(q, n=3)
        # Хотя бы один вариант (кроме оригинала) должен содержать key terms
        non_original = [v for v in variants if v != q]
        combined = " ".join(non_original).lower()
        assert "vpn" in combined or "authentication" in combined or "policy" in combined

    def test_reformulation_differs_from_original(self):
        q = "What is the remote work policy?"
        variants = generate_query_variants(q, n=3)
        # При n=3 должны быть варианты отличные от оригинала
        unique = set(v.lower() for v in variants)
        assert len(unique) >= 2, f"All variants are identical: {variants}"

    def test_rule_based_what_is_reformulation(self):
        """'What is X' → 'Explain X' переформулировка."""
        variants = _rule_based_variants("What is the code review process?", n=3)
        # Один из вариантов должен начинаться с Explain
        has_explain = any(v.lower().startswith("explain") for v in variants)
        assert has_explain, f"Expected 'Explain ...' variant, got: {variants}"

    def test_rule_based_how_does_reformulation(self):
        """'How does X' → 'Describe the process of X' переформулировка."""
        variants = _rule_based_variants("How does authentication work?", n=3)
        has_process = any("process" in v.lower() or "describe" in v.lower() for v in variants)
        assert has_process, f"Expected 'describe/process' variant, got: {variants}"

    def test_no_llm_mode_without_api_key(self, monkeypatch):
        """Без ANTHROPIC_API_KEY use_llm=True должен вернуть rule-based варианты."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        variants = generate_query_variants("What is VPN?", n=3, use_llm=True)
        assert len(variants) >= 1
        assert variants[0] == "What is VPN?"


# ---------------------------------------------------------------------------
# TestConsistencyScore — метрика согласованности результатов
# ---------------------------------------------------------------------------


class TestConsistencyScore:
    def _make_chunks(self, texts: list[str]) -> list[dict]:
        return [{"text": t, "metadata": {}} for t in texts]

    def test_single_variant_returns_one(self):
        results = [self._make_chunks(["chunk A", "chunk B"])]
        assert compute_consistency_score(results) == 1.0

    def test_empty_results_returns_one(self):
        assert compute_consistency_score([]) == 1.0

    def test_identical_results_return_one(self):
        chunks = self._make_chunks(["A " * 15, "B " * 15, "C " * 15])
        score = compute_consistency_score([chunks, chunks])
        assert score == 1.0

    def test_disjoint_results_return_zero(self):
        a = self._make_chunks(["Alpha text here abc", "Beta text here def"])
        b = self._make_chunks(["Gamma text xyz uvw", "Delta text xyz uvw"])
        score = compute_consistency_score([a, b])
        assert score == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        shared = "Shared chunk about VPN policy requirements"
        a = self._make_chunks([shared, "Only in A variant result"])
        b = self._make_chunks([shared, "Only in B variant result"])
        score = compute_consistency_score([a, b])
        assert 0.0 < score < 1.0

    def test_score_bounded_zero_to_one(self):
        a = self._make_chunks(["text one aaa", "text two bbb"])
        b = self._make_chunks(["text two bbb", "text three ccc"])
        score = compute_consistency_score([a, b])
        assert 0.0 <= score <= 1.0

    def test_three_variants_aggregated(self):
        """Три варианта с частичным перекрытием → score между 0 и 1."""
        common = "Common document chunk about policies"
        a = self._make_chunks([common, "Only A document text here"])
        b = self._make_chunks([common, "Only B document text here"])
        c = self._make_chunks([common, "Only C document text here"])
        score = compute_consistency_score([a, b, c])
        # Jaccard: 1/2 для каждой пары → avg = 0.333
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# TestMultiQueryResult — dataclass
# ---------------------------------------------------------------------------


class TestMultiQueryResult:
    def test_to_dict_keys(self):
        r = MultiQueryResult(
            chunks=[{"text": "x"}],
            query_variants=["q1", "q2"],
            consistency_score=0.75,
        )
        d = r.to_dict()
        assert "n_chunks" in d
        assert "query_variants" in d
        assert "consistency_score" in d
        assert "retrieval_method" in d

    def test_default_method(self):
        r = MultiQueryResult(chunks=[], query_variants=["q"], consistency_score=1.0)
        assert r.retrieval_method == "multi_query"

    def test_n_chunks_reflects_list(self):
        chunks = [{"text": f"chunk {i}"} for i in range(4)]
        r = MultiQueryResult(chunks=chunks, query_variants=["q"], consistency_score=0.5)
        assert r.to_dict()["n_chunks"] == 4


# ---------------------------------------------------------------------------
# TestMultiQueryRetrieve — интеграционный тест с реальной коллекцией
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _integration_indexed_state():
    """Module-level: создаёт коллекцию один раз для TestMultiQueryRetrieve."""
    import contextlib
    import shutil
    from pathlib import Path

    import chromadb
    from rag.ingestion.loader import chunk_documents, load_documents
    from rag.retrieval.hybrid import HybridIndex
    from rag.retrieval.store import get_or_create_collection, index_chunks

    data_dir = Path(__file__).resolve().parents[1] / "data" / "documents"
    db_path = Path("/tmp/test_chroma_mq_retrieve")
    with contextlib.suppress(Exception):
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    with contextlib.suppress(Exception):
        client.delete_collection("test_mq_docs")
    collection = get_or_create_collection(client, name="test_mq_docs")
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    index_chunks(chunks, collection)
    hybrid_index = HybridIndex.build(chunks)
    return collection, hybrid_index


class TestMultiQueryRetrieve:
    @pytest.fixture
    def indexed_collection_and_index(self, _integration_indexed_state):
        """Делегирует к module-level фикстуре для избежания WAL-блокировок ChromaDB."""
        return _integration_indexed_state

    def test_returns_multi_query_result(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        result = multi_query_retrieve(
            "What is the VPN policy?",
            collection,
            hybrid_index,
            n_results=3,
        )
        assert isinstance(result, MultiQueryResult)

    def test_chunks_not_empty(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        result = multi_query_retrieve(
            "remote work requirements",
            collection,
            hybrid_index,
            n_results=3,
        )
        assert len(result.chunks) > 0

    def test_returns_correct_n_results(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        result = multi_query_retrieve(
            "security policy",
            collection,
            hybrid_index,
            n_results=3,
        )
        assert len(result.chunks) <= 3

    def test_query_variants_length(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        result = multi_query_retrieve(
            "What is the code review process?",
            collection,
            hybrid_index,
            n_results=3,
            config=MultiQueryConfig(n_variants=3),
        )
        assert len(result.query_variants) == 3

    def test_first_variant_is_original(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        q = "How does onboarding work?"
        result = multi_query_retrieve(q, collection, hybrid_index, n_results=3)
        assert result.query_variants[0] == q

    def test_consistency_score_in_range(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        result = multi_query_retrieve(
            "VPN authentication requirements",
            collection,
            hybrid_index,
            n_results=3,
        )
        assert 0.0 <= result.consistency_score <= 1.0

    def test_chunks_have_text_field(self, indexed_collection_and_index):
        collection, hybrid_index = indexed_collection_and_index
        result = multi_query_retrieve(
            "remote work policy",
            collection,
            hybrid_index,
            n_results=5,
        )
        for chunk in result.chunks:
            assert "text" in chunk

    def test_works_without_hybrid_index(self, indexed_collection_and_index):
        """Fallback на semantic-only при отсутствии BM25-индекса."""
        collection, _ = indexed_collection_and_index
        result = multi_query_retrieve(
            "What is the VPN policy?",
            collection,
            None,  # нет BM25-индекса
            n_results=3,
        )
        assert isinstance(result, MultiQueryResult)
        assert len(result.chunks) >= 0


# ---------------------------------------------------------------------------
# TestMultiQueryAPIEndpoint — API endpoint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _shared_indexed_state():
    """Module-level: создаёт коллекцию один раз для всех API тестов.

    Использует PersistentClient (chromadb 1.5.5+) вместо deprecated Client
    для избежания singleton конфликта между фикстурами.
    """
    import contextlib
    import shutil
    from pathlib import Path

    import chromadb
    from rag.ingestion.loader import chunk_documents, load_documents
    from rag.retrieval.hybrid import HybridIndex
    from rag.retrieval.store import get_or_create_collection, index_chunks

    data_dir = Path(__file__).resolve().parents[1] / "data" / "documents"
    db_path = Path("/tmp/test_chroma_mq_api")
    with contextlib.suppress(Exception):
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_path))

    with contextlib.suppress(Exception):
        chroma_client.delete_collection("test_mq_api_docs")

    collection = get_or_create_collection(chroma_client, name="test_mq_api_docs")
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)
    index_chunks(chunks, collection)
    hybrid_index = HybridIndex.build(chunks)
    return collection, hybrid_index, chunks


class TestMultiQueryAPIEndpoint:
    @pytest.fixture
    def client_with_docs(self, _shared_indexed_state):
        """TestClient с проиндексированными документами."""
        import rag.api.app as app_module
        from fastapi.testclient import TestClient

        collection, hybrid_index, chunks = _shared_indexed_state

        # Инициализируем глобальное состояние приложения
        app_module._collection = collection
        app_module._hybrid_index = hybrid_index
        app_module._indexed_chunks = chunks

        yield TestClient(app_module.app)

        # Cleanup
        app_module._collection = None
        app_module._hybrid_index = None
        app_module._indexed_chunks = []

    def test_multi_query_status_200(self, client_with_docs):
        resp = client_with_docs.post(
            "/query",
            json={"question": "What is the VPN policy?", "retrieval_method": "multi_query"},
        )
        assert resp.status_code == 200

    def test_multi_query_response_structure(self, client_with_docs):
        resp = client_with_docs.post(
            "/query",
            json={"question": "remote work requirements", "retrieval_method": "multi_query"},
        )
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieval_method" in data
        assert "query_variants" in data
        assert "consistency_score" in data

    def test_retrieval_method_is_multi_query(self, client_with_docs):
        resp = client_with_docs.post(
            "/query",
            json={"question": "security policy", "retrieval_method": "multi_query"},
        )
        assert resp.json()["retrieval_method"] == "multi_query"

    def test_query_variants_list_populated(self, client_with_docs):
        resp = client_with_docs.post(
            "/query",
            json={
                "question": "What is the code review process?",
                "retrieval_method": "multi_query",
                "n_query_variants": 3,
            },
        )
        data = resp.json()
        assert isinstance(data["query_variants"], list)
        assert len(data["query_variants"]) == 3

    def test_first_variant_equals_question(self, client_with_docs):
        q = "How does onboarding work for new employees?"
        resp = client_with_docs.post(
            "/query",
            json={"question": q, "retrieval_method": "multi_query"},
        )
        assert resp.json()["query_variants"][0] == q

    def test_consistency_score_in_range(self, client_with_docs):
        resp = client_with_docs.post(
            "/query",
            json={"question": "VPN authentication", "retrieval_method": "multi_query"},
        )
        score = resp.json()["consistency_score"]
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_other_methods_have_null_variants(self, client_with_docs):
        """Для hybrid/semantic query_variants и consistency_score должны быть null."""
        resp = client_with_docs.post(
            "/query",
            json={"question": "VPN policy", "retrieval_method": "hybrid"},
        )
        data = resp.json()
        assert data["query_variants"] is None
        assert data["consistency_score"] is None

    def test_custom_n_query_variants(self, client_with_docs):
        resp = client_with_docs.post(
            "/query",
            json={
                "question": "What are the security requirements?",
                "retrieval_method": "multi_query",
                "n_query_variants": 2,
            },
        )
        data = resp.json()
        assert len(data["query_variants"]) == 2
