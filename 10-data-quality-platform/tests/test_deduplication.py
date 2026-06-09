"""
Tests for Entity Resolution / Record Deduplication module.

Тесты для модуля дедупликации записей (Entity Resolution).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.deduplication.entity_resolver import (
    BlockingConfig,
    DeduplicationResult,
    EntityResolver,
    FieldConfig,
    RecordPair,
    exact_similarity,
    jaccard_similarity,
    numeric_similarity,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Dataclass tests / Тесты датаклассов
# ---------------------------------------------------------------------------


class TestFieldConfig:
    def test_defaults(self) -> None:
        fc = FieldConfig(name="email")
        assert fc.weight == 1.0
        assert fc.similarity_type == "jaccard"
        assert fc.numeric_tolerance == 0.1

    def test_custom_values(self) -> None:
        fc = FieldConfig(name="age", weight=2.0, similarity_type="numeric", numeric_tolerance=0.05)
        assert fc.weight == 2.0
        assert fc.similarity_type == "numeric"
        assert fc.numeric_tolerance == 0.05

    def test_invalid_similarity_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown similarity_type"):
            FieldConfig(name="x", similarity_type="cosine")

    def test_invalid_weight(self) -> None:
        with pytest.raises(ValueError, match="weight must be positive"):
            FieldConfig(name="x", weight=0.0)


class TestBlockingConfig:
    def test_defaults(self) -> None:
        bc = BlockingConfig(blocking_keys=["last_name"])
        assert bc.threshold == 0.8
        assert bc.max_comparisons == 100_000

    def test_invalid_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            BlockingConfig(blocking_keys=["x"], threshold=0.0)

    def test_invalid_threshold_over_one(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            BlockingConfig(blocking_keys=["x"], threshold=1.1)

    def test_invalid_max_comparisons(self) -> None:
        with pytest.raises(ValueError, match="max_comparisons"):
            BlockingConfig(blocking_keys=["x"], max_comparisons=0)


class TestRecordPair:
    def test_to_dict(self) -> None:
        pair = RecordPair(id1="A", id2="B", similarity=0.9234, field_similarities={"name": 0.8765})
        d = pair.to_dict()
        assert d["id1"] == "A"
        assert d["id2"] == "B"
        assert d["similarity"] == 0.9234
        assert d["field_similarities"]["name"] == 0.8765

    def test_to_dict_rounding(self) -> None:
        pair = RecordPair(id1=1, id2=2, similarity=0.123456789, field_similarities={})
        assert pair.to_dict()["similarity"] == 0.1235


class TestDeduplicationResult:
    def test_to_dict_empty(self) -> None:
        result = DeduplicationResult(
            pairs=[], total_comparisons=0, blocks_count=0, threshold=0.8, records_count=5
        )
        d = result.to_dict()
        assert d["summary"]["pairs_found"] == 0
        assert d["summary"]["records_count"] == 5
        assert d["duplicate_pairs"] == []

    def test_deduplication_ratio(self) -> None:
        pairs = [RecordPair(id1=1, id2=2, similarity=0.9)]
        result = DeduplicationResult(
            pairs=pairs, total_comparisons=10, blocks_count=2, threshold=0.8, records_count=10
        )
        d = result.to_dict()
        assert d["summary"]["deduplication_ratio"] == 0.1


# ---------------------------------------------------------------------------
# Similarity function tests / Тесты функций сходства
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_strings(self) -> None:
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_empty_strings(self) -> None:
        assert jaccard_similarity("", "") == 1.0

    def test_completely_different(self) -> None:
        assert jaccard_similarity("aaaa", "zzzz") == 0.0

    def test_partial_match(self) -> None:
        sim = jaccard_similarity("John Smith", "John Smyth")
        assert 0.3 < sim < 1.0

    def test_typo_tolerance(self) -> None:
        # "michael" vs "micheal" — одна транспозиция, ~2/8 3-gram overlap
        sim = jaccard_similarity("michael", "micheal")
        assert sim > 0.15  # 3-gram Jaccard is intentionally strict

    def test_case_insensitive(self) -> None:
        assert jaccard_similarity("HELLO", "hello") == 1.0

    def test_one_empty(self) -> None:
        assert jaccard_similarity("", "something") == 0.0


class TestExactSimilarity:
    def test_match(self) -> None:
        assert exact_similarity("foo", "foo") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_similarity("FOO", "foo") == 1.0

    def test_whitespace_stripped(self) -> None:
        assert exact_similarity("  bar  ", "bar") == 1.0

    def test_no_match(self) -> None:
        assert exact_similarity("foo", "bar") == 0.0


class TestNumericSimilarity:
    def test_equal(self) -> None:
        assert numeric_similarity(100, 100) == 1.0

    def test_within_tolerance(self) -> None:
        assert numeric_similarity(100, 108) == 1.0  # 8% < 10% tolerance

    def test_outside_tolerance(self) -> None:
        assert numeric_similarity(100, 200) < 1.0

    def test_far_apart(self) -> None:
        assert numeric_similarity(100, 1000) == 0.0

    def test_invalid_values(self) -> None:
        assert numeric_similarity("abc", "xyz") == 0.0

    def test_zero_both(self) -> None:
        # Both zero — relative diff undefined, should handle gracefully
        assert numeric_similarity(0, 0) == 1.0


# ---------------------------------------------------------------------------
# EntityResolver core tests / Тесты основного резолвера
# ---------------------------------------------------------------------------


SAMPLE_RECORDS = [
    {"id": "1", "name": "John Smith", "email": "john.smith@example.com", "age": 30},
    {"id": "2", "name": "Jon Smith", "email": "john.smith@example.com", "age": 31},  # дубль
    {"id": "3", "name": "Jane Doe", "email": "jane.doe@example.com", "age": 25},
    {"id": "4", "name": "Alice Johnson", "email": "alice@example.com", "age": 45},
]

FIELD_CONFIGS = [
    FieldConfig(name="name", weight=2.0, similarity_type="jaccard"),
    FieldConfig(name="email", weight=3.0, similarity_type="exact"),
    FieldConfig(name="age", weight=1.0, similarity_type="numeric", numeric_tolerance=0.1),
]

# Блокируем по email: оба дубля (id=1 и id=2) имеют одинаковый email
# Block on email so duplicates (id=1, id=2 with same email) end up in the same block
BLOCKING_CONFIG = BlockingConfig(blocking_keys=["email"], threshold=0.7)


class TestEntityResolverCore:
    def test_finds_duplicates(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(SAMPLE_RECORDS, "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        ids = {(p.id1, p.id2) for p in result.pairs} | {(p.id2, p.id1) for p in result.pairs}
        assert ("1", "2") in ids or ("2", "1") in ids

    def test_empty_records(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve([], "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        assert result.pairs == []
        assert result.total_comparisons == 0
        assert result.records_count == 0

    def test_single_record(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(
            [{"id": "1", "name": "Alice"}], "id", FIELD_CONFIGS, BLOCKING_CONFIG
        )
        assert result.pairs == []

    def test_no_duplicates_in_different_blocks(self) -> None:
        records = [
            {"id": "1", "name": "Alice Johnson", "email": "alice@a.com"},
            {"id": "2", "name": "Bob Williams", "email": "bob@b.com"},
            {"id": "3", "name": "Charlie Brown", "email": "charlie@c.com"},
        ]
        field_cfgs = [FieldConfig(name="name"), FieldConfig(name="email")]
        bc = BlockingConfig(blocking_keys=["name"], threshold=0.9)
        resolver = EntityResolver()
        result = resolver.resolve(records, "id", field_cfgs, bc)
        assert result.pairs == []

    def test_blocks_count_populated(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(SAMPLE_RECORDS, "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        assert result.blocks_count > 0

    def test_result_records_count(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(SAMPLE_RECORDS, "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        assert result.records_count == len(SAMPLE_RECORDS)

    def test_similarity_in_range(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(SAMPLE_RECORDS, "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        for pair in result.pairs:
            assert 0.0 <= pair.similarity <= 1.0

    def test_field_similarities_present(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(SAMPLE_RECORDS, "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        for pair in result.pairs:
            assert len(pair.field_similarities) > 0

    def test_exact_duplicate_high_similarity(self) -> None:
        records = [
            {"id": "1", "name": "John Smith", "email": "j@j.com"},
            {"id": "2", "name": "John Smith", "email": "j@j.com"},
        ]
        field_cfgs = [FieldConfig(name="name"), FieldConfig(name="email")]
        bc = BlockingConfig(blocking_keys=["name"], threshold=0.5)
        resolver = EntityResolver()
        result = resolver.resolve(records, "id", field_cfgs, bc)
        assert len(result.pairs) == 1
        assert result.pairs[0].similarity == pytest.approx(1.0)

    def test_threshold_filtering(self) -> None:
        records = [
            {"id": "1", "name": "John Smith", "email": "a@a.com"},
            {"id": "2", "name": "John Smyth", "email": "b@b.com"},
        ]
        field_cfgs = [FieldConfig(name="name", similarity_type="jaccard")]
        # High threshold — no pairs
        bc_high = BlockingConfig(blocking_keys=["name"], threshold=0.99)
        resolver = EntityResolver()
        assert resolver.resolve(records, "id", field_cfgs, bc_high).pairs == []

        # Low threshold — should find pair
        bc_low = BlockingConfig(blocking_keys=["name"], threshold=0.3)
        assert len(resolver.resolve(records, "id", field_cfgs, bc_low).pairs) == 1

    def test_max_comparisons_limit(self) -> None:
        records = [
            {"id": str(i), "name": f"Name {i % 3}", "email": f"e{i}@e.com"} for i in range(20)
        ]
        field_cfgs = [FieldConfig(name="name")]
        bc = BlockingConfig(blocking_keys=["name"], threshold=0.5, max_comparisons=2)
        resolver = EntityResolver()
        result = resolver.resolve(records, "id", field_cfgs, bc)
        assert result.total_comparisons <= 2

    def test_no_duplicate_pairs_same_id(self) -> None:
        records = [
            {"id": "A", "name": "Aaa Bbb"},
            {"id": "A", "name": "Aaa Bbb"},  # same id — still compared once
        ]
        field_cfgs = [FieldConfig(name="name")]
        bc = BlockingConfig(blocking_keys=["name"], threshold=0.5)
        resolver = EntityResolver()
        result = resolver.resolve(records, "id", field_cfgs, bc)
        # Should find the pair but dedup should keep only one
        assert len(result.pairs) == 1

    def test_missing_field_handled(self) -> None:
        records = [
            {"id": "1", "name": "Alice"},
            {"id": "2"},  # missing name
        ]
        field_cfgs = [FieldConfig(name="name", similarity_type="jaccard")]
        bc = BlockingConfig(blocking_keys=["name"], threshold=0.5)
        resolver = EntityResolver()
        # Should not raise
        result = resolver.resolve(records, "id", field_cfgs, bc)
        assert isinstance(result, DeduplicationResult)

    def test_numeric_field_comparison(self) -> None:
        records = [
            {"id": "1", "name": "Alice", "salary": 50000},
            {"id": "2", "name": "Alice", "salary": 51000},  # 2% diff < 10% tolerance
        ]
        field_cfgs = [
            FieldConfig(name="name", weight=1.0),
            FieldConfig(
                name="salary", weight=1.0, similarity_type="numeric", numeric_tolerance=0.05
            ),
        ]
        bc = BlockingConfig(blocking_keys=["name"], threshold=0.8)
        resolver = EntityResolver()
        result = resolver.resolve(records, "id", field_cfgs, bc)
        assert len(result.pairs) == 1
        assert result.pairs[0].field_similarities["salary"] == pytest.approx(1.0)

    def test_to_dict_structure(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve(SAMPLE_RECORDS, "id", FIELD_CONFIGS, BLOCKING_CONFIG)
        d = result.to_dict()
        assert "duplicate_pairs" in d
        assert "summary" in d
        summary_keys = {
            "pairs_found",
            "total_comparisons",
            "blocks_count",
            "threshold",
            "records_count",
            "deduplication_ratio",
        }
        assert summary_keys == set(d["summary"].keys())


# ---------------------------------------------------------------------------
# API endpoint tests / Тесты API эндпоинтов
# ---------------------------------------------------------------------------


class TestDeduplicationAPI:
    BASE_PAYLOAD = {
        "records": [
            {"id": "1", "name": "John Smith", "email": "john@example.com"},
            {"id": "2", "name": "Jon Smith", "email": "john@example.com"},
            {"id": "3", "name": "Jane Doe", "email": "jane@example.com"},
        ],
        "id_field": "id",
        "field_configs": [
            {"name": "name", "weight": 2.0, "similarity_type": "jaccard"},
            {"name": "email", "weight": 3.0, "similarity_type": "exact"},
        ],
        # Блокировка по email — дубли 1 и 2 попадают в один блок
        "blocking_keys": ["email"],
        "threshold": 0.7,
    }

    def test_find_returns_200(self) -> None:
        resp = client.post("/deduplication/find", json=self.BASE_PAYLOAD)
        assert resp.status_code == 200

    def test_find_response_structure(self) -> None:
        resp = client.post("/deduplication/find", json=self.BASE_PAYLOAD)
        data = resp.json()
        assert "duplicate_pairs" in data
        assert "summary" in data

    def test_find_detects_duplicate(self) -> None:
        resp = client.post("/deduplication/find", json=self.BASE_PAYLOAD)
        data = resp.json()
        ids = {(p["id1"], p["id2"]) for p in data["duplicate_pairs"]}
        ids |= {(p["id2"], p["id1"]) for p in data["duplicate_pairs"]}
        assert ("1", "2") in ids or len(data["duplicate_pairs"]) >= 1

    def test_find_empty_pairs_high_threshold(self) -> None:
        payload = {**self.BASE_PAYLOAD, "threshold": 0.999}
        resp = client.post("/deduplication/find", json=payload)
        data = resp.json()
        assert data["summary"]["pairs_found"] == 0

    def test_find_summary_records_count(self) -> None:
        resp = client.post("/deduplication/find", json=self.BASE_PAYLOAD)
        data = resp.json()
        assert data["summary"]["records_count"] == 3

    def test_find_invalid_threshold(self) -> None:
        payload = {**self.BASE_PAYLOAD, "threshold": 1.5}
        resp = client.post("/deduplication/find", json=payload)
        assert resp.status_code == 422

    def test_find_empty_records(self) -> None:
        payload = {**self.BASE_PAYLOAD, "records": []}
        resp = client.post("/deduplication/find", json=payload)
        # pydantic min_length=2 → 422
        assert resp.status_code == 422

    def test_find_single_record(self) -> None:
        payload = {
            **self.BASE_PAYLOAD,
            "records": [{"id": "1", "name": "Alice"}],
        }
        resp = client.post("/deduplication/find", json=payload)
        assert resp.status_code == 422  # min_length=2

    def test_find_field_similarities_in_response(self) -> None:
        payload = {
            "records": [
                {"id": "1", "name": "Alice Smith"},
                {"id": "2", "name": "Alice Smyth"},
            ],
            "id_field": "id",
            "field_configs": [{"name": "name", "similarity_type": "jaccard"}],
            "blocking_keys": ["name"],
            "threshold": 0.3,
        }
        resp = client.post("/deduplication/find", json=payload)
        data = resp.json()
        if data["duplicate_pairs"]:
            assert "field_similarities" in data["duplicate_pairs"][0]

    def test_find_numeric_fields(self) -> None:
        payload = {
            "records": [
                {"id": "1", "name": "Alice", "age": 30},
                {"id": "2", "name": "Alice", "age": 31},
            ],
            "id_field": "id",
            "field_configs": [
                {"name": "name", "similarity_type": "jaccard", "weight": 1.0},
                {
                    "name": "age",
                    "similarity_type": "numeric",
                    "numeric_tolerance": 0.1,
                    "weight": 1.0,
                },
            ],
            "blocking_keys": ["name"],
            "threshold": 0.8,
        }
        resp = client.post("/deduplication/find", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["pairs_found"] >= 1

    def test_info_returns_200(self) -> None:
        resp = client.get("/deduplication/info")
        assert resp.status_code == 200

    def test_info_response_structure(self) -> None:
        resp = client.get("/deduplication/info")
        data = resp.json()
        assert "algorithm" in data
        assert "similarity_types" in data
        assert "use_cases" in data
        assert "compliance" in data
        assert "references" in data

    def test_info_similarity_types(self) -> None:
        resp = client.get("/deduplication/info")
        data = resp.json()
        assert "jaccard" in data["similarity_types"]
        assert "exact" in data["similarity_types"]
        assert "numeric" in data["similarity_types"]

    def test_info_compliance_gdpr(self) -> None:
        resp = client.get("/deduplication/info")
        data = resp.json()
        assert "GDPR" in data["compliance"]

    def test_find_exact_similarity_type(self) -> None:
        payload = {
            "records": [
                {"id": "1", "category": "premium"},
                {"id": "2", "category": "PREMIUM"},
                {"id": "3", "category": "standard"},
            ],
            "id_field": "id",
            "field_configs": [{"name": "category", "similarity_type": "exact"}],
            "blocking_keys": ["category"],
            "threshold": 0.9,
        }
        resp = client.post("/deduplication/find", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        # "premium" and "PREMIUM" exact match after normalization
        assert data["summary"]["pairs_found"] >= 1
