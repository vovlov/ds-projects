"""Tests for LLM Code Review pipeline."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from review.data.samples import CATEGORIES, get_sample_reviews
from review.models.classifier import (
    build_classifier,
    classify_batch,
    classify_comment,
    get_categories,
)
from review.models.reviewer import review_code

# ── TestData ─────────────────────────────────────────────────────────────────


class TestData:
    def test_sample_count(self):
        samples = get_sample_reviews()
        assert len(samples) >= 10

    def test_sample_keys(self):
        for s in get_sample_reviews():
            assert "code_diff" in s
            assert "review_comment" in s
            assert "category" in s

    def test_categories_valid(self):
        for s in get_sample_reviews():
            assert s["category"] in CATEGORIES, f"Invalid category: {s['category']}"

    def test_all_categories_represented(self):
        cats = {s["category"] for s in get_sample_reviews()}
        for c in CATEGORIES:
            assert c in cats, f"Category '{c}' has no sample"

    def test_diffs_look_real(self):
        for s in get_sample_reviews():
            diff = s["code_diff"]
            assert "---" in diff or "+++" in diff or "@@" in diff


# ── TestClassifier ───────────────────────────────────────────────────────────


class TestClassifier:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return build_classifier()

    def test_pipeline_trains(self, pipeline):
        assert pipeline is not None
        assert hasattr(pipeline, "predict")

    def test_predict_returns_valid_category(self, pipeline):
        result = classify_comment("SQL injection via f-string formatting", pipeline)
        assert result["category"] in CATEGORIES

    def test_confidence_range(self, pipeline):
        result = classify_comment("Variable names are not descriptive", pipeline)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_probabilities_sum(self, pipeline):
        result = classify_comment("This is an O(n^2) loop", pipeline)
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_batch_classify(self, pipeline):
        texts = ["Missing docstring", "SQL injection risk", "O(n^2) nested loop"]
        results = classify_batch(texts, pipeline)
        assert len(results) == 3
        assert all(r["category"] in CATEGORIES for r in results)

    def test_get_categories(self):
        cats = get_categories()
        assert "bug" in cats
        assert "security" in cats
        assert len(cats) == 5


# ── TestReviewer ─────────────────────────────────────────────────────────────


class TestReviewer:
    def test_missing_api_key_graceful(self):
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = review_code("--- a/foo.py\n+++ b/foo.py\n+x = 1")
            assert isinstance(result, list)
            assert len(result) >= 1
            assert "ANTHROPIC_API_KEY" in result[0]["comment"]
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_missing_api_key_returns_valid_structure(self):
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = review_code("diff content here")
            comment = result[0]
            assert "line" in comment
            assert "category" in comment
            assert "comment" in comment
            assert "severity" in comment
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key


# ── TestAPI ──────────────────────────────────────────────────────────────────


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from review.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm-code-review"

    def test_classify_endpoint(self):
        from fastapi.testclient import TestClient
        from review.api.app import app

        client = TestClient(app)
        resp = client.post("/classify", json={"text": "SQL injection vulnerability found"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] in CATEGORIES
        assert 0.0 <= data["confidence"] <= 1.0

    def test_review_endpoint_no_key(self):
        """Without API key the /review endpoint should still return 200 with an error comment."""
        from fastapi.testclient import TestClient
        from review.api.app import app

        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            client = TestClient(app)
            resp = client.post("/review", json={"diff": "+x = 1"})
            assert resp.status_code == 200
            assert len(resp.json()["comments"]) >= 1
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key
