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
from review.models.multi_review import (
    MultiReviewResult,
    SemgrepFinding,
    correctness_pass,
    multi_model_review,
    security_pass,
    self_consistency_score,
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


# ── TestMultiModelReview ─────────────────────────────────────────────────────


class TestMultiModelReview:
    """Tests for the two-pass multi-model review (no API key needed)."""

    SAMPLE_DIFF = (
        "--- a/auth.py\n+++ b/auth.py\n"
        "@@ -1,5 +1,6 @@\n"
        "+import os\n"
        " def login(user, password):\n"
        "-    query = f\"SELECT * FROM users WHERE name='{user}'\"\n"
        '+    query = "SELECT * FROM users WHERE name=\'" + user + "\'"\n'
        "+    secret = 'hardcoded_secret_123'\n"
        "     return db.execute(query)\n"
    )

    # ── Dataclass tests ───────────────────────────────────────────────────────

    def test_semgrep_finding_create(self):
        f = SemgrepFinding(
            rule_id="python.lang.security.audit.sqli",
            message="SQL injection via string concatenation",
            severity="ERROR",
            line=4,
            code_snippet='query = "SELECT..." + user',
        )
        assert f.rule_id == "python.lang.security.audit.sqli"
        assert f.line == 4

    def test_semgrep_finding_defaults(self):
        f = SemgrepFinding(rule_id="r", message="m", severity="WARNING", line=1)
        assert f.code_snippet == ""

    def test_multi_review_result_create(self):
        r = MultiReviewResult()
        assert r.correctness_comments == []
        assert r.security_comments == []
        assert r.consistency_score == 0.0
        assert isinstance(r.summary, dict)

    # ── correctness_pass (no API key) ─────────────────────────────────────────

    def test_correctness_pass_no_key_returns_list(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = correctness_pass(self.SAMPLE_DIFF)
            assert isinstance(result, list) and len(result) >= 1
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_correctness_pass_no_key_mentions_pass(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = correctness_pass(self.SAMPLE_DIFF)
            comment_text = result[0].get("comment", "")
            assert "correctness" in comment_text
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_correctness_pass_no_key_valid_structure(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = correctness_pass(self.SAMPLE_DIFF)
            for c in result:
                assert "line" in c and "category" in c
                assert "comment" in c and "severity" in c
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    # ── security_pass (no API key) ────────────────────────────────────────────

    def test_security_pass_no_key_returns_list(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = security_pass(self.SAMPLE_DIFF)
            assert isinstance(result, list) and len(result) >= 1
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_security_pass_no_key_mentions_pass(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = security_pass(self.SAMPLE_DIFF)
            comment_text = result[0].get("comment", "")
            assert "security" in comment_text
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_security_pass_with_semgrep_no_key(self):
        """Semgrep findings are accepted even without API key (graceful)."""
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            findings = [
                SemgrepFinding(
                    rule_id="sqli",
                    message="SQL injection",
                    severity="ERROR",
                    line=4,
                )
            ]
            result = security_pass(self.SAMPLE_DIFF, semgrep_findings=findings)
            assert isinstance(result, list) and len(result) >= 1
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    # ── self_consistency_score ────────────────────────────────────────────────

    def test_self_consistency_empty_returns_zero(self):
        assert self_consistency_score("diff", []) == 0.0

    def test_self_consistency_error_comment_returns_zero(self):
        error_comments = [
            {
                "line": "",
                "category": "documentation",
                "comment": "Error: ANTHROPIC_API_KEY not set.",
                "severity": "critical",
            }
        ]
        assert self_consistency_score("diff", error_comments) == 0.0

    def test_self_consistency_good_review_above_half(self):
        """A specific, multi-severity, multi-category review should score > 0.5."""
        good_comments = [
            {
                "line": "+query = '...' + user",
                "category": "security",
                "comment": (
                    "SQL injection via string concatenation (CWE-89). "
                    "Use parameterized queries: cursor.execute(q, (user,))."
                ),
                "severity": "critical",
            },
            {
                "line": "+secret = 'hardcoded_secret_123'",
                "category": "bug",
                "comment": (
                    "Hardcoded credential detected. "
                    "Move to environment variable or secrets manager."
                ),
                "severity": "major",
            },
            {
                "line": "def login(user, password):",
                "category": "documentation",
                "comment": "Missing docstring. Add param types and return type annotation.",
                "severity": "minor",
            },
        ]
        score = self_consistency_score("diff", good_comments)
        assert score > 0.5

    def test_self_consistency_always_in_range(self):
        import random

        random.seed(42)
        for _ in range(10):
            n = random.randint(0, 5)
            comments = [
                {
                    "line": "line" if random.random() > 0.5 else "",
                    "category": random.choice(["bug", "security", "style"]),
                    "comment": "x" * random.randint(0, 100),
                    "severity": random.choice(["critical", "major", "minor"]),
                }
                for _ in range(n)
            ]
            score = self_consistency_score("diff", comments)
            assert 0.0 <= score <= 1.0

    # ── multi_model_review (no API key) ───────────────────────────────────────

    def test_multi_model_review_returns_result_type(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = multi_model_review(self.SAMPLE_DIFF)
            assert isinstance(result, MultiReviewResult)
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_multi_model_review_no_key_verdict(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = multi_model_review(self.SAMPLE_DIFF)
            assert result.summary["verdict"] == "api_key_missing"
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_multi_model_review_summary_has_required_keys(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = multi_model_review(self.SAMPLE_DIFF)
            for key in (
                "total",
                "correctness_issues",
                "security_issues",
                "by_severity",
                "consistency_score",
                "verdict",
            ):
                assert key in result.summary, f"Missing key: {key}"
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_multi_model_review_all_comments_is_union(self):
        """all_comments should contain items from both passes (no dups)."""
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = multi_model_review(self.SAMPLE_DIFF)
            n_correctness = len(result.correctness_comments)
            n_security = len(result.security_comments)
            n_all = len(result.all_comments)
            # all_comments is a deduplicated union — size ≤ sum of both
            assert n_all <= n_correctness + n_security
            assert n_all >= max(n_correctness, n_security)
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_multi_model_review_consistency_score_is_float(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = multi_model_review(self.SAMPLE_DIFF)
            assert isinstance(result.consistency_score, float)
            assert 0.0 <= result.consistency_score <= 1.0
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    # ── /review/multi API endpoint ────────────────────────────────────────────

    def test_multi_review_endpoint_no_key(self):
        """Without API key the /review/multi endpoint should return 200."""
        from fastapi.testclient import TestClient
        from review.api.app import app

        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            client = TestClient(app)
            resp = client.post("/review/multi", json={"diff": self.SAMPLE_DIFF})
            assert resp.status_code == 200
            data = resp.json()
            assert "correctness_comments" in data
            assert "security_comments" in data
            assert "all_comments" in data
            assert "consistency_score" in data
            assert "summary" in data
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_multi_review_endpoint_verdict_field(self):
        from fastapi.testclient import TestClient
        from review.api.app import app

        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            client = TestClient(app)
            resp = client.post("/review/multi", json={"diff": "+x = 1"})
            assert resp.status_code == 200
            verdict = resp.json()["summary"]["verdict"]
            assert verdict in ("pass", "review_required", "fail", "api_key_missing")
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old
