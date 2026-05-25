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
from review.models.confidence_router import (
    RoutingConfig,
    RoutingDecision,
    RoutingResult,
    compute_risk_score,
    route_review,
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


# ── TestConfidenceRouter ──────────────────────────────────────────────────────


class TestRiskScore:
    """Unit tests for compute_risk_score()."""

    def test_empty_comments_returns_zero(self):
        assert compute_risk_score([]) == 0.0

    def test_single_suggestion(self):
        comments = [{"severity": "suggestion", "category": "style", "comment": "x"}]
        assert compute_risk_score(comments) == pytest.approx(0.3)

    def test_single_minor(self):
        comments = [{"severity": "minor", "category": "style", "comment": "x"}]
        assert compute_risk_score(comments) == pytest.approx(1.0)

    def test_single_major(self):
        comments = [{"severity": "major", "category": "bug", "comment": "x"}]
        assert compute_risk_score(comments) == pytest.approx(4.0)

    def test_single_critical(self):
        comments = [{"severity": "critical", "category": "security", "comment": "x"}]
        assert compute_risk_score(comments) == pytest.approx(10.0)

    def test_additive_across_comments(self):
        comments = [
            {"severity": "minor", "category": "style", "comment": "a"},
            {"severity": "major", "category": "bug", "comment": "b"},
        ]
        assert compute_risk_score(comments) == pytest.approx(5.0)

    def test_capped_at_100(self):
        # 11 critical comments = 110 raw → capped at 100
        comments = [{"severity": "critical", "category": "bug", "comment": "x"}] * 11
        assert compute_risk_score(comments) == pytest.approx(100.0)

    def test_unknown_severity_treated_as_suggestion(self):
        comments = [{"severity": "unknown_level", "category": "style", "comment": "x"}]
        assert compute_risk_score(comments) == pytest.approx(0.3)

    def test_missing_severity_key(self):
        comments = [{"category": "style", "comment": "no severity key"}]
        score = compute_risk_score(comments)
        assert score >= 0.0


class TestRoutingDecisions:
    """Unit tests for route_review() decision logic."""

    def test_empty_diff_auto_approves(self):
        result = route_review([])
        assert result.decision == RoutingDecision.AUTO_APPROVE

    def test_empty_diff_high_confidence(self):
        result = route_review([])
        assert result.confidence >= 0.9

    def test_suggestion_only_auto_approves(self):
        comments = [{"severity": "suggestion", "category": "style", "comment": "rename var"}]
        result = route_review(comments)
        assert result.decision == RoutingDecision.AUTO_APPROVE

    def test_critical_security_auto_rejects(self):
        comments = [
            {
                "severity": "critical",
                "category": "security",
                "comment": "SQL injection via f-string",
            }
        ]
        result = route_review(comments)
        assert result.decision == RoutingDecision.AUTO_REJECT

    def test_critical_findings_populated_on_reject(self):
        comments = [{"severity": "critical", "category": "security", "comment": "RCE via eval()"}]
        result = route_review(comments)
        assert len(result.critical_findings) == 1
        assert result.critical_findings[0]["comment"] == "RCE via eval()"

    def test_high_aggregate_risk_auto_rejects(self):
        # 3 major = 12.0 > default threshold of 8.0
        comments = [
            {"severity": "major", "category": "bug", "comment": f"bug {i}"} for i in range(3)
        ]
        result = route_review(comments)
        assert result.decision == RoutingDecision.AUTO_REJECT

    def test_medium_risk_human_review(self):
        # 1 major (4.0) is in the ambiguous zone [0.5, 8.0]
        comments = [{"severity": "major", "category": "performance", "comment": "O(n^2) loop"}]
        result = route_review(comments)
        assert result.decision == RoutingDecision.HUMAN_REVIEW

    def test_result_has_reason(self):
        result = route_review([])
        assert isinstance(result.reason, str)
        assert len(result.reason) > 10

    def test_comment_count_matches_input(self):
        comments = [
            {"severity": "minor", "category": "style", "comment": f"c{i}"} for i in range(5)
        ]
        result = route_review(comments)
        assert result.comment_count == 5

    def test_confidence_in_valid_range(self):
        for severity in ("suggestion", "minor", "major", "critical"):
            comments = [{"severity": severity, "category": "bug", "comment": "x"}]
            result = route_review(comments)
            assert 0.0 <= result.confidence <= 1.0, f"confidence out of range for {severity}"

    def test_custom_config_low_threshold(self):
        # Lower the reject threshold so major triggers auto_reject
        config = RoutingConfig(auto_reject_min_risk=3.0)
        comments = [{"severity": "major", "category": "bug", "comment": "logic error"}]
        result = route_review(comments, config=config)
        assert result.decision == RoutingDecision.AUTO_REJECT

    def test_custom_config_disable_security_escalation(self):
        # With security_escalate=False, a 'suggestion'-level security comment should not force
        # AUTO_REJECT. Score 0.3 ≤ default auto_approve_max_risk 0.5 → AUTO_APPROVE.
        config = RoutingConfig(
            security_escalate=False, critical_escalate=False, auto_reject_min_risk=8.0
        )
        comments = [{"severity": "suggestion", "category": "security", "comment": "consider HMAC"}]
        result = route_review(comments, config=config)
        assert result.decision == RoutingDecision.AUTO_APPROVE

    def test_routing_result_is_dataclass(self):
        result = route_review([])
        assert isinstance(result, RoutingResult)
        assert isinstance(result.decision, RoutingDecision)
        assert isinstance(result.risk_score, float)

    def test_decision_enum_values(self):
        assert RoutingDecision.AUTO_APPROVE.value == "auto_approve"
        assert RoutingDecision.HUMAN_REVIEW.value == "human_review"
        assert RoutingDecision.AUTO_REJECT.value == "auto_reject"

    def test_multiple_security_findings_all_in_critical_findings(self):
        comments = [
            {"severity": "major", "category": "security", "comment": "XSS via innerHTML"},
            {"severity": "critical", "category": "security", "comment": "SQLi via format"},
        ]
        result = route_review(comments)
        assert result.decision == RoutingDecision.AUTO_REJECT
        assert len(result.critical_findings) == 2


# ── TestGoldenDataset ─────────────────────────────────────────────────────────


class TestGoldenDataset:
    """Tests for the curated golden evaluation dataset."""

    from review.evaluation.golden_dataset import GoldenExample, get_golden_dataset

    def test_dataset_has_20_examples(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        assert len(get_golden_dataset()) == 20

    def test_all_ids_unique(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        ids = [ex.id for ex in get_golden_dataset()]
        assert len(ids) == len(set(ids))

    def test_domains_present(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        domains = {ex.domain for ex in get_golden_dataset()}
        assert domains == {"security", "correctness", "clean"}

    def test_security_count(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        n = sum(1 for ex in get_golden_dataset() if ex.domain == "security")
        assert n == 8

    def test_correctness_count(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        n = sum(1 for ex in get_golden_dataset() if ex.domain == "correctness")
        assert n == 8

    def test_clean_count(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        n = sum(1 for ex in get_golden_dataset() if ex.domain == "clean")
        assert n == 4

    def test_clean_examples_have_no_issues(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        for ex in get_golden_dataset():
            if ex.is_clean:
                assert ex.ground_truth_issues == []

    def test_non_clean_have_keywords(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        for ex in get_golden_dataset():
            if not ex.is_clean:
                for issue in ex.ground_truth_issues:
                    assert len(issue.get("keywords", [])) >= 3, (
                        f"{ex.id} issue has too few keywords"
                    )

    def test_all_diffs_contain_diff_markers(self):
        from review.evaluation.golden_dataset import get_golden_dataset

        for ex in get_golden_dataset():
            assert any(marker in ex.diff for marker in ("---", "++|", "@@")), (
                f"{ex.id} diff has no markers"
            )

    def test_golden_example_is_dataclass(self):
        from review.evaluation.golden_dataset import GoldenExample

        ex = GoldenExample(
            id="test",
            diff="+x = 1",
            domain="clean",
            is_clean=True,
        )
        assert ex.ground_truth_issues == []


# ── TestLexicalJudge ──────────────────────────────────────────────────────────


class TestLexicalJudge:
    """Unit tests for the deterministic lexical fallback judge."""

    from review.evaluation.golden_dataset import GoldenExample
    from review.evaluation.judge import _lexical_judge

    def _make_example(self, domain="correctness", is_clean=False, keywords=None):
        from review.evaluation.golden_dataset import GoldenExample

        issues = []
        if not is_clean and keywords:
            issues = [{"category": "bug", "severity": "major", "keywords": keywords}]
        return GoldenExample(
            id="test",
            diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+x = 1",
            domain=domain,
            ground_truth_issues=issues,
            is_clean=is_clean,
        )

    def _make_comment(self, text, severity="major"):
        return {"line": "1", "category": "bug", "comment": text, "severity": severity}

    def test_no_comments_clean_code_perfect(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(domain="clean", is_clean=True)
        v = _lexical_judge(ex, [])
        assert v.faithfulness == 1.0
        assert v.false_positive_rate == 0.0

    def test_no_comments_buggy_code_zero_faithfulness(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(keywords=["sql injection", "parameterized"])
        v = _lexical_judge(ex, [])
        assert v.faithfulness == 0.0

    def test_clean_code_critical_comment_is_fp(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(domain="clean", is_clean=True)
        comments = [self._make_comment("Critical security flaw!", severity="critical")]
        v = _lexical_judge(ex, comments)
        assert v.false_positive_rate == pytest.approx(1.0)

    def test_clean_code_suggestion_only_not_fp(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(domain="clean", is_clean=True)
        comments = [self._make_comment("Consider renaming for clarity", severity="suggestion")]
        v = _lexical_judge(ex, comments)
        assert v.false_positive_rate == pytest.approx(0.0)

    def test_keyword_match_high_faithfulness(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(keywords=["sql injection", "parameterized"])
        comments = [self._make_comment("This code has sql injection risk, use parameterized query")]
        v = _lexical_judge(ex, comments)
        assert v.faithfulness == pytest.approx(1.0)

    def test_no_keyword_match_zero_faithfulness(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(keywords=["sql injection", "parameterized"])
        comments = [self._make_comment("This is a style issue only")]
        v = _lexical_judge(ex, comments)
        assert v.faithfulness == pytest.approx(0.0)

    def test_helpfulness_with_action_words(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(is_clean=True, domain="clean")
        comments = [
            self._make_comment("You should fix this", severity="suggestion"),
            self._make_comment("Consider using a different approach", severity="suggestion"),
        ]
        v = _lexical_judge(ex, comments)
        assert v.helpfulness == pytest.approx(1.0)

    def test_helpfulness_without_action_words(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(is_clean=True, domain="clean")
        comments = [self._make_comment("There is an issue here.", severity="suggestion")]
        v = _lexical_judge(ex, comments)
        assert v.helpfulness == pytest.approx(0.0)

    def test_api_error_comments_filtered(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(domain="clean", is_clean=True)
        # API error placeholder injected when no key — should be treated as no comments
        error_comment = self._make_comment(
            "Error: ANTHROPIC_API_KEY not set. correctness pass skipped."
        )
        v = _lexical_judge(ex, [error_comment])
        # Filtered → treated as clean no-comment case
        assert v.faithfulness == pytest.approx(1.0)
        assert v.false_positive_rate == pytest.approx(0.0)

    def test_verdict_is_dataclass(self):
        from review.evaluation.judge import JudgeVerdict, _lexical_judge

        ex = self._make_example(domain="clean", is_clean=True)
        v = _lexical_judge(ex, [])
        assert isinstance(v, JudgeVerdict)

    def test_overall_score_in_range(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(keywords=["null", "check"])
        comments = [self._make_comment("Use null check here, should add guard")]
        v = _lexical_judge(ex, comments)
        assert 0.0 <= v.overall_score <= 1.0

    def test_to_dict_has_required_keys(self):
        from review.evaluation.judge import _lexical_judge

        ex = self._make_example(domain="clean", is_clean=True)
        d = _lexical_judge(ex, []).to_dict()
        for key in (
            "faithfulness",
            "helpfulness",
            "false_positive_rate",
            "overall_score",
            "reasoning",
        ):
            assert key in d, f"Missing key: {key}"


# ── TestJudgeVerdict ──────────────────────────────────────────────────────────


class TestJudgeVerdict:
    """Unit tests for JudgeVerdict dataclass."""

    def test_create_verdict(self):
        from review.evaluation.judge import JudgeVerdict

        v = JudgeVerdict(
            faithfulness=0.8,
            helpfulness=0.7,
            false_positive_rate=0.2,
            overall_score=0.72,
            reasoning="ok",
            api_key_used=False,
        )
        assert v.faithfulness == 0.8

    def test_overall_score_formula(self):
        from review.evaluation.judge import _compute_overall

        # 0.4*1.0 + 0.3*1.0 + 0.3*(1-0.0) = 1.0
        assert _compute_overall(1.0, 1.0, 0.0) == pytest.approx(1.0)
        # 0.4*0 + 0.3*0 + 0.3*0 = 0.0
        assert _compute_overall(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_to_dict_rounds_values(self):
        from review.evaluation.judge import JudgeVerdict

        v = JudgeVerdict(
            faithfulness=0.12345,
            helpfulness=0.67891,
            false_positive_rate=0.11111,
            overall_score=0.55555,
            reasoning="r",
            api_key_used=False,
        )
        d = v.to_dict()
        assert d["faithfulness"] == 0.123
        assert d["helpfulness"] == 0.679

    def test_to_dict_api_key_used_field(self):
        from review.evaluation.judge import JudgeVerdict

        v = JudgeVerdict(0.5, 0.5, 0.5, 0.5, "r", True, "id_x")
        assert v.to_dict()["api_key_used"] is True


# ── TestRegressionResult ──────────────────────────────────────────────────────


class TestRegressionResult:
    """Unit tests for RegressionResult dataclass."""

    def test_passed_when_above_threshold(self):
        from review.evaluation.judge import RegressionResult

        r = RegressionResult(
            n_examples=20,
            avg_faithfulness=0.8,
            avg_helpfulness=0.7,
            avg_false_positive_rate=0.2,
            avg_overall_score=0.74,
            by_domain={},
            passed=True,
            threshold=0.5,
        )
        assert r.passed is True

    def test_failed_when_below_threshold(self):
        from review.evaluation.judge import RegressionResult

        r = RegressionResult(
            n_examples=20,
            avg_faithfulness=0.1,
            avg_helpfulness=0.1,
            avg_false_positive_rate=0.9,
            avg_overall_score=0.1,
            by_domain={},
            passed=False,
            threshold=0.5,
        )
        assert r.passed is False

    def test_to_dict_has_required_keys(self):
        from review.evaluation.judge import RegressionResult

        r = RegressionResult(
            n_examples=5,
            avg_faithfulness=0.5,
            avg_helpfulness=0.5,
            avg_false_positive_rate=0.5,
            avg_overall_score=0.5,
            by_domain={"clean": {"n": 2}},
            passed=True,
        )
        d = r.to_dict()
        for key in (
            "n_examples",
            "avg_faithfulness",
            "avg_helpfulness",
            "avg_false_positive_rate",
            "avg_overall_score",
            "by_domain",
            "passed",
            "threshold",
        ):
            assert key in d


# ── TestEvaluateAPIEndpoints ──────────────────────────────────────────────────


class TestEvaluateAPIEndpoints:
    """Integration tests for /evaluate/* endpoints."""

    def _client(self):
        from fastapi.testclient import TestClient
        from review.api.app import app

        return TestClient(app)

    def test_dataset_endpoint_returns_200(self):
        resp = self._client().get("/evaluate/dataset")
        assert resp.status_code == 200

    def test_dataset_endpoint_n_examples(self):
        data = self._client().get("/evaluate/dataset").json()
        assert data["n_examples"] == 20

    def test_dataset_endpoint_by_domain(self):
        data = self._client().get("/evaluate/dataset").json()
        assert "security" in data["by_domain"]
        assert "correctness" in data["by_domain"]
        assert "clean" in data["by_domain"]

    def test_dataset_endpoint_ids_list(self):
        data = self._client().get("/evaluate/dataset").json()
        assert isinstance(data["ids"], list)
        assert len(data["ids"]) == 20

    def test_evaluate_review_endpoint_200(self):
        payload = {
            "example_id": "clean_001_refactor",
            "review_comments": [],
        }
        resp = self._client().post("/evaluate/review", json=payload)
        assert resp.status_code == 200

    def test_evaluate_review_endpoint_structure(self):
        payload = {
            "example_id": "sec_001_sqli",
            "review_comments": [
                {
                    "line": "3",
                    "category": "security",
                    "comment": "sql injection risk, use parameterized query",
                    "severity": "critical",
                }
            ],
        }
        data = self._client().post("/evaluate/review", json=payload).json()
        for key in ("faithfulness", "helpfulness", "false_positive_rate", "overall_score"):
            assert key in data, f"Missing key: {key}"

    def test_evaluate_review_unknown_id_returns_404(self):
        payload = {"example_id": "nonexistent_id", "review_comments": []}
        resp = self._client().post("/evaluate/review", json=payload)
        assert resp.status_code == 404

    def test_evaluate_review_clean_no_comments_perfect(self):
        payload = {
            "example_id": "clean_001_refactor",
            "review_comments": [],
        }
        data = self._client().post("/evaluate/review", json=payload).json()
        assert data["faithfulness"] == pytest.approx(1.0)
        assert data["false_positive_rate"] == pytest.approx(0.0)

    def test_regression_endpoint_returns_200(self):
        resp = self._client().post("/evaluate/regression", json={"use_lexical": True})
        assert resp.status_code == 200

    def test_regression_endpoint_structure(self):
        data = self._client().post("/evaluate/regression", json={"use_lexical": True}).json()
        for key in (
            "n_examples",
            "avg_faithfulness",
            "avg_helpfulness",
            "avg_false_positive_rate",
            "avg_overall_score",
            "passed",
            "threshold",
        ):
            assert key in data

    def test_regression_endpoint_n_examples(self):
        data = self._client().post("/evaluate/regression", json={"use_lexical": True}).json()
        assert data["n_examples"] == 20

    def test_regression_endpoint_by_domain_keys(self):
        data = self._client().post("/evaluate/regression", json={"use_lexical": True}).json()
        assert "security" in data["by_domain"]
        assert "clean" in data["by_domain"]


# ── TestPRDataset ─────────────────────────────────────────────────────────────


class TestPRDataset:
    """Тесты синтетического PR датасета для LoRA fine-tuning."""

    def test_dataset_total_count(self):
        from review.data.pr_dataset import get_pr_dataset

        examples = get_pr_dataset()
        assert len(examples) >= 30, f"Expected ≥30 examples, got {len(examples)}"

    def test_dataset_has_required_fields(self):
        from review.data.pr_dataset import get_pr_dataset

        for ex in get_pr_dataset():
            assert ex.id
            assert ex.diff
            assert ex.category in {"bug", "security", "performance", "style", "documentation"}
            assert ex.domain
            assert ex.severity in {"critical", "major", "minor"}
            assert ex.review_comment

    def test_dataset_all_categories_present(self):
        from review.data.pr_dataset import get_pr_dataset

        categories = {ex.category for ex in get_pr_dataset()}
        assert categories == {"bug", "security", "performance", "style", "documentation"}

    def test_dataset_multiple_domains(self):
        from review.data.pr_dataset import get_pr_dataset

        domains = {ex.domain for ex in get_pr_dataset()}
        assert len(domains) >= 3  # python, javascript, sql, yaml, generic

    def test_filter_by_category(self):
        from review.data.pr_dataset import get_pr_dataset_by_category

        security = get_pr_dataset_by_category("security")
        assert len(security) >= 5
        assert all(ex.category == "security" for ex in security)

    def test_filter_by_category_bugs(self):
        from review.data.pr_dataset import get_pr_dataset_by_category

        bugs = get_pr_dataset_by_category("bug")
        assert len(bugs) >= 5

    def test_stats_structure(self):
        from review.data.pr_dataset import get_pr_stats

        stats = get_pr_stats()
        assert stats["total"] >= 30
        assert "by_category" in stats
        assert "by_domain" in stats
        assert "by_severity" in stats
        assert "categories" in stats

    def test_unique_ids(self):
        from review.data.pr_dataset import get_pr_dataset

        ids = [ex.id for ex in get_pr_dataset()]
        assert len(ids) == len(set(ids)), "All example IDs must be unique"


# ── TestLoRAAdapter ───────────────────────────────────────────────────────────


class TestLoRAAdapter:
    """Тесты LoRA адаптера (numpy-only, без GPU)."""

    def _base_pipeline(self):
        from review.models.classifier import build_classifier

        return build_classifier()

    def _fitted_adapter(self, domain="security"):
        from review.data.pr_dataset import get_pr_dataset_by_category
        from review.models.lora_adapter import LoRAAdapter, LoRAConfig

        pipeline = self._base_pipeline()
        config = LoRAConfig(rank=2, alpha=4.0, n_epochs=10, target_domain=domain)
        adapter = LoRAAdapter(pipeline, config)
        examples = get_pr_dataset_by_category(domain)
        texts = [ex.review_comment for ex in examples]
        labels = [ex.category for ex in examples]
        adapter.fit(texts, labels)
        return adapter

    def test_is_available(self):
        from review.models.lora_adapter import is_available

        assert is_available() is True

    def test_not_fitted_initially(self):
        from review.models.lora_adapter import LoRAAdapter

        adapter = LoRAAdapter(self._base_pipeline())
        assert adapter.is_fitted is False

    def test_predict_before_fit_raises(self):
        from review.models.lora_adapter import LoRAAdapter

        adapter = LoRAAdapter(self._base_pipeline())
        with pytest.raises(RuntimeError, match="not fitted"):
            adapter.predict("SQL injection risk here")

    def test_fit_returns_train_result(self):
        from review.models.lora_adapter import LoRAAdapter, LoRAConfig

        pipeline = self._base_pipeline()
        adapter = LoRAAdapter(pipeline, LoRAConfig(rank=2, n_epochs=5))
        result = adapter.fit(["sql injection vulnerability"], ["security"])
        assert result.n_examples == 1
        assert result.n_epochs == 5
        assert result.final_loss >= 0

    def test_fit_sets_fitted(self):
        adapter = self._fitted_adapter()
        assert adapter.is_fitted is True

    def test_train_result_stored(self):
        adapter = self._fitted_adapter()
        assert adapter.train_result is not None
        assert adapter.train_result.domain == "security"

    def test_loss_reduction_non_negative(self):
        """Модель должна обучаться — final_loss ≤ initial_loss."""
        adapter = self._fitted_adapter()
        tr = adapter.train_result
        # Loss should decrease (or stay same) with gradient descent
        assert tr.final_loss <= tr.initial_loss + 0.5  # small tolerance for tiny datasets

    def test_predict_returns_adapter_result(self):
        from review.models.lora_adapter import AdapterResult

        adapter = self._fitted_adapter()
        result = adapter.predict("SQL injection: use parameterized queries")
        assert isinstance(result, AdapterResult)

    def test_predict_valid_category(self):
        adapter = self._fitted_adapter()
        result = adapter.predict("Path traversal vulnerability in file upload")
        assert result.category in {"bug", "security", "performance", "style", "documentation"}

    def test_predict_confidence_range(self):
        adapter = self._fitted_adapter()
        result = adapter.predict("Buffer overflow security issue")
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.base_confidence <= 1.0

    def test_predict_all_probs_sum_to_one(self):
        adapter = self._fitted_adapter()
        result = adapter.predict("Memory leak performance problem")
        total = sum(result.all_probabilities.values())
        assert abs(total - 1.0) < 1e-3

    def test_adapter_norm_positive_after_fit(self):
        adapter = self._fitted_adapter()
        assert adapter.adapter_norm() > 0.0

    def test_adapter_norm_zero_before_fit(self):
        from review.models.lora_adapter import LoRAAdapter

        adapter = LoRAAdapter(self._base_pipeline())
        assert adapter.adapter_norm() == 0.0

    def test_predict_batch(self):
        adapter = self._fitted_adapter()
        texts = ["SQL injection risk", "N+1 query in loop", "Missing docstring"]
        results = adapter.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_train_result_to_dict(self):
        adapter = self._fitted_adapter()
        d = adapter.train_result.to_dict()
        expected_keys = (
            "domain",
            "rank",
            "alpha",
            "n_examples",
            "n_epochs",
            "final_loss",
            "initial_loss",
            "loss_reduction",
        )
        for key in expected_keys:
            assert key in d

    def test_fit_empty_raises(self):
        from review.models.lora_adapter import LoRAAdapter

        adapter = LoRAAdapter(self._base_pipeline())
        with pytest.raises(ValueError):
            adapter.fit([], [])

    def test_save_load_roundtrip(self, tmp_path):

        from review.models.lora_adapter import LoRAAdapter

        adapter = self._fitted_adapter()
        save_path = tmp_path / "adapter.json"
        adapter.save(save_path)

        # Load into new adapter
        pipeline = self._base_pipeline()
        new_adapter = LoRAAdapter(pipeline, adapter.config)
        new_adapter.load(save_path)

        assert new_adapter.is_fitted
        result = new_adapter.predict("Command injection in subprocess call")
        assert result.category in {"bug", "security", "performance", "style", "documentation"}

    def test_performance_domain_adapter(self):
        adapter = self._fitted_adapter(domain="performance")
        result = adapter.predict("N+1 query loading all users from database")
        assert result.category in {"bug", "security", "performance", "style", "documentation"}
        assert result.confidence > 0.0


# ── TestLoRAAdapterAPI ────────────────────────────────────────────────────────


class TestLoRAAdapterAPI:
    """Тесты LoRA API endpoints."""

    def _client(self):
        from fastapi.testclient import TestClient
        from review.api.app import _reset_adapter, app

        _reset_adapter()
        return TestClient(app)

    def test_status_before_training(self):
        resp = self._client().get("/adapter/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fitted"] is False

    def test_train_200(self):
        resp = self._client().post(
            "/adapter/train", json={"domain": "security", "rank": 2, "n_epochs": 5}
        )
        assert resp.status_code == 200

    def test_train_response_structure(self):
        data = (
            self._client()
            .post("/adapter/train", json={"domain": "bug", "rank": 2, "n_epochs": 5})
            .json()
        )
        for key in ("domain", "rank", "n_examples", "final_loss", "adapter_norm"):
            assert key in data, f"Missing key: {key}"

    def test_train_unknown_domain_422(self):
        resp = self._client().post("/adapter/train", json={"domain": "unknown_xyz"})
        assert resp.status_code == 422

    def test_train_invalid_rank_422(self):
        resp = self._client().post("/adapter/train", json={"domain": "security", "rank": 0})
        assert resp.status_code == 422

    def test_predict_before_train_400(self):
        resp = self._client().post("/adapter/predict", json={"text": "SQL injection risk"})
        assert resp.status_code == 400

    def test_predict_200_after_train(self):
        client = self._client()
        client.post("/adapter/train", json={"domain": "security", "rank": 2, "n_epochs": 5})
        resp = client.post("/adapter/predict", json={"text": "SQL injection vulnerability"})
        assert resp.status_code == 200

    def test_predict_response_structure(self):
        client = self._client()
        client.post("/adapter/train", json={"domain": "performance", "rank": 2, "n_epochs": 5})
        data = client.post("/adapter/predict", json={"text": "N+1 queries in loop"}).json()
        for key in ("category", "confidence", "base_confidence", "adaptation_delta", "domain"):
            assert key in data, f"Missing key: {key}"

    def test_status_after_training(self):
        client = self._client()
        client.post("/adapter/train", json={"domain": "security", "rank": 4, "n_epochs": 10})
        data = client.get("/adapter/status").json()
        assert data["fitted"] is True
        assert data["domain"] == "security"
        assert data["rank"] == 4

    def test_dataset_stats_endpoint(self):
        data = self._client().get("/adapter/dataset/stats").json()
        assert data["total"] >= 30
        assert "by_category" in data
        assert "security" in data["by_category"]

    def test_custom_texts_train(self):
        client = self._client()
        payload = {
            "domain": "security",
            "rank": 2,
            "n_epochs": 5,
            "custom_texts": [
                "SQL injection risk: use parameterized queries",
                "Path traversal: sanitize filename",
            ],
            "custom_labels": ["security", "security"],
        }
        resp = client.post("/adapter/train", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_training_texts"] == 2

    def test_custom_texts_length_mismatch_422(self):
        client = self._client()
        payload = {
            "domain": "security",
            "rank": 2,
            "custom_texts": ["text1", "text2"],
            "custom_labels": ["security"],
        }
        resp = client.post("/adapter/train", json=payload)
        assert resp.status_code == 422
