"""
Тесты расширенной батареи статистических тестов дрейфа.
Tests for extended statistical drift test battery (stat_tests.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.quality.stat_tests import (
    batch_extended_drift,
    chi2_test,
    extended_drift_test,
    js_divergence,
    js_severity,
    wasserstein_distance,
    wasserstein_severity,
)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# TestWassersteinDistance
# ---------------------------------------------------------------------------


class TestWassersteinDistance:
    def test_identical_distributions_zero(self) -> None:
        x = RNG.normal(0, 1, 200).tolist()
        assert wasserstein_distance(x, x) == 0.0

    def test_shifted_distributions_positive(self) -> None:
        ref = RNG.normal(0, 1, 500)
        cur = RNG.normal(5, 1, 500)  # сдвинуто на 5 σ
        d = wasserstein_distance(ref, cur)
        assert d > 0.5, f"Ожидали > 0.5, получили {d}"

    def test_same_scale_same_shift_proportional(self) -> None:
        ref = RNG.normal(0, 1, 500)
        cur_small = RNG.normal(0.5, 1, 500)
        cur_large = RNG.normal(2.0, 1, 500)
        d_small = wasserstein_distance(ref, cur_small)
        d_large = wasserstein_distance(ref, cur_large)
        assert d_small < d_large, "Больший сдвиг должен давать большее расстояние"

    def test_returns_float(self) -> None:
        ref = list(range(100))
        cur = list(range(50, 150))
        result = wasserstein_distance(ref, cur)
        assert isinstance(result, float)

    def test_empty_reference_returns_zero(self) -> None:
        assert wasserstein_distance([], [1, 2, 3]) == 0.0

    def test_symmetric(self) -> None:
        ref = RNG.normal(0, 1, 200)
        cur = RNG.normal(1, 1, 200)
        assert abs(wasserstein_distance(ref, cur) - wasserstein_distance(cur, ref)) < 1e-5


class TestWassersteinSeverity:
    def test_ok_when_small(self) -> None:
        assert wasserstein_severity(0.01, scale=1.0) == "ok"

    def test_moderate_when_medium(self) -> None:
        assert wasserstein_severity(0.1, scale=1.0) == "moderate"

    def test_critical_when_large(self) -> None:
        assert wasserstein_severity(0.5, scale=1.0) == "critical"

    def test_scale_invariant(self) -> None:
        # расстояние 100 при масштабе 1000 = 0.1 normalised → moderate
        assert wasserstein_severity(100.0, scale=1000.0) == "moderate"

    def test_zero_scale_falls_back_to_one(self) -> None:
        sev = wasserstein_severity(0.5, scale=0.0)
        assert sev in ("moderate", "critical")


# ---------------------------------------------------------------------------
# TestJSDivergence
# ---------------------------------------------------------------------------


class TestJSDivergence:
    def test_identical_distributions_near_zero(self) -> None:
        x = RNG.normal(0, 1, 500).tolist()
        js = js_divergence(x, x)
        assert js < 0.01, f"Идентичные распределения → ~0, получили {js}"

    def test_very_different_distributions_high(self) -> None:
        ref = RNG.normal(0, 0.1, 500)
        cur = RNG.normal(10, 0.1, 500)
        js = js_divergence(ref, cur)
        assert js > 0.5, f"Очень разные распределения → high JS, получили {js}"

    def test_bounded_zero_to_one(self) -> None:
        ref = RNG.normal(0, 1, 300)
        cur = RNG.uniform(5, 10, 300)
        js = js_divergence(ref, cur)
        assert 0.0 <= js <= 1.0

    def test_returns_float(self) -> None:
        result = js_divergence([1, 2, 3], [1, 2, 4])
        assert isinstance(result, float)

    def test_empty_returns_zero(self) -> None:
        assert js_divergence([], [1, 2, 3]) == 0.0

    def test_constant_feature_returns_zero(self) -> None:
        assert js_divergence([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0


class TestJSSeverity:
    def test_ok(self) -> None:
        assert js_severity(0.02) == "ok"

    def test_moderate(self) -> None:
        assert js_severity(0.07) == "moderate"

    def test_critical(self) -> None:
        assert js_severity(0.15) == "critical"


# ---------------------------------------------------------------------------
# TestChi2Test
# ---------------------------------------------------------------------------


class TestChi2Test:
    def test_identical_categories_high_p(self) -> None:
        cats = [0, 1, 2, 0, 1, 2] * 50
        result = chi2_test(cats, cats)
        assert result["p_value"] > 0.05, "Идентичные → p > 0.05"

    def test_shifted_categories_low_p(self) -> None:
        # Reference: равномерное; Current: сильно скошено к одной категории
        ref = [0, 1, 2] * 100
        cur = [2] * 250 + [0] * 25 + [1] * 25
        result = chi2_test(ref, cur)
        assert result["p_value"] < 0.05, "Сильный сдвиг → p < 0.05"

    def test_returns_required_keys(self) -> None:
        result = chi2_test([0, 1, 0], [1, 1, 0])
        assert "statistic" in result
        assert "p_value" in result
        assert "dof" in result

    def test_statistic_non_negative(self) -> None:
        ref = list(range(5)) * 20
        cur = list(range(5)) * 20
        result = chi2_test(ref, cur)
        assert result["statistic"] >= 0.0

    def test_single_category_dof_zero(self) -> None:
        result = chi2_test([1, 1, 1], [1, 1, 1])
        assert result["dof"] == 0

    def test_float_input_cast_to_int(self) -> None:
        ref = [0.0, 1.0, 2.0] * 30
        cur = [0.0, 1.0, 2.0] * 30
        result = chi2_test(ref, cur)
        assert result["p_value"] > 0.05


# ---------------------------------------------------------------------------
# TestExtendedDriftTest
# ---------------------------------------------------------------------------


class TestExtendedDriftTest:
    def test_continuous_no_drift(self) -> None:
        ref = RNG.normal(0, 1, 300).tolist()
        cur = RNG.normal(0, 1, 300).tolist()
        result = extended_drift_test(ref, cur, feature_type="continuous")
        assert result["feature_type"] == "continuous"
        assert "wasserstein" in result["tests"]
        assert "js_divergence" in result["tests"]
        assert result["severity"] in ("ok", "moderate", "critical")

    def test_continuous_drift_detected(self) -> None:
        ref = RNG.normal(0, 1, 500).tolist()
        cur = RNG.normal(10, 1, 500).tolist()  # явный сдвиг
        result = extended_drift_test(ref, cur, feature_type="continuous")
        assert result["drift_detected"] is True
        assert result["severity"] == "critical"

    def test_categorical_no_drift(self) -> None:
        cats = [0, 1, 2] * 100
        result = extended_drift_test(cats, cats, feature_type="categorical")
        assert result["feature_type"] == "categorical"
        assert "chi2" in result["tests"]
        assert "js_divergence" in result["tests"]

    def test_auto_detects_categorical(self) -> None:
        cats = [0, 1, 2, 3] * 50
        result = extended_drift_test(cats, cats, feature_type="auto")
        assert result["feature_type"] == "categorical"

    def test_auto_detects_continuous(self) -> None:
        ref = RNG.normal(0, 1, 500).tolist()
        cur = RNG.normal(0, 1, 500).tolist()
        result = extended_drift_test(ref, cur, feature_type="auto")
        assert result["feature_type"] == "continuous"

    def test_confidence_between_zero_and_one(self) -> None:
        ref = RNG.normal(0, 1, 200).tolist()
        cur = RNG.normal(0, 1, 200).tolist()
        result = extended_drift_test(ref, cur)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_one_when_all_drift(self) -> None:
        ref = RNG.normal(0, 0.01, 300).tolist()
        cur = RNG.normal(100, 0.01, 300).tolist()
        result = extended_drift_test(ref, cur, feature_type="continuous")
        assert result["confidence"] == 1.0


# ---------------------------------------------------------------------------
# TestBatchExtendedDrift
# ---------------------------------------------------------------------------


class TestBatchExtendedDrift:
    def test_returns_summary_fields(self) -> None:
        ref = {"a": RNG.normal(0, 1, 100).tolist()}
        cur = {"a": RNG.normal(0, 1, 100).tolist()}
        report = batch_extended_drift(ref, cur)
        assert "columns_checked" in report
        assert "columns_with_drift" in report
        assert "critical_columns" in report
        assert "overall_drift" in report
        assert "details" in report

    def test_multiple_columns(self) -> None:
        ref = {
            "feature_a": RNG.normal(0, 1, 200).tolist(),
            "feature_b": RNG.normal(5, 1, 200).tolist(),
        }
        cur = {
            "feature_a": RNG.normal(0, 1, 200).tolist(),
            "feature_b": RNG.normal(5, 1, 200).tolist(),
        }
        report = batch_extended_drift(ref, cur)
        assert report["columns_checked"] == 2

    def test_critical_columns_populated_on_drift(self) -> None:
        ref = {"x": RNG.normal(0, 0.01, 300).tolist()}
        cur = {"x": RNG.normal(100, 0.01, 300).tolist()}
        report = batch_extended_drift(ref, cur)
        assert "x" in report["critical_columns"]

    def test_feature_types_override(self) -> None:
        ref = {"cat": [0, 1, 2] * 50}
        cur = {"cat": [0, 1, 2] * 50}
        report = batch_extended_drift(ref, cur, feature_types={"cat": "categorical"})
        detail = report["details"][0]
        assert detail["feature_type"] == "categorical"

    def test_skips_missing_column(self) -> None:
        # Only column "a" is in both — "b" only in current
        ref = {"a": [1, 2, 3] * 20}
        cur = {"a": [1, 2, 3] * 20, "b": [4, 5, 6] * 20}
        report = batch_extended_drift(ref, cur)
        # "b" not in reference → not checked
        assert report["columns_checked"] == 1


# ---------------------------------------------------------------------------
# TestExtendedDriftAPIEndpoint
# ---------------------------------------------------------------------------


class TestExtendedDriftAPIEndpoint:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def _payload(
        self,
        ref: dict | None = None,
        cur: dict | None = None,
        feature_types: dict | None = None,
    ) -> dict:
        rng = np.random.default_rng(7)
        if ref is None:
            ref = {
                "amount": rng.normal(100, 10, 100).tolist(),
                "score": rng.uniform(0, 1, 100).tolist(),
            }
        if cur is None:
            cur = {
                "amount": rng.normal(100, 10, 100).tolist(),
                "score": rng.uniform(0, 1, 100).tolist(),
            }
        payload: dict = {"reference": ref, "current": cur}
        if feature_types:
            payload["feature_types"] = feature_types
        return payload

    def test_status_200(self, client: TestClient) -> None:
        resp = client.post("/drift/extended", json=self._payload())
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        resp = client.post("/drift/extended", json=self._payload())
        data = resp.json()
        assert "columns_checked" in data
        assert "columns_with_drift" in data
        assert "critical_columns" in data
        assert "overall_drift" in data
        assert "details" in data

    def test_columns_checked_count(self, client: TestClient) -> None:
        resp = client.post("/drift/extended", json=self._payload())
        assert resp.json()["columns_checked"] == 2

    def test_no_drift_on_identical_data(self, client: TestClient) -> None:
        rng = np.random.default_rng(0)
        vals = rng.normal(0, 1, 200).tolist()
        payload = {"reference": {"x": vals}, "current": {"x": vals}}
        resp = client.post("/drift/extended", json=payload)
        data = resp.json()
        assert data["overall_drift"] is False

    def test_drift_detected_on_shifted_data(self, client: TestClient) -> None:
        rng = np.random.default_rng(1)
        ref_vals = rng.normal(0, 0.01, 300).tolist()
        cur_vals = rng.normal(100, 0.01, 300).tolist()
        payload = {"reference": {"x": ref_vals}, "current": {"x": cur_vals}}
        resp = client.post("/drift/extended", json=payload)
        data = resp.json()
        assert data["overall_drift"] is True
        assert "x" in data["critical_columns"]

    def test_details_per_column_structure(self, client: TestClient) -> None:
        resp = client.post("/drift/extended", json=self._payload())
        detail = resp.json()["details"][0]
        assert "column" in detail
        assert "drift_detected" in detail
        assert "severity" in detail
        assert "tests" in detail

    def test_categorical_feature_type_override(self, client: TestClient) -> None:
        cats = list(range(3)) * 50
        payload = {
            "reference": {"cat": cats},
            "current": {"cat": cats},
            "feature_types": {"cat": "categorical"},
        }
        resp = client.post("/drift/extended", json=payload)
        assert resp.status_code == 200
        detail = resp.json()["details"][0]
        assert detail["feature_type"] == "categorical"
        assert "chi2" in detail["tests"]

    def test_custom_bins_param(self, client: TestClient) -> None:
        rng = np.random.default_rng(3)
        vals = rng.normal(0, 1, 100).tolist()
        payload = {"reference": {"x": vals}, "current": {"x": vals}, "bins": 10}
        resp = client.post("/drift/extended", json=payload)
        assert resp.status_code == 200

    def test_critical_columns_list_type(self, client: TestClient) -> None:
        resp = client.post("/drift/extended", json=self._payload())
        assert isinstance(resp.json()["critical_columns"], list)
