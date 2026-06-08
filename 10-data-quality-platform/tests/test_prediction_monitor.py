"""
Тесты для Prediction Distribution Monitor.
Tests for Prediction Distribution Monitor (concept drift detection).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.monitoring.prediction_monitor import (
    PredictionMonitor,
    PredictionStats,
    _compute_hist,
    _psi,
    _welch_z,
    reset_prediction_monitor,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_monitor():
    """Изолируем каждый тест — сбрасываем глобальный singleton."""
    reset_prediction_monitor()
    yield
    reset_prediction_monitor()


def _stable_preds(n: int = 300, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, n).tolist()


def _drifted_preds(n: int = 300, seed: int = 42) -> list[float]:
    """Полностью сдвинутое распределение — все предсказания близко к 1."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.8, 1.0, n).tolist()


# ---------------------------------------------------------------------------
# Unit: _compute_hist
# ---------------------------------------------------------------------------


class TestComputeHist:
    def test_length(self):
        h = _compute_hist(np.array([0.1, 0.5, 0.9]))
        assert len(h) == 10

    def test_sums_to_one(self):
        rng = np.random.default_rng(0)
        h = _compute_hist(rng.uniform(0, 1, 500))
        assert abs(h.sum() - 1.0) < 1e-9

    def test_clips_out_of_range(self):
        """Значения вне [0, 1] должны клипироваться, не вызывать ошибку."""
        h = _compute_hist(np.array([-0.5, 0.5, 1.5]))
        assert len(h) == 10
        assert abs(h.sum() - 1.0) < 1e-9

    def test_empty_returns_uniform(self):
        h = _compute_hist(np.array([]))
        assert len(h) == 10
        assert abs(h.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Unit: _psi
# ---------------------------------------------------------------------------


class TestPSI:
    def test_identical_distributions_zero(self):
        h = np.ones(10) / 10
        assert _psi(h.copy(), h.copy()) < 0.01

    def test_extreme_drift_high(self):
        ref = np.ones(10) / 10
        cur = np.zeros(10)
        cur[0] = 1.0
        psi = _psi(ref, cur)
        assert psi > 0.2

    def test_symmetry_not_required(self):
        """PSI не симметричен по определению — проверяем, что оба значения конечны."""
        rng = np.random.default_rng(7)
        a = rng.dirichlet(np.ones(10))
        b = rng.dirichlet(np.ones(10))
        psi_ab = _psi(a, b)
        psi_ba = _psi(b, a)
        assert np.isfinite(psi_ab)
        assert np.isfinite(psi_ba)


# ---------------------------------------------------------------------------
# Unit: _welch_z
# ---------------------------------------------------------------------------


class TestWelchZ:
    def test_zero_when_equal(self):
        z = _welch_z(0.5, 0.5, 0.1, 0.1, 100, 100)
        assert z == 0.0

    def test_nonzero_on_shift(self):
        z = _welch_z(0.8, 0.5, 0.1, 0.1, 100, 100)
        assert abs(z) > 1.96

    def test_zero_variance_safe(self):
        z = _welch_z(0.5, 0.5, 0.0, 0.0, 100, 100)
        assert z == 0.0


# ---------------------------------------------------------------------------
# Unit: PredictionMonitor core
# ---------------------------------------------------------------------------


class TestPredictionMonitorCore:
    def test_observe_returns_count(self):
        mon = PredictionMonitor(min_reference_size=10)
        n = mon.observe([0.1, 0.2, 0.3])
        assert n == 3

    def test_empty_observe_returns_zero(self):
        mon = PredictionMonitor(min_reference_size=10)
        assert mon.observe([]) == 0

    def test_reference_not_ready_initially(self):
        mon = PredictionMonitor(min_reference_size=100)
        mon.observe([0.5] * 50)
        assert mon.get_status().is_ready is False

    def test_reference_auto_set_after_min_size(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe([0.5] * 60)
        assert mon.get_status().is_ready is True

    def test_detect_raises_before_reference(self):
        mon = PredictionMonitor(min_reference_size=100)
        mon.observe([0.5] * 20)
        with pytest.raises(ValueError, match="Reference window"):
            mon.detect_drift()

    def test_detect_raises_too_few_current(self):
        mon = PredictionMonitor(min_reference_size=10)
        mon.observe([0.5] * 15)  # установит reference (≥10)
        mon._current.clear()  # принудительно очищаем текущее окно
        mon._current.append(0.5)  # только 1 наблюдение
        with pytest.raises(ValueError, match="Too few"):
            mon.detect_drift()

    def test_detect_no_drift_stable(self):
        mon = PredictionMonitor(min_reference_size=50)
        stable = _stable_preds(400)
        mon.observe(stable[:200])  # устанавливает reference
        mon.observe(stable[200:])  # current (похожее распределение)
        result = mon.detect_drift()
        assert result.severity in ("ok", "warning")  # стабильное → не critical

    def test_detect_critical_drift(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(200))  # reference: uniform [0.1, 0.9]
        mon.observe(_drifted_preds(200))  # current: [0.8, 1.0]
        result = mon.detect_drift()
        assert result.has_drift is True
        assert result.severity == "critical"
        assert result.psi >= 0.2

    def test_drift_result_fields(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(200))
        mon.observe(_drifted_preds(200))
        result = mon.detect_drift()
        d = result.to_dict()
        for key in (
            "drift_id",
            "timestamp",
            "has_drift",
            "severity",
            "psi",
            "z_score_mean",
            "rate_delta",
            "reason",
            "reference_stats",
            "current_stats",
        ):
            assert key in d

    def test_stats_fields(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(200))
        mon.observe(_stable_preds(200, seed=1))
        result = mon.detect_drift()
        ref = result.reference_stats.to_dict()
        for key in (
            "n",
            "mean",
            "std",
            "min",
            "max",
            "positive_rate",
            "histogram",
            "histogram_bins",
        ):
            assert key in ref

    def test_histogram_bins_count(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(200))
        mon.observe(_stable_preds(200, seed=1))
        result = mon.detect_drift()
        assert len(result.reference_stats.hist) == 10

    def test_rate_delta_positive_on_drift(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.set_reference([0.1] * 200)  # reference явно: positive_rate = 0
        mon.observe([0.9] * 200)  # current: positive_rate = 1
        result = mon.detect_drift()
        assert result.rate_delta > 0.8

    def test_status_total_observed(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe([0.5] * 100)
        mon.observe([0.5] * 50)
        assert mon.get_status().total_observed == 150

    def test_status_last_drift_check_none_before(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(200))
        assert mon.get_status().last_drift_check is None

    def test_status_last_drift_check_set_after(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(200))
        mon.observe(_stable_preds(200, seed=1))
        mon.detect_drift()
        assert mon.get_status().last_drift_check is not None

    def test_reset_clears_state(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(300))
        mon.reset()
        status = mon.get_status()
        assert status.is_ready is False
        assert status.total_observed == 0

    def test_set_reference_explicit(self):
        mon = PredictionMonitor(min_reference_size=200)
        ref_stats = mon.set_reference([0.2, 0.4, 0.6, 0.8])
        assert mon.get_status().is_ready is True
        assert isinstance(ref_stats, PredictionStats)

    def test_set_reference_clears_current(self):
        mon = PredictionMonitor(min_reference_size=50)
        mon.observe(_stable_preds(300))
        mon.set_reference([0.5] * 100)
        assert mon.get_status().current_window_size == 0

    def test_window_size_respected(self):
        """Скользящее окно не превышает window_size."""
        mon = PredictionMonitor(window_size=100, min_reference_size=50)
        mon.observe([0.5] * 500)
        assert mon.get_status().current_window_size <= 100

    def test_set_reference_requires_min_two(self):
        mon = PredictionMonitor()
        with pytest.raises(ValueError):
            mon.set_reference([0.5])


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestPredictionMonitorAPI:
    def test_observe_201(self):
        resp = client.post(
            "/predictions/observe",
            json={"predictions": [0.1, 0.5, 0.9], "min_reference_size": 10},
        )
        assert resp.status_code == 201

    def test_observe_structure(self):
        resp = client.post(
            "/predictions/observe",
            json={"predictions": [0.5] * 50, "min_reference_size": 10},
        )
        data = resp.json()
        assert "n_added" in data
        assert "status" in data
        assert data["n_added"] == 50

    def test_observe_empty_422(self):
        resp = client.post(
            "/predictions/observe",
            json={"predictions": [], "min_reference_size": 10},
        )
        assert resp.status_code == 422

    def test_set_reference_201(self):
        resp = client.post(
            "/predictions/reference",
            json={"predictions": [0.2, 0.4, 0.6, 0.8, 0.5]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "reference_stats" in data

    def test_drift_400_before_observe(self):
        resp = client.get("/predictions/drift")
        assert resp.status_code == 400

    def test_drift_400_insufficient_observations(self):
        client.post(
            "/predictions/observe",
            json={"predictions": [0.5] * 5, "min_reference_size": 10},
        )
        resp = client.get("/predictions/drift")
        assert resp.status_code == 400

    def test_drift_200_after_sufficient_observations(self):
        # Устанавливаем reference через явный endpoint
        client.post(
            "/predictions/reference",
            json={"predictions": _stable_preds(200)},
        )
        client.post(
            "/predictions/observe",
            json={"predictions": _stable_preds(50, seed=1), "min_reference_size": 10},
        )
        resp = client.get("/predictions/drift")
        assert resp.status_code == 200

    def test_drift_response_structure(self):
        client.post(
            "/predictions/reference",
            json={"predictions": _stable_preds(200)},
        )
        client.post(
            "/predictions/observe",
            json={"predictions": _stable_preds(50, seed=1), "min_reference_size": 10},
        )
        data = client.get("/predictions/drift").json()
        for key in (
            "drift_id",
            "timestamp",
            "has_drift",
            "severity",
            "psi",
            "z_score_mean",
            "rate_delta",
            "reason",
            "reference_stats",
            "current_stats",
        ):
            assert key in data

    def test_drift_detects_critical(self):
        client.post(
            "/predictions/reference",
            json={"predictions": [0.1] * 300},
        )
        client.post(
            "/predictions/observe",
            json={"predictions": [0.95] * 300, "min_reference_size": 10},
        )
        data = client.get("/predictions/drift").json()
        assert data["has_drift"] is True
        assert data["severity"] == "critical"

    def test_status_endpoint(self):
        resp = client.get("/predictions/status")
        assert resp.status_code == 200
        data = resp.json()
        for key in (
            "is_ready",
            "reference_size",
            "current_window_size",
            "window_capacity",
            "total_observed",
            "last_drift_check",
        ):
            assert key in data

    def test_status_ready_false_initially(self):
        data = client.get("/predictions/status").json()
        assert data["is_ready"] is False

    def test_status_ready_true_after_reference(self):
        client.post(
            "/predictions/reference",
            json={"predictions": [0.5] * 50},
        )
        data = client.get("/predictions/status").json()
        assert data["is_ready"] is True

    def test_reset_endpoint(self):
        client.post(
            "/predictions/reference",
            json={"predictions": [0.5] * 50},
        )
        resp = client.post("/predictions/reset")
        assert resp.status_code == 200
        data = client.get("/predictions/status").json()
        assert data["is_ready"] is False

    def test_full_cycle(self):
        """Полный цикл: observe → drift OK → наблюдаем дрейф → drift CRITICAL."""
        # 1. Устанавливаем reference (стабильные предсказания)
        client.post(
            "/predictions/reference",
            json={"predictions": list(np.linspace(0.2, 0.8, 300))},
        )
        # 2. Первая проверка — добавляем похожие данные
        client.post(
            "/predictions/observe",
            json={"predictions": list(np.linspace(0.25, 0.75, 100)), "min_reference_size": 10},
        )
        result1 = client.get("/predictions/drift").json()
        assert result1["severity"] in ("ok", "warning")

        # 3. Сброс current и добавляем сдвинутые данные
        client.post("/predictions/reset")
        client.post(
            "/predictions/reference",
            json={"predictions": list(np.linspace(0.2, 0.8, 300))},
        )
        client.post(
            "/predictions/observe",
            json={"predictions": [0.99] * 300, "min_reference_size": 10},
        )
        result2 = client.get("/predictions/drift").json()
        assert result2["has_drift"] is True
        assert result2["severity"] == "critical"
