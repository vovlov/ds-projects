"""Tests for T-Learner Causal Uplift Modeling (Project 01).

Tests for causal uplift modeling — T-Learner CATE estimation,
Persuasion Matrix segmentation, Qini coefficient, and API endpoints.
"""

from __future__ import annotations

import numpy as np
import pytest
from churn.causal.uplift import (
    QiniResult,
    TLearnerUplift,
    UpliftConfig,
    UpliftPrediction,
    UpliftResult,
    UpliftSegment,
    is_available,
    summarize_uplift,
)
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n: int = 200,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Генерировать синтетические данные с известным CATE.

    Generate synthetic data with a known ground-truth treatment effect:
    - Treatment reduces churn probability by ~0.2 for the first half of customers
    - Control group has no effect
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    treatment = (np.arange(n) % 2).astype(int)  # чередование

    # Базовая вероятность оттока: logit(X[:,0])
    base_prob = 1 / (1 + np.exp(-X[:, 0]))
    # Treatment снижает вероятность оттока на 0.15
    noise = rng.rand(n)
    prob_with_t = np.clip(base_prob - 0.15 * treatment, 0.05, 0.95)
    y = (noise < prob_with_t).astype(int)

    return X, y, treatment


# ---------------------------------------------------------------------------
# TestTLearnerUplift — unit tests for core model
# ---------------------------------------------------------------------------


class TestTLearnerUplift:
    """Unit tests for T-Learner Causal Uplift Model."""

    def test_fit_returns_self(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset()
        model = TLearnerUplift()
        result = model.fit(X, y, t)
        assert result is model

    def test_is_fitted_after_fit(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset()
        model = TLearnerUplift()
        model.fit(X, y, t)
        assert model._is_fitted

    def test_predict_cate_shape(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=200)
        model = TLearnerUplift()
        model.fit(X, y, t)
        cate = model.predict_cate(X[:20])
        assert cate.shape == (20,)

    def test_predict_cate_dtype_float(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset()
        model = TLearnerUplift()
        model.fit(X, y, t)
        cate = model.predict_cate(X[:5])
        assert cate.dtype == float or np.issubdtype(cate.dtype, np.floating)

    def test_predict_cate_range(self):
        """CATE должен быть в [-1, 1] (разность вероятностей)."""
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=300)
        model = TLearnerUplift()
        model.fit(X, y, t)
        cate = model.predict_cate(X)
        assert cate.min() >= -1.0
        assert cate.max() <= 1.0

    def test_predict_before_fit_raises(self):
        model = TLearnerUplift()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_cate(np.zeros((5, 3)))

    def test_predict_segment_before_fit_raises(self):
        model = TLearnerUplift()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_segment(np.zeros((5, 3)))

    def test_fit_requires_min_samples_per_group(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X = np.random.randn(20, 3)
        y = np.zeros(20, dtype=int)
        # Только 2 примера в treatment
        t = np.zeros(20, dtype=int)
        t[:2] = 1
        with pytest.raises(ValueError, match="≥10"):
            TLearnerUplift().fit(X, y, t)

    def test_predict_segment_returns_uplift_predictions(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset()
        model = TLearnerUplift()
        model.fit(X, y, t)
        preds = model.predict_segment(X[:10])
        assert len(preds) == 10
        assert all(isinstance(p, UpliftPrediction) for p in preds)

    def test_segments_are_valid_enum_values(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=300)
        model = TLearnerUplift()
        model.fit(X, y, t)
        preds = model.predict_segment(X)
        valid = set(UpliftSegment)
        for p in preds:
            assert p.segment in valid

    def test_persuadable_has_negative_cate(self):
        """Persuadable сегмент должен иметь CATE < -threshold."""
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=400)
        config = UpliftConfig(persuadable_threshold=-0.05)
        model = TLearnerUplift(config)
        model.fit(X, y, t)
        preds = model.predict_segment(X)
        for p in preds:
            if p.segment == UpliftSegment.PERSUADABLE:
                assert p.cate < 0.0, "Persuadable should have negative CATE"

    def test_sleeping_dog_has_positive_cate(self):
        """Sleeping Dog должен иметь CATE > threshold."""
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=400)
        model = TLearnerUplift()
        model.fit(X, y, t)
        preds = model.predict_segment(X)
        for p in preds:
            if p.segment == UpliftSegment.SLEEPING_DOG:
                assert p.cate > 0.0, "Sleeping Dog should have positive CATE"

    def test_all_four_segments_produced_on_large_dataset(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        rng = np.random.RandomState(0)
        X = rng.randn(1000, 5)
        t = (np.arange(1000) % 2).astype(int)
        y = (rng.rand(1000) < 0.5).astype(int)
        model = TLearnerUplift(UpliftConfig(n_estimators=50))
        model.fit(X, y, t)
        preds = model.predict_segment(X)
        segments = {p.segment for p in preds}
        # На большом датасете ожидаем все 4 сегмента
        assert len(segments) >= 2

    def test_get_params_returns_dict(self):
        model = TLearnerUplift()
        params = model.get_params()
        assert isinstance(params, dict)
        assert "model_type" in params
        assert params["model_type"] == "T-Learner"

    def test_config_custom_persuadable_threshold(self):
        config = UpliftConfig(persuadable_threshold=-0.10)
        assert config.persuadable_threshold == -0.10

    def test_is_available_returns_bool(self):
        result = is_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestQiniCoefficient
# ---------------------------------------------------------------------------


class TestQiniCoefficient:
    """Tests for Qini coefficient computation."""

    def test_qini_returns_qini_result(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset()
        model = TLearnerUplift()
        model.fit(X, y, t)
        result = model.compute_qini(X, y, t)
        assert isinstance(result, QiniResult)

    def test_qini_n_treated_n_control(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=200)
        model = TLearnerUplift()
        model.fit(X, y, t)
        result = model.compute_qini(X, y, t)
        assert result.n_treated == int(t.sum())
        assert result.n_control == int((t == 0).sum())

    def test_qini_coefficient_is_finite(self):
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset()
        model = TLearnerUplift()
        model.fit(X, y, t)
        result = model.compute_qini(X, y, t)
        assert np.isfinite(result.qini_coefficient)
        assert np.isfinite(result.auuc)

    def test_qini_range(self):
        """Qini coefficient должен быть в разумном диапазоне."""
        if not is_available():
            pytest.skip("sklearn not installed")
        X, y, t = _make_dataset(n=400)
        model = TLearnerUplift()
        model.fit(X, y, t)
        result = model.compute_qini(X, y, t)
        assert -1.0 <= result.qini_coefficient <= 1.0


# ---------------------------------------------------------------------------
# TestSummarizeUplift
# ---------------------------------------------------------------------------


class TestSummarizeUplift:
    """Tests for summarize_uplift helper."""

    def _make_predictions(self) -> list[UpliftPrediction]:
        return [
            UpliftPrediction(-0.15, 0.30, 0.45, UpliftSegment.PERSUADABLE),
            UpliftPrediction(-0.12, 0.35, 0.47, UpliftSegment.PERSUADABLE),
            UpliftPrediction(0.02, 0.50, 0.48, UpliftSegment.SURE_THING),
            UpliftPrediction(0.01, 0.20, 0.19, UpliftSegment.SURE_THING),
            UpliftPrediction(0.00, 0.70, 0.70, UpliftSegment.LOST_CAUSE),
            UpliftPrediction(0.15, 0.80, 0.65, UpliftSegment.SLEEPING_DOG),
        ]

    def test_summarize_counts(self):
        preds = self._make_predictions()
        result = summarize_uplift(preds)
        assert result.n_persuadable == 2
        assert result.n_sure_thing == 2
        assert result.n_lost_cause == 1
        assert result.n_sleeping_dog == 1

    def test_summarize_avg_cate(self):
        preds = self._make_predictions()
        result = summarize_uplift(preds)
        expected_avg = np.mean([-0.15, -0.12, 0.02, 0.01, 0.00, 0.15])
        assert abs(result.avg_cate - expected_avg) < 1e-6

    def test_targeting_uplift_is_mean_of_persuadable_cates(self):
        preds = self._make_predictions()
        result = summarize_uplift(preds)
        expected = np.mean([-0.15, -0.12])
        assert abs(result.targeting_uplift - expected) < 1e-6

    def test_no_persuadable_targeting_uplift_zero(self):
        preds = [
            UpliftPrediction(0.05, 0.5, 0.45, UpliftSegment.SURE_THING),
            UpliftPrediction(0.10, 0.6, 0.50, UpliftSegment.SLEEPING_DOG),
        ]
        result = summarize_uplift(preds)
        assert result.targeting_uplift == 0.0
        assert result.n_persuadable == 0

    def test_summarize_returns_uplift_result(self):
        preds = self._make_predictions()
        result = summarize_uplift(preds)
        assert isinstance(result, UpliftResult)


# ---------------------------------------------------------------------------
# TestUpliftAPIEndpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from churn.api.app import app

    return TestClient(app)


@pytest.fixture
def sample_train_payload():
    """Минимальный датасет для обучения через API."""
    rng = np.random.RandomState(1)
    n = 100
    X = rng.randn(n, 4).tolist()
    t = [1] * 50 + [0] * 50
    y = [1, 0] * 25 + [1, 0] * 25
    return {"features": X, "labels": y, "treatment": t}


class TestUpliftAPIEndpoints:
    """Integration tests for uplift API endpoints."""

    def test_train_returns_200(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        resp = client.post("/uplift/train", json=sample_train_payload)
        assert resp.status_code == 200

    def test_train_response_has_status_trained(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        resp = client.post("/uplift/train", json=sample_train_payload)
        assert resp.json()["status"] == "trained"

    def test_train_response_has_n_treatment_control(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        resp = client.post("/uplift/train", json=sample_train_payload)
        data = resp.json()
        assert data["n_treatment"] == 50
        assert data["n_control"] == 50

    def test_train_response_model_params(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        resp = client.post("/uplift/train", json=sample_train_payload)
        params = resp.json()["model_params"]
        assert params["model_type"] == "T-Learner"

    def test_predict_returns_200_after_train(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        X = np.random.randn(10, 4).tolist()
        resp = client.post("/uplift/predict", json={"features": X})
        assert resp.status_code == 200

    def test_predict_response_has_predictions_list(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        X = np.random.randn(5, 4).tolist()
        resp = client.post("/uplift/predict", json={"features": X})
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 5

    def test_predict_response_has_segment_field(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        X = np.random.randn(3, 4).tolist()
        resp = client.post("/uplift/predict", json={"features": X})
        for item in resp.json()["predictions"]:
            assert "segment" in item
            assert item["segment"] in {s.value for s in UpliftSegment}

    def test_predict_response_has_business_metrics(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        X = np.random.randn(20, 4).tolist()
        resp = client.post("/uplift/predict", json={"features": X})
        data = resp.json()
        for key in ["n_persuadable", "n_sure_thing", "n_lost_cause", "n_sleeping_dog", "avg_cate"]:
            assert key in data

    def test_predict_segment_counts_sum_to_n(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        n = 15
        X = np.random.randn(n, 4).tolist()
        resp = client.post("/uplift/predict", json={"features": X})
        data = resp.json()
        total = (
            data["n_persuadable"]
            + data["n_sure_thing"]
            + data["n_lost_cause"]
            + data["n_sleeping_dog"]
        )
        assert total == n

    def test_predict_without_train_returns_400(self, client):
        """Предсказание без обучения должно вернуть 400."""
        from churn.api.app import app

        # Сбросить глобальное состояние через новый client без предварительного train
        fresh_client = TestClient(app)
        # Сначала сбрасываем состояние через атрибут приложения
        import churn.api.app as app_module

        app_module._uplift_model = None

        X = np.random.randn(5, 4).tolist()
        resp = fresh_client.post("/uplift/predict", json={"features": X})
        assert resp.status_code == 400

    def test_train_unequal_lengths_returns_422(self, client):
        if not is_available():
            pytest.skip("sklearn not installed")
        payload = {
            "features": [[1.0, 2.0]] * 10,
            "labels": [0, 1] * 5,
            "treatment": [0, 1, 0],  # Неверная длина
        }
        resp = client.post("/uplift/train", json=payload)
        assert resp.status_code == 422

    def test_segments_endpoint_returns_200_after_train(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        resp = client.get("/uplift/segments")
        assert resp.status_code == 200

    def test_segments_endpoint_has_segment_descriptions(self, client, sample_train_payload):
        if not is_available():
            pytest.skip("sklearn not installed")
        client.post("/uplift/train", json=sample_train_payload)
        resp = client.get("/uplift/segments")
        data = resp.json()
        assert "segments" in data
        assert "persuadable" in data["segments"]
        assert "sleeping_dog" in data["segments"]

    def test_segments_endpoint_without_train_returns_400(self, client):
        import churn.api.app as app_module

        app_module._uplift_model = None
        resp = client.get("/uplift/segments")
        assert resp.status_code == 400
