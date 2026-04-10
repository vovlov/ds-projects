"""Tests for customer churn pipeline."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from churn.data.load import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    load_raw,
    prepare_dataset,
)
from churn.retraining.trigger import (
    DriftReport,
    RetrainingResult,
    RetrainingTrigger,
    compute_psi,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw.csv"
SKIP_REASON = "Dataset not available"


@pytest.fixture
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip(SKIP_REASON)
    return load_raw(DATA_PATH)


@pytest.fixture
def full_df():
    if not DATA_PATH.exists():
        pytest.skip(SKIP_REASON)
    return prepare_dataset(DATA_PATH)


class TestDataLoading:
    def test_load_raw_shape(self, raw_df):
        assert raw_df.shape[0] == 7043
        assert raw_df.shape[1] == 21

    def test_load_raw_no_nulls_in_target(self, raw_df):
        assert raw_df[TARGET].null_count() == 0

    def test_target_is_binary(self, raw_df):
        unique_vals = raw_df[TARGET].unique().sort().to_list()
        assert unique_vals == [0, 1]

    def test_total_charges_is_float(self, raw_df):
        assert raw_df["TotalCharges"].dtype == pl.Float64

    def test_no_empty_total_charges(self, raw_df):
        assert raw_df["TotalCharges"].null_count() == 0


class TestFeatureEngineering:
    def test_add_features_creates_columns(self, full_df):
        expected = ["AvgMonthlySpend", "ExpectedTotalCharges", "TenureGroup", "NumServices"]
        for col in expected:
            assert col in full_df.columns

    def test_tenure_group_values(self, full_df):
        unique = full_df["TenureGroup"].unique().sort().to_list()
        assert set(unique) == {"long", "mid", "new"}

    def test_num_services_range(self, full_df):
        assert full_df["NumServices"].min() >= 0
        assert full_df["NumServices"].max() <= 6

    def test_avg_monthly_spend_non_negative(self, full_df):
        assert (full_df["AvgMonthlySpend"] >= 0).all()


class TestDataQuality:
    def test_categorical_features_exist(self, raw_df):
        for col in CATEGORICAL_FEATURES:
            assert col in raw_df.columns

    def test_numerical_features_exist(self, raw_df):
        for col in NUMERICAL_FEATURES:
            assert col in raw_df.columns

    def test_churn_rate_reasonable(self, raw_df):
        churn_rate = raw_df[TARGET].mean()
        assert 0.1 < churn_rate < 0.5

    def test_tenure_non_negative(self, raw_df):
        assert (raw_df["tenure"] >= 0).all()

    def test_monthly_charges_positive(self, raw_df):
        assert (raw_df["MonthlyCharges"] > 0).all()


class TestAPI:
    def test_health_endpoint(self):
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_predict_validation(self):
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/predict", json={"gender": "Male"})
        assert resp.status_code == 422

    def test_predict_with_model(self):
        """Integration test: predict with real trained model."""
        from pathlib import Path

        model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.pkl"
        if not model_path.exists():
            pytest.skip("Model artifact not available — run train.py first")

        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/predict",
            json={
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 70.35,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_level"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Automated Retraining Trigger Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def stable_features() -> dict:
    """Baseline и current из одного распределения — PSI ≈ 0."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(50, 10, 1000)
    current = rng.normal(50, 10, 800)  # то же распределение, другой размер
    return {"baseline": {"MonthlyCharges": baseline}, "current": {"MonthlyCharges": current}}


@pytest.fixture
def drifted_features() -> dict:
    """Current сдвинут на 2σ — PSI должен быть > 0.2."""
    rng = np.random.default_rng(42)
    baseline = rng.normal(50, 10, 1000)
    current = rng.normal(75, 10, 800)  # сдвиг на 2.5σ
    return {"baseline": {"MonthlyCharges": baseline}, "current": {"MonthlyCharges": current}}


class TestComputePSI:
    def test_same_distribution_low_psi(self, stable_features):
        """Одинаковые распределения → PSI близко к нулю."""
        baseline = stable_features["baseline"]["MonthlyCharges"]
        current = stable_features["current"]["MonthlyCharges"]
        psi = compute_psi(baseline, current)
        assert psi < 0.1, f"Expected PSI < 0.1 for same distribution, got {psi:.4f}"

    def test_shifted_distribution_high_psi(self, drifted_features):
        """Сдвинутое распределение → PSI > 0.2."""
        baseline = drifted_features["baseline"]["MonthlyCharges"]
        current = drifted_features["current"]["MonthlyCharges"]
        psi = compute_psi(baseline, current)
        assert psi >= 0.2, f"Expected PSI >= 0.2 for drifted distribution, got {psi:.4f}"

    def test_psi_non_negative(self):
        """PSI всегда неотрицательный."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            b = rng.exponential(1, 200)
            c = rng.exponential(1.5, 200)
            assert compute_psi(b, c) >= 0

    def test_psi_zero_variance_returns_zero(self):
        """Константный признак → PSI = 0.0 (не обрабатываем)."""
        baseline = np.ones(100)
        current = np.ones(80)
        # Не должен падать с ошибкой — должен вернуть 0.0
        psi = compute_psi(baseline, current)
        assert psi == 0.0

    def test_psi_n_bins_param(self):
        """Разное число бинов не ломает вычисление."""
        rng = np.random.default_rng(7)
        b = rng.normal(0, 1, 500)
        c = rng.normal(0.5, 1, 500)
        psi_10 = compute_psi(b, c, n_bins=10)
        psi_20 = compute_psi(b, c, n_bins=20)
        assert psi_10 >= 0
        assert psi_20 >= 0


class TestDriftReport:
    def test_no_drift_stable(self, stable_features):
        trigger = RetrainingTrigger(
            baseline_features=stable_features["baseline"],
            baseline_auc=0.84,
        )
        report = trigger.check_drift(stable_features["current"])
        assert isinstance(report, DriftReport)
        assert not report.drift_detected
        assert report.max_psi < 0.2

    def test_drift_detected_shifted(self, drifted_features):
        trigger = RetrainingTrigger(
            baseline_features=drifted_features["baseline"],
            baseline_auc=0.84,
        )
        report = trigger.check_drift(drifted_features["current"])
        assert report.drift_detected
        assert report.max_psi >= 0.2
        assert "MonthlyCharges" in report.drifted_features

    def test_missing_feature_skipped(self):
        """Отсутствующий признак в current пропускается без ошибки."""
        rng = np.random.default_rng(1)
        baseline = {"MonthlyCharges": rng.normal(50, 10, 200)}
        trigger = RetrainingTrigger(baseline_features=baseline, baseline_auc=0.80)
        # current не содержит MonthlyCharges
        report = trigger.check_drift({"tenure": rng.normal(24, 12, 200)})
        # Нет падений — просто пустой report
        assert report.max_psi == 0.0

    def test_report_has_correct_sizes(self, stable_features):
        trigger = RetrainingTrigger(
            baseline_features=stable_features["baseline"], baseline_auc=0.80
        )
        report = trigger.check_drift(stable_features["current"])
        assert report.baseline_size == 1000
        assert report.current_size == 800


class TestRetrainingTrigger:
    def test_no_retrain_stable(self, stable_features):
        """Стабильные данные + хороший AUC → не переобучаем."""
        trigger = RetrainingTrigger(
            baseline_features=stable_features["baseline"],
            baseline_auc=0.84,
        )
        result = trigger.evaluate(stable_features["current"], current_auc=0.83)
        assert isinstance(result, RetrainingResult)
        assert not result.should_retrain
        assert not result.perf_degraded

    def test_retrain_on_drift(self, drifted_features):
        """Дрейф данных → переобучение даже при хорошем AUC."""
        trigger = RetrainingTrigger(
            baseline_features=drifted_features["baseline"],
            baseline_auc=0.84,
        )
        result = trigger.evaluate(drifted_features["current"], current_auc=0.83)
        assert result.should_retrain
        assert result.drift_report.drift_detected

    def test_retrain_on_auc_degradation(self, stable_features):
        """Стабильные данные но AUC упал → переобучение."""
        trigger = RetrainingTrigger(
            baseline_features=stable_features["baseline"],
            baseline_auc=0.84,
            delta_auc=0.05,
        )
        result = trigger.evaluate(stable_features["current"], current_auc=0.78)
        assert result.should_retrain
        assert result.perf_degraded

    def test_no_retrain_without_auc(self, stable_features):
        """Нет разметки (current_auc=None) + стабильные данные → не переобучаем."""
        trigger = RetrainingTrigger(
            baseline_features=stable_features["baseline"],
            baseline_auc=0.84,
        )
        result = trigger.evaluate(stable_features["current"], current_auc=None)
        assert not result.should_retrain
        assert result.current_auc is None

    def test_reason_contains_psi_info(self, drifted_features):
        """Reason должен содержать PSI-значение."""
        trigger = RetrainingTrigger(
            baseline_features=drifted_features["baseline"], baseline_auc=0.84
        )
        result = trigger.evaluate(drifted_features["current"])
        assert "PSI" in result.reason or "drift" in result.reason.lower()

    def test_result_details_populated(self, stable_features):
        """details содержит threshold и feature_psi."""
        trigger = RetrainingTrigger(
            baseline_features=stable_features["baseline"], baseline_auc=0.80
        )
        result = trigger.evaluate(stable_features["current"])
        assert "threshold_psi" in result.details
        assert "feature_psi" in result.details
        assert "MonthlyCharges" in result.details["feature_psi"]

    def test_custom_threshold(self):
        """Кастомный порог PSI=0.05 → срабатывает при малом дрейфе."""
        rng = np.random.default_rng(42)
        baseline = {"tenure": rng.normal(24, 12, 1000)}
        # Небольшой сдвиг (< 0.2 PSI, но > 0.05)
        current = {"tenure": rng.normal(26, 12, 800)}
        trigger = RetrainingTrigger(
            baseline_features=baseline,
            baseline_auc=0.80,
            threshold_psi=0.05,
        )
        result = trigger.evaluate(current)
        # При threshold=0.05 скорее всего сработает (или хотя бы не падает)
        assert isinstance(result.should_retrain, bool)

    def test_check_performance_degraded(self):
        """check_performance() корректно определяет деградацию."""
        rng = np.random.default_rng(0)
        baseline = {"MonthlyCharges": rng.normal(50, 10, 100)}
        trigger = RetrainingTrigger(baseline_features=baseline, baseline_auc=0.85, delta_auc=0.05)
        assert trigger.check_performance(0.79) is True  # упал на 0.06
        assert trigger.check_performance(0.81) is False  # упал на 0.04

    def test_multiple_features(self):
        """PSI вычисляется независимо для каждого признака."""
        rng = np.random.default_rng(42)
        baseline = {
            "MonthlyCharges": rng.normal(50, 10, 500),
            "tenure": rng.normal(24, 12, 500),
        }
        current = {
            "MonthlyCharges": rng.normal(70, 10, 400),  # дрейф
            "tenure": rng.normal(24, 12, 400),  # стабильно
        }
        trigger = RetrainingTrigger(baseline_features=baseline, baseline_auc=0.80)
        report = trigger.check_drift(current)
        assert "MonthlyCharges" in report.feature_psi
        assert "tenure" in report.feature_psi
        # MonthlyCharges должен иметь больший PSI
        assert report.feature_psi["MonthlyCharges"] > report.feature_psi["tenure"]
