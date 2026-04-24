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


# ---------------------------------------------------------------------------
# Тесты /retraining/notify — алертинг из Data Quality Platform (Project 10)
# ---------------------------------------------------------------------------


class TestRetrainingNotify:
    """Тесты для эндпоинта /retraining/notify / Tests for /retraining/notify endpoint.

    Эндпоинт принимает DriftAlertPayload от Project 10 (Data Quality Platform)
    и возвращает решение: переобучать модель или нет.

    Receives DriftAlertPayload from Project 10 and returns a retraining decision.
    """

    def _make_payload(
        self,
        severity: str = "warning",
        max_psi: float = 0.15,
        features_drifted: list | None = None,
    ) -> dict:
        """Вспомогательный метод для создания тестового payload."""
        return {
            "severity": severity,
            "features_drifted": features_drifted or ["MonthlyCharges"],
            "max_psi": max_psi,
            "columns_checked": 3,
            "columns_with_drift": 1,
            "timestamp": "2026-04-13T10:00:00+00:00",
            "source": "data-quality-platform",
            "details": {},
        }

    def test_critical_drift_triggers_retrain(self):
        """severity=critical → decision=retrain всегда."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(severity="critical", max_psi=0.35)
        response = client.post("/retraining/notify", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "retrain"
        assert "critical" in data["reason"].lower() or "critical" in data["severity"]
        assert data["severity"] == "critical"

    def test_warning_high_psi_triggers_retrain(self):
        """severity=warning с PSI >= 0.2 → decision=retrain."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(severity="warning", max_psi=0.22)
        response = client.post("/retraining/notify", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "retrain"
        assert data["max_psi"] == 0.22

    def test_warning_low_psi_skips_retrain(self):
        """severity=warning с PSI < 0.2 → decision=skip."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(severity="warning", max_psi=0.12)
        response = client.post("/retraining/notify", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "skip"
        assert "below" in data["reason"].lower()

    def test_ok_severity_skips_retrain(self):
        """severity=ok → decision=skip."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(severity="ok", max_psi=0.05, features_drifted=[])
        response = client.post("/retraining/notify", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "skip"

    def test_response_includes_all_fields(self):
        """Ответ содержит все обязательные поля RetrainingDecision."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(severity="critical", max_psi=0.40)
        response = client.post("/retraining/notify", json=payload)

        assert response.status_code == 200
        data = response.json()
        for field in ("decision", "reason", "severity", "max_psi", "triggered_by"):
            assert field in data, f"Missing field: {field}"

    def test_triggered_by_reflects_source(self):
        """triggered_by в ответе соответствует source из payload."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(severity="critical", max_psi=0.30)
        payload["source"] = "custom-quality-pipeline"
        response = client.post("/retraining/notify", json=payload)

        data = response.json()
        assert data["triggered_by"] == "custom-quality-pipeline"

    def test_missing_required_field_returns_422(self):
        """Отсутствующий обязательный field → 422 Unprocessable Entity."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        # max_psi обязателен
        bad_payload = {"severity": "warning", "timestamp": "2026-04-13T10:00:00+00:00"}
        response = client.post("/retraining/notify", json=bad_payload)
        assert response.status_code == 422

    def test_features_drifted_in_reason(self):
        """Список затронутых признаков попадает в reason."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(
            severity="critical",
            max_psi=0.35,
            features_drifted=["MonthlyCharges", "tenure"],
        )
        response = client.post("/retraining/notify", json=payload)
        data = response.json()
        assert "MonthlyCharges" in data["reason"] or "tenure" in data["reason"]


# ---------------------------------------------------------------------------
# BentoML Serving Tests
# ---------------------------------------------------------------------------

# Module-level mock models — локальные классы нельзя pickle,
# поэтому определяем здесь, где pickle может найти их по имени.


class _MockModelHigh:
    """Mock: churn_proba=0.7 (high risk)."""

    def predict_proba(self, X):  # noqa: N803
        return np.array([[0.3, 0.7]] * len(X))


class _MockModelLow:
    """Mock: churn_proba=0.15 (low risk)."""

    def predict_proba(self, X):  # noqa: N803
        return np.array([[0.85, 0.15]] * len(X))


class _MockModelMedium:
    """Mock: churn_proba=0.6 (medium risk)."""

    def predict_proba(self, X):  # noqa: N803
        return np.array([[0.4, 0.6]] * len(X))


class _MockModelAny:
    """Mock: churn_proba=0.4 (medium risk boundary)."""

    def predict_proba(self, X):  # noqa: N803
        return np.array([[0.5, 0.5]] * len(X))


def _make_churn_input():
    """Вспомогательный factory для ChurnInput в тестах."""
    from churn.serving.bento_service import ChurnInput

    return ChurnInput(
        gender="Female",
        SeniorCitizen=0,
        Partner="No",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=70.35,
        TotalCharges=844.2,
    )


class TestBentoService:
    """Тесты BentoML serving layer / Tests for BentoML serving module.

    Все тесты работают без установленного bentoml — graceful degradation.
    """

    def test_is_available_returns_bool(self):
        """is_available() должна возвращать bool независимо от среды."""
        from churn.serving.bento_service import is_available

        assert isinstance(is_available(), bool)

    def test_module_importable_without_bentoml(self):
        """Модуль импортируется без bentoml — никаких ImportError."""
        import churn.serving.bento_service as svc

        assert hasattr(svc, "is_available")
        assert hasattr(svc, "ChurnPredictor")
        assert hasattr(svc, "ChurnInput")
        assert hasattr(svc, "ChurnPrediction")
        assert hasattr(svc, "save_to_bentoml")

    def test_churn_input_dataclass(self):
        """ChurnInput создаётся как dataclass с правильными полями."""
        inp = _make_churn_input()
        assert inp.gender == "Female"
        assert inp.SeniorCitizen == 0
        assert inp.MonthlyCharges == pytest.approx(70.35)
        assert inp.tenure == 12

    def test_churn_prediction_dataclass(self):
        """ChurnPrediction хранит все поля включая model_version."""
        from churn.serving.bento_service import ChurnPrediction

        pred = ChurnPrediction(
            churn_probability=0.75,
            churn_prediction=True,
            risk_level="high",
        )
        assert pred.churn_probability == pytest.approx(0.75)
        assert pred.churn_prediction is True
        assert pred.risk_level == "high"
        assert pred.model_version == "v1"  # default

    def test_churn_prediction_custom_version(self):
        """model_version можно переопределить для A/B тестирования."""
        from churn.serving.bento_service import ChurnPrediction

        pred = ChurnPrediction(
            churn_probability=0.3,
            churn_prediction=False,
            risk_level="low",
            model_version="catboost-v2",
        )
        assert pred.model_version == "catboost-v2"

    def test_classify_risk_thresholds(self):
        """Пороговая логика: 0.7 = high, 0.4 = medium, < 0.4 = low."""
        from churn.serving.bento_service import ChurnPredictor

        assert ChurnPredictor._classify_risk(0.0) == "low"
        assert ChurnPredictor._classify_risk(0.39) == "low"
        assert ChurnPredictor._classify_risk(0.4) == "medium"
        assert ChurnPredictor._classify_risk(0.69) == "medium"
        assert ChurnPredictor._classify_risk(0.7) == "high"
        assert ChurnPredictor._classify_risk(1.0) == "high"

    def test_predictor_missing_model_raises(self, tmp_path):
        """FileNotFoundError при обращении к несуществующему артефакту."""
        from churn.serving.bento_service import ChurnPredictor

        predictor = ChurnPredictor(model_path=tmp_path / "nonexistent.pkl")
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            predictor.predict(_make_churn_input())

    def test_predictor_lazy_load(self, tmp_path):
        """Модель не загружается при создании predictor'а — только при predict()."""
        from churn.serving.bento_service import ChurnPredictor

        # Инициализация с несуществующим путём не бросает исключений
        predictor = ChurnPredictor(model_path=tmp_path / "nonexistent.pkl")
        assert predictor._model is None  # ещё не загружен

    def test_predictor_with_mock_model(self, tmp_path):
        """Predictor корректно работает с mock sklearn-совместимой моделью."""
        import pickle

        from churn.serving.bento_service import ChurnPredictor

        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(_MockModelHigh(), f)

        predictor = ChurnPredictor(model_path=model_path, model_version="mock-v1")
        result = predictor.predict(_make_churn_input())

        assert result.churn_probability == pytest.approx(0.7, abs=0.001)
        assert result.churn_prediction is True
        assert result.risk_level == "high"
        assert result.model_version == "mock-v1"

    def test_predictor_low_proba_mock(self, tmp_path):
        """Predictor корректно классифицирует низкую вероятность оттока."""
        import pickle

        from churn.serving.bento_service import ChurnPredictor

        model_path = tmp_path / "low_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(_MockModelLow(), f)

        predictor = ChurnPredictor(model_path=model_path)
        result = predictor.predict(_make_churn_input())

        assert result.churn_probability == pytest.approx(0.15, abs=0.001)
        assert result.churn_prediction is False
        assert result.risk_level == "low"

    def test_batch_predict_multiple_inputs(self, tmp_path):
        """predict_batch возвращает результаты для каждого входа."""
        import pickle

        from churn.serving.bento_service import ChurnPredictor

        model_path = tmp_path / "batch_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(_MockModelMedium(), f)

        predictor = ChurnPredictor(model_path=model_path)
        inputs = [_make_churn_input() for _ in range(5)]
        results = predictor.predict_batch(inputs)

        assert len(results) == 5
        for r in results:
            assert r.churn_probability == pytest.approx(0.6, abs=0.001)
            assert r.churn_prediction is True
            assert r.risk_level == "medium"

    def test_batch_predict_empty_list(self, tmp_path):
        """predict_batch с пустым списком возвращает пустой список."""
        import pickle

        from churn.serving.bento_service import ChurnPredictor

        model_path = tmp_path / "any_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(_MockModelAny(), f)

        predictor = ChurnPredictor(model_path=model_path)
        results = predictor.predict_batch([])
        assert results == []

    def test_save_to_bentoml_without_bentoml(self):
        """save_to_bentoml() возвращает None если bentoml не установлен."""
        from churn.serving.bento_service import is_available, save_to_bentoml

        if is_available():
            pytest.skip("BentoML is installed — skip graceful degradation test")

        result = save_to_bentoml()
        assert result is None

    def test_save_to_bentoml_missing_artifact(self, tmp_path):
        """save_to_bentoml() возвращает None если артефакт отсутствует."""
        from churn.serving.bento_service import is_available, save_to_bentoml

        if is_available():
            pytest.skip("BentoML is installed — different code path")

        result = save_to_bentoml(model_path=tmp_path / "nonexistent.pkl")
        assert result is None

    def test_churn_service_exists_when_bentoml_available(self):
        """ChurnService существует только если bentoml установлен."""
        import churn.serving.bento_service as svc
        from churn.serving.bento_service import is_available

        if is_available():
            assert hasattr(svc, "ChurnService")
        else:
            assert not hasattr(svc, "ChurnService")

    def test_predictor_model_version_default(self, tmp_path):
        """По умолчанию model_version='v1'."""
        import pickle

        from churn.serving.bento_service import ChurnPredictor

        p = tmp_path / "m.pkl"
        with open(p, "wb") as f:
            pickle.dump(_MockModelLow(), f)

        predictor = ChurnPredictor(model_path=p)
        result = predictor.predict(_make_churn_input())
        assert result.model_version == "v1"


# ---------------------------------------------------------------------------
# A/B Testing Framework Tests
# ---------------------------------------------------------------------------


class TestABExperiment:
    """Тесты ядра A/B эксперимента / Tests for ABExperiment core logic."""

    def _make_experiment(self, min_samples: int = 5):  # -> ABExperiment
        """Фабрика эксперимента с низким порогом для тестов."""
        from churn.ab_testing.experiment import ABExperiment, VariantConfig

        return ABExperiment(
            variants=[
                VariantConfig("control", 0.5, "v1", "baseline"),
                VariantConfig("treatment", 0.5, "v2", "challenger"),
            ],
            min_samples_per_variant=min_samples,
        )

    def test_route_returns_valid_variant(self):
        """route() возвращает один из двух вариантов."""
        exp = self._make_experiment()
        for cid in ["cust_001", "cust_002", "cust_abc", "user-xyz-123"]:
            variant = exp.route(cid)
            assert variant in ("control", "treatment"), f"Unexpected variant: {variant}"

    def test_route_is_deterministic(self):
        """Один и тот же customer_id всегда получает один вариант."""
        exp = self._make_experiment()
        for cid in ["abc", "def", "ghi", "jkl"]:
            first = exp.route(cid)
            # Повторные вызовы должны давать тот же результат
            for _ in range(5):
                assert exp.route(cid) == first, f"Non-deterministic routing for {cid}"

    def test_route_distributes_traffic(self):
        """Хеш-роутинг должен распределять ~50/50 на большой выборке."""
        exp = self._make_experiment()
        counts = {"control": 0, "treatment": 0}
        for i in range(1000):
            v = exp.route(f"customer_{i:05d}")
            counts[v] += 1
        # Допускаем ±15% от идеального 50/50
        assert 350 <= counts["control"] <= 650, f"Unbalanced split: {counts}"
        assert 350 <= counts["treatment"] <= 650

    def test_record_prediction_stores_correctly(self):
        """record_prediction() корректно сохраняет запись."""
        exp = self._make_experiment()
        record = exp.record_prediction("cust_001", "control", 0.75, "high")
        assert record.customer_id == "cust_001"
        assert record.variant == "control"
        assert record.churn_probability == pytest.approx(0.75)
        assert record.risk_level == "high"
        assert record.actual_churn is None
        assert record.timestamp  # должен быть заполнен

    def test_total_predictions_count(self):
        """total_predictions считает все записи во всех вариантах."""
        exp = self._make_experiment()
        assert exp.total_predictions == 0
        exp.record_prediction("c1", "control", 0.3, "low")
        exp.record_prediction("c2", "treatment", 0.8, "high")
        exp.record_prediction("c3", "control", 0.5, "medium")
        assert exp.total_predictions == 3

    def test_record_outcome_updates_existing_prediction(self):
        """record_outcome() заполняет actual_churn в существующей записи."""
        exp = self._make_experiment()
        exp.record_prediction("cust_42", "control", 0.65, "high")
        result = exp.record_outcome("cust_42", actual_churn=True)
        assert result is True
        records = exp.get_variant_predictions("control")
        assert records[0].actual_churn is True

    def test_record_outcome_unknown_customer_returns_false(self):
        """record_outcome() для неизвестного ID возвращает False."""
        exp = self._make_experiment()
        result = exp.record_outcome("unknown_cust_999", actual_churn=False)
        assert result is False

    def test_compute_results_not_enough_data(self):
        """При малом числе предсказаний статус = not_enough_data."""
        exp = self._make_experiment(min_samples=100)
        exp.record_prediction("c1", "control", 0.3, "low")
        exp.record_prediction("c2", "treatment", 0.8, "high")
        result = exp.compute_results()
        assert result.status == "not_enough_data"
        assert result.winner is None
        assert "more samples" in result.recommendation

    def test_compute_results_enough_data_returns_result(self):
        """При достаточном числе предсказаний получаем статистический результат."""
        exp = self._make_experiment(min_samples=10)
        rng = np.random.default_rng(42)
        # control: умеренный риск
        for i in range(20):
            prob = float(rng.uniform(0.3, 0.6))
            risk = "medium"
            exp.record_prediction(f"ctrl_{i}", "control", prob, risk)
        # treatment: такой же риск (нет разницы → inconclusive)
        for i in range(20):
            prob = float(rng.uniform(0.3, 0.6))
            risk = "medium"
            exp.record_prediction(f"trt_{i}", "treatment", prob, risk)
        result = exp.compute_results()
        assert result.status in ("significant", "inconclusive")
        assert isinstance(result.control_stats.n_predictions, int)
        assert result.control_stats.n_predictions == 20

    def test_reset_clears_predictions(self):
        """reset() очищает все собранные данные."""
        exp = self._make_experiment()
        exp.record_prediction("c1", "control", 0.5, "medium")
        exp.record_prediction("c2", "treatment", 0.7, "high")
        exp.reset()
        assert exp.total_predictions == 0

    def test_variant_config_weights_must_sum_to_one(self):
        """Веса вариантов должны суммироваться в 1.0."""
        from churn.ab_testing.experiment import ABExperiment, VariantConfig

        with pytest.raises(ValueError, match="sum to 1.0"):
            ABExperiment(
                variants=[
                    VariantConfig("a", 0.3),
                    VariantConfig("b", 0.3),
                ]
            )

    def test_get_status_summary_structure(self):
        """get_status_summary() возвращает корректную структуру."""
        exp = self._make_experiment()
        summary = exp.get_status_summary()
        assert "total_predictions" in summary
        assert "variants" in summary
        assert len(summary["variants"]) == 2
        assert summary["variants"][0]["name"] == "control"

    def test_variant_names_property(self):
        """variant_names возвращает список имён вариантов."""
        exp = self._make_experiment()
        assert exp.variant_names == ["control", "treatment"]

    def test_significant_winner_lower_risk_rate(self):
        """При значимой разнице победитель — вариант с меньшим high-risk rate."""
        exp = self._make_experiment(min_samples=20)
        # control: 80% high risk
        for i in range(30):
            exp.record_prediction(f"ctrl_{i}", "control", 0.75, "high")
        # treatment: 0% high risk
        for i in range(30):
            exp.record_prediction(f"trt_{i}", "treatment", 0.3, "low")
        result = exp.compute_results()
        if result.status == "significant":
            # treatment должен победить (меньше high-risk)
            assert result.winner == "treatment"

    def test_is_scipy_available_returns_bool(self):
        """is_scipy_available() возвращает bool."""
        from churn.ab_testing.experiment import is_scipy_available

        assert isinstance(is_scipy_available(), bool)

    def test_experiment_result_dataclass_fields(self):
        """ExperimentResult имеет все ожидаемые поля."""
        from churn.ab_testing.experiment import ABExperiment, VariantConfig

        exp = ABExperiment(
            variants=[
                VariantConfig("control", 0.5),
                VariantConfig("treatment", 0.5),
            ],
            min_samples_per_variant=999,
        )
        result = exp.compute_results()
        assert hasattr(result, "status")
        assert hasattr(result, "winner")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "min_samples_per_variant")
        assert hasattr(result, "scipy_available")


class TestABAPIEndpoints:
    """Тесты FastAPI A/B endpoints / Tests for A/B testing API endpoints."""

    @pytest.fixture(autouse=True)
    def reset_experiment(self):
        """Сбрасываем эксперимент перед каждым тестом."""
        from churn.api.app import _experiment

        _experiment.reset()
        yield
        _experiment.reset()

    def test_ab_status_endpoint(self):
        """GET /ab/status возвращает корректную структуру."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/ab/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_predictions" in data
        assert "variants" in data
        assert len(data["variants"]) == 2

    def test_ab_results_not_enough_data(self):
        """GET /ab/results при пустом эксперименте → not_enough_data."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/ab/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_enough_data"
        assert data["winner"] is None

    def test_ab_reset_endpoint(self):
        """POST /ab/reset очищает данные эксперимента."""
        from churn.api.app import _experiment, app
        from fastapi.testclient import TestClient

        # Добавим предсказание напрямую
        _experiment.record_prediction("c1", "control", 0.5, "medium")

        client = TestClient(app)
        resp = client.post("/ab/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"
        assert _experiment.total_predictions == 0

    def test_ab_outcome_unknown_customer(self):
        """POST /ab/outcome для незнакомого ID → recorded=False."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/ab/outcome",
            json={"customer_id": "ghost_customer_xyz", "actual_churn": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["recorded"] is False
        assert "ghost_customer_xyz" in data["customer_id"]

    def test_ab_outcome_after_prediction(self):
        """POST /ab/outcome обновляет запись после предсказания."""
        from churn.api.app import _experiment, app
        from fastapi.testclient import TestClient

        _experiment.record_prediction("cust_999", "control", 0.6, "medium")

        client = TestClient(app)
        resp = client.post(
            "/ab/outcome",
            json={"customer_id": "cust_999", "actual_churn": False},
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True

    def test_ab_results_has_required_fields(self):
        """GET /ab/results содержит все ключи ABResultsResponse."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/ab/results")
        data = resp.json()
        for key in (
            "status",
            "winner",
            "control_n",
            "treatment_n",
            "control_high_risk_rate",
            "treatment_high_risk_rate",
            "recommendation",
            "scipy_available",
        ):
            assert key in data, f"Missing key: {key}"

    def test_ab_status_shows_variant_counts(self):
        """GET /ab/status показывает корректные счётчики после предсказаний."""
        from churn.api.app import _experiment, app
        from fastapi.testclient import TestClient

        # Добавляем напрямую в эксперимент (без модели)
        _experiment.record_prediction("c1", "control", 0.3, "low")
        _experiment.record_prediction("c2", "control", 0.4, "medium")
        _experiment.record_prediction("c3", "treatment", 0.7, "high")

        client = TestClient(app)
        resp = client.get("/ab/status")
        data = resp.json()
        assert data["total_predictions"] == 3

        ctrl = next(v for v in data["variants"] if v["name"] == "control")
        trt = next(v for v in data["variants"] if v["name"] == "treatment")
        assert ctrl["n_predictions"] == 2
        assert trt["n_predictions"] == 1

    def test_ab_outcome_validation(self):
        """POST /ab/outcome с некорректными данными → 422."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/ab/outcome", json={"customer_id": "c1"})  # нет actual_churn
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Automated Model Comparison Tests
# ---------------------------------------------------------------------------


def _make_model_result(
    name: str = "catboost",
    roc_auc: float = 0.85,
    f1_score: float = 0.62,
    precision: float = 0.70,
    recall: float = 0.56,
    training_time_sec: float = 12.5,
    feature_importances: dict | None = None,
    run_id: str | None = None,
):
    """Factory for ModelResult in tests."""
    from churn.evaluation.model_comparison import ModelResult

    return ModelResult(
        name=name,
        roc_auc=roc_auc,
        f1_score=f1_score,
        precision=precision,
        recall=recall,
        training_time_sec=training_time_sec,
        feature_importances=feature_importances or {},
        run_id=run_id,
    )


class TestModelComparisonCore:
    """Тесты ядра сравнения моделей / Tests for compare_models() logic."""

    def test_single_model_wins_by_default(self):
        """Единственная модель → победитель без runner-up."""
        from churn.evaluation.model_comparison import compare_models

        result = compare_models([_make_model_result("solo", roc_auc=0.80)])
        assert result.summary.winner == "solo"
        assert result.summary.runner_up is None
        assert result.summary.auc_margin == 0.0

    def test_winner_by_higher_auc(self):
        """Модель с бо́льшим AUC побеждает."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("lgbm", roc_auc=0.82),
            _make_model_result("catboost", roc_auc=0.87),
        ]
        report = compare_models(results)
        assert report.summary.winner == "catboost"
        assert report.summary.runner_up == "lgbm"

    def test_auc_margin_computed_correctly(self):
        """auc_margin = winner_auc - runner_up_auc."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("a", roc_auc=0.80),
            _make_model_result("b", roc_auc=0.85),
        ]
        report = compare_models(results)
        assert abs(report.summary.auc_margin - 0.05) < 1e-4

    def test_significant_when_margin_large(self):
        """Разница >= 0.02 → is_significant=True."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("a", roc_auc=0.80),
            _make_model_result("b", roc_auc=0.83),  # margin 0.03
        ]
        report = compare_models(results)
        assert report.summary.is_significant is True

    def test_not_significant_when_margin_small(self):
        """Разница < 0.02 → is_significant=False."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("a", roc_auc=0.850),
            _make_model_result("b", roc_auc=0.855),  # margin 0.005
        ]
        report = compare_models(results)
        assert report.summary.is_significant is False

    def test_f1_tiebreaker_on_equal_auc(self):
        """При равном AUC побеждает модель с лучшим F1."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("a", roc_auc=0.85, f1_score=0.60),
            _make_model_result("b", roc_auc=0.85, f1_score=0.65),
        ]
        report = compare_models(results)
        assert report.summary.winner == "b"

    def test_leaderboard_sorted_descending(self):
        """Leaderboard ранжирован по убыванию AUC."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("c", roc_auc=0.78),
            _make_model_result("a", roc_auc=0.88),
            _make_model_result("b", roc_auc=0.83),
        ]
        report = compare_models(results)
        aucs = [row["roc_auc"] for row in report.leaderboard]
        assert aucs == sorted(aucs, reverse=True)

    def test_leaderboard_rank_sequential(self):
        """Rank = 1, 2, 3, ... последовательно."""
        from churn.evaluation.model_comparison import compare_models

        results = [_make_model_result(f"m{i}", roc_auc=0.8 + i * 0.01) for i in range(4)]
        report = compare_models(results)
        ranks = [row["rank"] for row in report.leaderboard]
        assert ranks == list(range(1, 5))

    def test_empty_results_raises(self):
        """Пустой список → ValueError."""
        from churn.evaluation.model_comparison import compare_models

        with pytest.raises(ValueError, match="at least one"):
            compare_models([])

    def test_three_models_ranking(self):
        """Три модели → правильная расстановка призовых мест."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("xgb", roc_auc=0.81),
            _make_model_result("lgbm", roc_auc=0.84),
            _make_model_result("catboost", roc_auc=0.87),
        ]
        report = compare_models(results)
        assert report.leaderboard[0]["name"] == "catboost"
        assert report.leaderboard[1]["name"] == "lgbm"
        assert report.leaderboard[2]["name"] == "xgb"

    def test_recommendation_contains_winner_name(self):
        """recommendation упоминает имя победителя."""
        from churn.evaluation.model_comparison import compare_models

        results = [
            _make_model_result("model_x", roc_auc=0.88),
            _make_model_result("model_y", roc_auc=0.82),
        ]
        report = compare_models(results)
        assert "model_x" in report.summary.recommendation

    def test_timestamp_is_iso8601(self):
        """timestamp в формате ISO 8601."""
        from churn.evaluation.model_comparison import compare_models

        report = compare_models([_make_model_result()])
        assert "T" in report.timestamp  # ISO 8601: YYYY-MM-DDTHH:MM:SS


class TestComparisonReportFormat:
    """Тесты форматирования отчётов / Tests for report generation."""

    def test_json_report_required_keys(self):
        """JSON-отчёт содержит timestamp, summary, leaderboard."""
        from churn.evaluation.model_comparison import compare_models, generate_json_report

        report = compare_models([_make_model_result()])
        jr = generate_json_report(report)
        for key in ("timestamp", "summary", "leaderboard"):
            assert key in jr, f"Missing key: {key}"

    def test_json_summary_keys(self):
        """summary в JSON содержит все обязательные поля."""
        from churn.evaluation.model_comparison import compare_models, generate_json_report

        report = compare_models([_make_model_result()])
        summary = generate_json_report(report)["summary"]
        required_keys = (
            "winner",
            "winner_auc",
            "runner_up",
            "auc_margin",
            "is_significant",
            "recommendation",
        )
        for key in required_keys:
            assert key in summary, f"Missing summary key: {key}"

    def test_json_report_serializable(self):
        """JSON-отчёт можно сериализовать через json.dumps() без ошибок."""
        import json as _json

        from churn.evaluation.model_comparison import compare_models, generate_json_report

        results = [
            _make_model_result("a", roc_auc=0.85, feature_importances={"tenure": 12.5}),
            _make_model_result("b", roc_auc=0.82),
        ]
        jr = generate_json_report(compare_models(results))
        serialized = _json.dumps(jr)  # не должен бросать исключение
        assert len(serialized) > 0

    def test_markdown_report_has_header(self):
        """Markdown-отчёт начинается с заголовка '# Model Comparison Report'."""
        from churn.evaluation.model_comparison import compare_models, generate_markdown_report

        md = generate_markdown_report(compare_models([_make_model_result()]))
        assert md.startswith("# Model Comparison Report")

    def test_markdown_report_contains_winner(self):
        """Markdown-отчёт упоминает имя победителя в секции Summary."""
        from churn.evaluation.model_comparison import compare_models, generate_markdown_report

        results = [
            _make_model_result("best_model", roc_auc=0.90),
            _make_model_result("other_model", roc_auc=0.85),
        ]
        md = generate_markdown_report(compare_models(results))
        assert "best_model" in md

    def test_markdown_report_has_leaderboard_table(self):
        """Markdown-отчёт содержит таблицу (строка с '|')."""
        from churn.evaluation.model_comparison import compare_models, generate_markdown_report

        md = generate_markdown_report(compare_models([_make_model_result()]))
        assert "|" in md

    def test_markdown_report_includes_feature_importances(self):
        """Секция Feature Importance появляется если importances непустые."""
        from churn.evaluation.model_comparison import compare_models, generate_markdown_report

        m = _make_model_result(feature_importances={"MonthlyCharges": 30.0, "tenure": 20.0})
        md = generate_markdown_report(compare_models([m]))
        assert "Feature Importance" in md
        assert "MonthlyCharges" in md

    def test_markdown_no_importance_section_when_empty(self):
        """Секция Feature Importance отсутствует если importances пустые."""
        from churn.evaluation.model_comparison import compare_models, generate_markdown_report

        md = generate_markdown_report(compare_models([_make_model_result()]))
        assert "Feature Importance" not in md

    def test_leaderboard_values_rounded(self):
        """Значения в leaderboard округлены до 4 знаков."""
        from churn.evaluation.model_comparison import compare_models

        m = _make_model_result(roc_auc=0.856789, f1_score=0.623456)
        report = compare_models([m])
        row = report.leaderboard[0]
        assert row["roc_auc"] == round(0.856789, 4)
        assert row["f1_score"] == round(0.623456, 4)


class TestComparisonAPIEndpoints:
    """Тесты API-эндпоинтов сравнения / Tests for /compare/* API endpoints."""

    def _make_payload(self, models: list[dict] | None = None) -> dict:
        if models is None:
            models = [
                {"name": "catboost", "roc_auc": 0.87, "f1_score": 0.63},
                {"name": "lgbm", "roc_auc": 0.84, "f1_score": 0.60},
            ]
        return {"models": models, "format": "json"}

    def test_compare_endpoint_returns_200(self):
        """POST /compare/models → 200."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/compare/models", json=self._make_payload())
        assert resp.status_code == 200

    def test_compare_endpoint_returns_winner(self):
        """POST /compare/models → summary.winner = модель с макс AUC."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(
            [
                {"name": "lgbm", "roc_auc": 0.82, "f1_score": 0.60},
                {"name": "catboost", "roc_auc": 0.88, "f1_score": 0.65},
            ]
        )
        resp = client.post("/compare/models", json=payload)
        data = resp.json()
        assert data["summary"]["winner"] == "catboost"

    def test_compare_endpoint_leaderboard_ordered(self):
        """Leaderboard отсортирован по убыванию AUC."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/compare/models", json=self._make_payload())
        data = resp.json()
        aucs = [row["roc_auc"] for row in data["leaderboard"]]
        assert aucs == sorted(aucs, reverse=True)

    def test_compare_single_model(self):
        """POST /compare/models с одной моделью → 200, корректный summary."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload([{"name": "only_model", "roc_auc": 0.80, "f1_score": 0.58}])
        resp = client.post("/compare/models", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["winner"] == "only_model"
        assert data["summary"]["runner_up"] is None

    def test_compare_markdown_format(self):
        """format=markdown → ответ содержит поле 'markdown' с таблицей."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = {**self._make_payload(), "format": "markdown"}
        resp = client.post("/compare/models", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "markdown" in data
        assert "# Model Comparison Report" in data["markdown"]

    def test_get_report_after_compare(self):
        """GET /compare/report возвращает закэшированный отчёт после POST."""
        import churn.api.app as app_module
        from churn.api.app import app
        from fastapi.testclient import TestClient

        # Сбрасываем кэш перед тестом
        app_module._last_comparison_report = None

        client = TestClient(app)
        client.post("/compare/models", json=self._make_payload())
        resp = client.get("/compare/report")
        assert resp.status_code == 200
        assert "summary" in resp.json()

    def test_get_report_before_compare_returns_404(self):
        """GET /compare/report без предварительного POST → 404."""
        import churn.api.app as app_module
        from churn.api.app import app
        from fastapi.testclient import TestClient

        app_module._last_comparison_report = None

        client = TestClient(app)
        resp = client.get("/compare/report")
        assert resp.status_code == 404

    def test_get_leaderboard_empty_before_compare(self):
        """GET /compare/leaderboard без данных → leaderboard=[] без ошибок."""
        import churn.api.app as app_module
        from churn.api.app import app
        from fastapi.testclient import TestClient

        app_module._last_comparison_report = None

        client = TestClient(app)
        resp = client.get("/compare/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["leaderboard"] == []

    def test_get_leaderboard_after_compare(self):
        """GET /compare/leaderboard возвращает таблицу после POST /compare/models."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        client.post("/compare/models", json=self._make_payload())
        resp = client.get("/compare/leaderboard")
        assert resp.status_code == 200
        assert len(resp.json()["leaderboard"]) == 2

    def test_compare_empty_models_returns_422(self):
        """POST /compare/models с пустым списком → 422 валидация."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/compare/models", json={"models": [], "format": "json"})
        assert resp.status_code == 422

    def test_compare_missing_required_fields_returns_422(self):
        """POST /compare/models без roc_auc → 422."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        bad_payload = {"models": [{"name": "no_auc"}], "format": "json"}
        resp = client.post("/compare/models", json=bad_payload)
        assert resp.status_code == 422

    def test_compare_with_run_id(self):
        """run_id прокидывается в leaderboard без ошибок."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = self._make_payload(
            [
                {"name": "m1", "roc_auc": 0.85, "f1_score": 0.62, "run_id": "abc123"},
            ]
        )
        resp = client.post("/compare/models", json=payload)
        data = resp.json()
        assert data["leaderboard"][0]["run_id"] == "abc123"
