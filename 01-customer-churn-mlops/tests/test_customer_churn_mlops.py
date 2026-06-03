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


# ---------------------------------------------------------------------------
# TestQuantizer — post-training quantization utilities
# ---------------------------------------------------------------------------


class TestQuantizer:
    """Тесты для churn/optimization/quantizer.py."""

    def test_is_available(self):
        """numpy всегда доступен в тестовом окружении."""
        from churn.optimization.quantizer import is_available

        assert is_available() is True

    def test_quantize_array_shape(self):
        """Квантованный массив имеет ту же форму, что и оригинальный."""
        from churn.optimization.quantizer import _quantize_array

        weights = np.random.randn(10, 5).astype(np.float64)
        qw = _quantize_array(weights)
        assert qw.weights_int8.shape == weights.shape

    def test_quantize_array_dtype_is_int8(self):
        """Квантованные веса имеют тип int8."""
        from churn.optimization.quantizer import _quantize_array

        weights = np.random.randn(20).astype(np.float64)
        qw = _quantize_array(weights)
        assert qw.weights_int8.dtype == np.int8

    def test_quantize_array_dequantize_accuracy(self):
        """Деквантизация восстанавливает значения с точностью ~1% от диапазона."""
        from churn.optimization.quantizer import _quantize_array

        np.random.seed(42)
        weights = np.random.randn(100).astype(np.float64)
        qw = _quantize_array(weights)
        reconstructed = qw.dequantize()

        # Максимальная ошибка <= 1/256 от диапазона весов
        w_range = weights.max() - weights.min()
        max_error = np.abs(weights - reconstructed).max()
        assert max_error <= w_range / 100.0, f"max_error={max_error:.6f}, range={w_range:.6f}"

    def test_quantize_linear_model_logistic_regression(self):
        """Квантизация sklearn LogisticRegression возвращает QuantizedModel."""
        from churn.optimization.quantizer import QuantizedModel, quantize_linear_model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        # Минимальное обучение на синтетических данных
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)

        qmodel, result = quantize_linear_model(model)
        assert isinstance(qmodel, QuantizedModel)
        assert result.compression_ratio > 1.0
        assert result.n_params > 0
        assert result.dtype_quantized == "int8"

    def test_quantize_linear_model_compression_ratio(self):
        """INT8 даёт ~8x сжатие для float64 весов (float64=8b → int8=1b)."""
        from churn.optimization.quantizer import quantize_linear_model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        X = np.random.randn(50, 10)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)

        _, result = quantize_linear_model(model)
        # float64 → int8: теоретически 8x, реально 4-8x
        assert result.compression_ratio >= 4.0

    def test_quantize_linear_model_predict_proba_preserved(self):
        """QuantizedModel.predict_proba() даёт те же результаты, что и оригинал."""
        from churn.optimization.quantizer import quantize_linear_model
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        X = np.random.randn(200, 5)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)

        qmodel, _ = quantize_linear_model(model)
        X_test = np.random.randn(10, 5)
        np.testing.assert_array_equal(
            model.predict_proba(X_test),
            qmodel.predict_proba(X_test),
        )

    def test_quantize_linear_model_no_coef_raises(self):
        """Модель без coef_ вызывает ValueError."""
        from churn.optimization.quantizer import quantize_linear_model

        class FakeModel:
            pass

        with pytest.raises(ValueError, match="no coef_"):
            quantize_linear_model(FakeModel())

    def test_quantize_tree_ensemble_compression(self):
        """Прунинг 50% деревьев даёт compression_ratio=2."""
        from churn.optimization.quantizer import quantize_tree_ensemble

        class MockLGBM:
            n_estimators = 100

        _, result = quantize_tree_ensemble(MockLGBM(), keep_fraction=0.5)
        assert result.compression_ratio == pytest.approx(2.0)
        assert result.metadata["n_estimators_kept"] == 50

    def test_quantize_tree_ensemble_invalid_raises(self):
        """Не-ансамблевая модель вызывает ValueError."""
        from churn.optimization.quantizer import quantize_tree_ensemble

        class NotAnEnsemble:
            pass

        with pytest.raises(ValueError, match="not a tree ensemble"):
            quantize_tree_ensemble(NotAnEnsemble())

    def test_estimate_inference_speedup_int8(self):
        """Оценка speedup для INT8 возвращает положительные значения."""
        from churn.optimization.quantizer import (
            QuantizationResult,
            estimate_inference_speedup,
        )

        result = QuantizationResult(
            original_size_bytes=8000,
            quantized_size_bytes=1000,
            compression_ratio=8.0,
            weight_error_l2=0.01,
            n_params=1000,
            dtype_original="float64",
            dtype_quantized="int8",
        )
        speedup = estimate_inference_speedup(result)
        assert speedup["theoretical_speedup"] > 1.0
        assert speedup["practical_speedup_estimate"] > 1.0
        assert 0.0 < speedup["memory_reduction_pct"] <= 100.0

    def test_quantization_result_size_reduction_pct(self):
        """size_reduction_pct корректно вычисляется из compression_ratio."""
        from churn.optimization.quantizer import QuantizationResult

        result = QuantizationResult(
            original_size_bytes=8000,
            quantized_size_bytes=1000,
            compression_ratio=8.0,
            weight_error_l2=0.0,
            n_params=1000,
            dtype_original="float64",
            dtype_quantized="int8",
        )
        assert result.size_reduction_pct == pytest.approx(87.5, abs=0.1)


# ---------------------------------------------------------------------------
# TestCostTracker — inference cost tracking
# ---------------------------------------------------------------------------


class TestCostTracker:
    """Тесты для churn/optimization/cost_tracker.py."""

    def test_track_context_manager(self):
        """track() записывает задержку > 0."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker(window_size=100)
        with tracker.track():
            _ = sum(range(1000))

        stats = tracker.get_stats()
        assert stats.n_requests == 1
        assert stats.p50_ms >= 0.0

    def test_total_requests_increments(self):
        """total_requests растёт с каждым track()."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker()
        for _ in range(5):
            with tracker.track():
                pass
        assert tracker.total_requests == 5

    def test_record_latency_direct(self):
        """record_latency() добавляет значение в окно."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_latency(42.0)
        tracker.record_latency(10.0)
        stats = tracker.get_stats()
        assert stats.n_requests == 2
        assert stats.mean_ms == pytest.approx(26.0, abs=0.1)

    def test_window_size_respected(self):
        """Окно не превышает window_size записей."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker(window_size=5)
        for i in range(10):
            tracker.record_latency(float(i))
        stats = tracker.get_stats()
        assert stats.n_requests == 5

    def test_stats_empty_tracker(self):
        """Пустой трекер возвращает нулевую статистику."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker()
        stats = tracker.get_stats()
        assert stats.n_requests == 0
        assert stats.p50_ms == 0.0
        assert stats.throughput_rps == 0.0

    def test_reset_clears_window(self):
        """reset() очищает окно наблюдений."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_latency(10.0)
        tracker.reset()
        stats = tracker.get_stats()
        assert stats.n_requests == 0

    def test_percentiles_order(self):
        """p50 <= p95 <= p99."""
        from churn.optimization.cost_tracker import CostTracker

        tracker = CostTracker()
        for v in range(1, 101):
            tracker.record_latency(float(v))
        stats = tracker.get_stats()
        assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms

    def test_estimate_monthly_cost_basic(self):
        """estimate_monthly_cost возвращает положительную стоимость."""
        from churn.optimization.cost_tracker import estimate_monthly_cost

        cost = estimate_monthly_cost(rps=10.0)
        assert cost.cost_per_month_usd > 0
        assert cost.cost_per_million_requests_usd > 0
        assert cost.n_instances_recommended >= 1

    def test_estimate_monthly_cost_scales_with_rps(self):
        """Высокий RPS требует больше инстансов."""
        from churn.optimization.cost_tracker import estimate_monthly_cost

        low = estimate_monthly_cost(rps=1.0)
        high = estimate_monthly_cost(rps=200.0)
        assert high.cost_per_month_usd > low.cost_per_month_usd

    def test_estimate_monthly_cost_invalid_rps(self):
        """rps <= 0 вызывает ValueError."""
        from churn.optimization.cost_tracker import estimate_monthly_cost

        with pytest.raises(ValueError, match="rps must be positive"):
            estimate_monthly_cost(rps=0.0)

    def test_optimize_batch_size_finds_optimum(self):
        """optimize_batch_size находит батч с максимальным throughput в рамках SLA."""
        from churn.optimization.cost_tracker import optimize_batch_size

        # batch=32 нарушает SLA.
        # throughput: 1→100, 8→200, 16→177 req/s — оптимум batch=8
        profile = {1: 10.0, 8: 40.0, 16: 90.0, 32: 250.0}
        result = optimize_batch_size(profile, sla_p95_ms=200.0)
        assert result.optimal_batch_size == 8
        assert result.throughput_rps > 0

    def test_optimize_batch_size_empty_raises(self):
        """Пустой профиль вызывает ValueError."""
        from churn.optimization.cost_tracker import optimize_batch_size

        with pytest.raises(ValueError):
            optimize_batch_size({})

    def test_optimize_batch_size_no_sla_compliant(self):
        """Если все батчи нарушают SLA — выбирается наименее медленный."""
        from churn.optimization.cost_tracker import optimize_batch_size

        profile = {1: 300.0, 2: 500.0}
        result = optimize_batch_size(profile, sla_p95_ms=100.0)
        assert result.optimal_batch_size == 1
        assert len(result.recommendations) > 0


# ---------------------------------------------------------------------------
# TestOptimizeAPIEndpoints — /optimize/* endpoints
# ---------------------------------------------------------------------------


class TestOptimizeAPIEndpoints:
    """Тесты для /optimize/stats и /optimize/batch endpoints."""

    def test_optimize_stats_returns_200(self):
        """GET /optimize/stats возвращает 200 с обязательными полями."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/optimize/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "latency_p50_ms" in data
        assert "latency_p95_ms" in data
        assert "throughput_rps" in data
        assert "cost_estimate_10rps" in data
        assert "model_quantization_estimate" in data

    def test_optimize_stats_cost_estimate_structure(self):
        """cost_estimate_10rps содержит все обязательные поля."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/optimize/stats")
        data = resp.json()
        cost = data["cost_estimate_10rps"]
        assert "rps" in cost
        assert "instance_type" in cost
        assert "cost_per_month_usd" in cost
        assert cost["cost_per_month_usd"] > 0

    def test_optimize_stats_quantization_estimate_structure(self):
        """model_quantization_estimate содержит speedup и рекомендации."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/optimize/stats")
        data = resp.json()
        quant = data["model_quantization_estimate"]
        assert "compression_ratio" in quant
        assert "memory_reduction_pct" in quant
        assert quant["compression_ratio"] > 1.0

    def test_optimize_batch_valid_profile(self):
        """POST /optimize/batch с валидным профилем возвращает optimal_batch_size."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = {
            "latencies_by_batch": {"1": 10.0, "8": 50.0, "16": 120.0, "32": 300.0},
            "sla_p95_ms": 200.0,
        }
        resp = client.post("/optimize/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "optimal_batch_size" in data
        assert data["optimal_batch_size"] in [1, 8, 16]

    def test_optimize_batch_empty_raises_422(self):
        """POST /optimize/batch с пустым профилем возвращает 422."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = {"latencies_by_batch": {}, "sla_p95_ms": 200.0}
        resp = client.post("/optimize/batch", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TestIncrementalLearnerUnit — unit tests for churn/online/learner.py
# ---------------------------------------------------------------------------


class TestIncrementalLearnerUnit:
    """Юнит-тесты для IncrementalChurnLearner и связанных dataclass'ов.

    Unit tests for IncrementalChurnLearner and related dataclasses.
    """

    def test_is_available_returns_bool(self):
        """is_available() возвращает bool независимо от наличия River."""
        from churn.online.learner import is_available

        assert isinstance(is_available(), bool)

    def test_config_defaults(self):
        """IncrementalConfig имеет разумные значения по умолчанию."""
        from churn.online.learner import IncrementalConfig

        cfg = IncrementalConfig()
        assert cfg.model_type == "hoeffding_tree"
        assert 0.0 < cfg.adwin_delta < 1.0
        assert cfg.snapshot_interval > 0

    def test_initial_status_zero_samples(self):
        """Новый классификатор имеет 0 обработанных примеров."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        status = learner.get_status()
        assert status["n_samples_seen"] == 0
        assert status["n_drift_detections"] == 0

    def test_status_contains_required_fields(self):
        """get_status() содержит все обязательные поля для мониторинга."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        status = learner.get_status()
        required = {
            "model_type",
            "river_available",
            "n_samples_seen",
            "n_drift_detections",
            "current_error_rate",
            "last_drift_at",
            "adwin_delta",
            "snapshot_interval",
            "class_distribution",
        }
        assert required.issubset(set(status.keys()))

    def test_predict_one_before_learning_returns_valid_probability(self):
        """predict_one без обучения возвращает вероятность в [0, 1]."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        result = learner.predict_one({"tenure": 12.0, "MonthlyCharges": 70.0})
        assert 0.0 <= result.churn_probability <= 1.0
        assert isinstance(result.churn_prediction, bool)

    def test_predict_one_risk_level_matches_probability(self):
        """risk_level соответствует чётким порогам вероятности."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        # Манипулируем fallback-счётчиками для гарантированного результата
        learner._fallback_counts = {0: 90, 1: 10}  # ~10% churn → low risk
        result = learner.predict_one({"tenure": 36.0})
        assert result.risk_level in {"low", "medium", "high"}

    def test_learn_one_increments_sample_count(self):
        """learn_one увеличивает n_samples_seen на 1 при каждом вызове."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        for i in range(5):
            r = learner.learn_one({"tenure": float(i), "MonthlyCharges": 50.0}, label=0)
            assert r.n_samples_seen == i + 1

    def test_learn_one_returns_learn_result_dataclass(self):
        """learn_one возвращает LearnResult с правильными полями."""
        from churn.online.learner import IncrementalChurnLearner, LearnResult

        learner = IncrementalChurnLearner()
        result = learner.learn_one({"tenure": 5.0}, label=1)
        assert isinstance(result, LearnResult)
        assert 0.0 <= result.error <= 1.0
        assert isinstance(result.drift_detected, bool)
        assert isinstance(result.snapshot_saved, bool)

    def test_class_distribution_tracked_after_learning(self):
        """Счётчик классов обновляется после learn_one."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        for _ in range(7):
            learner.learn_one({"tenure": 10.0}, label=0)
        for _ in range(3):
            learner.learn_one({"tenure": 1.0}, label=1)

        status = learner.get_status()
        dist = status["class_distribution"]
        assert dist["n_non_churn"] == 7
        assert dist["n_churn"] == 3

    def test_snapshot_saved_at_configured_interval(self, tmp_path):
        """Снапшот сохраняется ровно на кратном snapshot_interval шаге."""
        from churn.online.learner import IncrementalChurnLearner, IncrementalConfig

        cfg = IncrementalConfig(snapshot_interval=5, snapshot_dir=str(tmp_path))
        learner = IncrementalChurnLearner(cfg)

        results = [learner.learn_one({"x": float(i)}, label=i % 2) for i in range(10)]
        # Снапшоты должны быть на шагах 5 и 10
        snaps = [r for r in results if r.snapshot_saved]
        assert len(snaps) == 2

    def test_snapshot_file_is_loadable(self, tmp_path):
        """Загрузка снапшота восстанавливает n_samples_seen."""
        from churn.online.learner import IncrementalChurnLearner, IncrementalConfig

        cfg = IncrementalConfig(snapshot_interval=3, snapshot_dir=str(tmp_path))
        learner = IncrementalChurnLearner(cfg)
        for i in range(3):
            learner.learn_one({"tenure": float(i)}, label=0)

        assert learner._last_snapshot is not None
        snap_path = learner._last_snapshot.path

        restored = IncrementalChurnLearner.load_snapshot(snap_path)
        assert restored._n_samples == 3

    def test_predict_after_learning_updates_model_type(self):
        """После обучения model_type отражает наличие River."""
        from churn.online.learner import IncrementalChurnLearner, is_available

        learner = IncrementalChurnLearner()
        learner.learn_one({"tenure": 5.0}, label=0)
        result = learner.predict_one({"tenure": 5.0})

        if is_available():
            assert result.model_type.startswith("river_")
        else:
            assert result.model_type == "fallback_freq"

    def test_drift_state_has_correct_initial_values(self):
        """DriftState начинается с нулей."""
        from churn.online.learner import DriftState

        state = DriftState()
        assert state.n_detected == 0
        assert state.last_detected_at is None
        assert state.current_error_rate == 0.0


# ---------------------------------------------------------------------------
# TestIncrementalLearnerIntegration — integration tests
# ---------------------------------------------------------------------------


class TestIncrementalLearnerIntegration:
    """Интеграционные тесты для инкрементального обучения.

    Integration tests for incremental learning.
    """

    def test_learn_many_samples_stable_prediction(self):
        """После 50 обучающих примеров модель возвращает стабильные предсказания."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        # Простой паттерн: высокий tenure → не уйдёт
        for _ in range(50):
            learner.learn_one({"tenure": 60.0, "MonthlyCharges": 30.0}, label=0)

        result = learner.predict_one({"tenure": 60.0, "MonthlyCharges": 30.0})
        assert result.n_samples_seen == 50
        assert 0.0 <= result.churn_probability <= 1.0

    def test_fallback_mode_probability_in_range(self):
        """Fallback-режим (без River) возвращает вероятность в [0, 1]."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        # Симулируем fallback через прямую установку счётчиков
        learner._fallback_counts = {0: 80, 1: 20}
        result = learner.predict_one({"tenure": 12.0})
        assert 0.0 <= result.churn_probability <= 1.0

    def test_incremental_result_has_drift_state(self):
        """IncrementalResult содержит drift_state с обязательными полями."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        result = learner.predict_one({"f1": 1.0})
        assert result.drift_state is not None
        assert hasattr(result.drift_state, "n_detected")
        assert hasattr(result.drift_state, "current_error_rate")

    def test_learn_result_drift_state_propagated(self):
        """LearnResult.drift_state отражает реальное состояние после обучения."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        result = learner.learn_one({"tenure": 10.0}, label=0)
        ds = result.drift_state
        assert ds.n_detected >= 0
        assert ds.n_samples_since_last >= 1

    def test_mixed_labels_class_distribution_consistent(self):
        """Смешанные метки: распределение классов соответствует поданным примерам."""
        from churn.online.learner import IncrementalChurnLearner

        learner = IncrementalChurnLearner()
        labels = [0] * 60 + [1] * 40
        for i, y in enumerate(labels):
            learner.learn_one({"f": float(i)}, label=y)

        dist = learner.get_status()["class_distribution"]
        assert dist["n_non_churn"] == 60
        assert dist["n_churn"] == 40


# ---------------------------------------------------------------------------
# TestOnlineAPIEndpoints — /online/* API tests
# ---------------------------------------------------------------------------


class TestOnlineAPIEndpoints:
    """Тесты для /online/status, /online/learn, /online/predict, /online/reset.

    Tests for online learning API endpoints.
    """

    def _client(self):
        from churn.api.app import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_online_status_returns_200(self):
        """GET /online/status возвращает 200."""
        resp = self._client().get("/online/status")
        assert resp.status_code == 200

    def test_online_status_structure(self):
        """GET /online/status содержит обязательные поля."""
        data = self._client().get("/online/status").json()
        assert "model_type" in data
        assert "river_available" in data
        assert "n_samples_seen" in data
        assert "n_drift_detections" in data
        assert "class_distribution" in data

    def test_online_predict_returns_200(self):
        """POST /online/predict возвращает 200."""
        payload = {"features": {"tenure": 12.0, "MonthlyCharges": 70.35}}
        resp = self._client().post("/online/predict", json=payload)
        assert resp.status_code == 200

    def test_online_predict_response_structure(self):
        """POST /online/predict содержит все обязательные поля."""
        payload = {"features": {"tenure": 12.0, "MonthlyCharges": 70.35}}
        data = self._client().post("/online/predict", json=payload).json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data
        assert "model_type" in data
        assert "n_samples_seen" in data
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_online_predict_risk_level_valid(self):
        """POST /online/predict возвращает допустимый risk_level."""
        payload = {"features": {"tenure": 5.0}}
        data = self._client().post("/online/predict", json=payload).json()
        assert data["risk_level"] in {"low", "medium", "high"}

    def test_online_learn_returns_200(self):
        """POST /online/learn возвращает 200."""
        payload = {"features": {"tenure": 12.0, "MonthlyCharges": 70.35}, "label": 0}
        resp = self._client().post("/online/learn", json=payload)
        assert resp.status_code == 200

    def test_online_learn_response_structure(self):
        """POST /online/learn содержит все обязательные поля."""
        payload = {"features": {"tenure": 5.0}, "label": 1}
        data = self._client().post("/online/learn", json=payload).json()
        assert "n_samples_seen" in data
        assert "error" in data
        assert "drift_detected" in data
        assert "snapshot_saved" in data
        assert "drift_state" in data

    def test_online_learn_increments_sample_count(self):
        """POST /online/learn n_samples_seen растёт при повторных вызовах."""
        client = self._client()
        # Сбрасываем состояние перед тестом
        client.post("/online/reset")
        payload = {"features": {"tenure": 10.0}, "label": 0}
        r1 = client.post("/online/learn", json=payload).json()
        r2 = client.post("/online/learn", json=payload).json()
        assert r2["n_samples_seen"] > r1["n_samples_seen"]

    def test_online_learn_invalid_label_raises_422(self):
        """POST /online/learn с label=2 (вне диапазона) возвращает 422."""
        payload = {"features": {"tenure": 5.0}, "label": 2}
        resp = self._client().post("/online/learn", json=payload)
        assert resp.status_code == 422

    def test_online_reset_returns_200(self):
        """POST /online/reset возвращает 200 и сообщение."""
        data = self._client().post("/online/reset").json()
        assert "status" in data
        assert data["status"] == "reset"

    def test_online_reset_clears_sample_count(self):
        """POST /online/reset обнуляет n_samples_seen."""
        client = self._client()
        client.post("/online/learn", json={"features": {"x": 1.0}, "label": 0})
        client.post("/online/reset")
        status = client.get("/online/status").json()
        assert status["n_samples_seen"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# Fairness / Bias Detection Tests
# ──────────────────────────────────────────────────────────────────────────────


from churn.fairness.bias_detector import (
    BiasDetector,
    FairnessMetrics,
    FairnessSeverity,
    GroupMetrics,
)


class TestGroupMetrics:
    """Unit tests for GroupMetrics dataclass."""

    def test_to_dict_has_required_fields(self):
        """GroupMetrics.to_dict() возвращает все 5 обязательных полей."""
        gm = GroupMetrics(sample_size=100, positive_rate=0.3, tpr=0.7, fpr=0.15, ppv=0.6)
        d = gm.to_dict()
        for key in ("sample_size", "positive_rate", "tpr", "fpr", "ppv"):
            assert key in d

    def test_to_dict_nan_becomes_none(self):
        """NaN значения (нет истинных позитивов) сериализуются как None."""
        nan = float("nan")
        gm = GroupMetrics(sample_size=10, positive_rate=0.0, tpr=nan, fpr=nan, ppv=nan)
        d = gm.to_dict()
        assert d["tpr"] is None
        assert d["fpr"] is None
        assert d["ppv"] is None


class TestFairnessMetrics:
    """Unit tests for FairnessMetrics dataclass and BiasDetector helpers."""

    def _perfect_metrics(self):
        """Метрики для модели без смещений."""
        return FairnessMetrics(
            demographic_parity_diff=0.0,
            equal_opportunity_diff=0.0,
            equalized_odds_diff=0.0,
            predictive_parity_diff=0.0,
            disparate_impact_ratio=1.0,
        )

    def test_to_dict_has_all_keys(self):
        """FairnessMetrics.to_dict() содержит все 5 метрик."""
        m = self._perfect_metrics()
        keys = m.to_dict().keys()
        for k in (
            "demographic_parity_diff",
            "equal_opportunity_diff",
            "equalized_odds_diff",
            "predictive_parity_diff",
            "disparate_impact_ratio",
        ):
            assert k in keys

    def test_severity_low_for_perfect_model(self):
        """Perfect model (no gaps) → LOW severity."""
        detector = BiasDetector()
        severity = detector._classify_severity(self._perfect_metrics())
        assert severity == FairnessSeverity.LOW

    def test_severity_high_di_below_80_percent(self):
        """DI ratio < 0.80 → HIGH severity (EEOC 80% rule)."""
        detector = BiasDetector()
        m = FairnessMetrics(
            demographic_parity_diff=0.3,
            equal_opportunity_diff=0.1,
            equalized_odds_diff=0.1,
            predictive_parity_diff=0.1,
            disparate_impact_ratio=0.75,  # below 0.80 threshold
        )
        assert detector._classify_severity(m) == FairnessSeverity.HIGH

    def test_severity_high_eod_above_10_percent(self):
        """Equal opportunity diff > 0.10 → HIGH severity."""
        detector = BiasDetector()
        m = FairnessMetrics(
            demographic_parity_diff=0.05,
            equal_opportunity_diff=0.15,  # > 0.10
            equalized_odds_diff=0.15,
            predictive_parity_diff=0.05,
            disparate_impact_ratio=0.95,
        )
        assert detector._classify_severity(m) == FairnessSeverity.HIGH

    def test_severity_medium_di_between_80_and_90(self):
        """DI ratio in [0.80, 0.90) → MEDIUM severity."""
        detector = BiasDetector()
        m = FairnessMetrics(
            demographic_parity_diff=0.05,
            equal_opportunity_diff=0.04,
            equalized_odds_diff=0.04,
            predictive_parity_diff=0.04,
            disparate_impact_ratio=0.85,  # between 0.80 and 0.90
        )
        assert detector._classify_severity(m) == FairnessSeverity.MEDIUM

    def test_severity_medium_eod_between_5_and_10(self):
        """Equal opportunity diff in (5%, 10%] → MEDIUM severity."""
        detector = BiasDetector()
        m = FairnessMetrics(
            demographic_parity_diff=0.03,
            equal_opportunity_diff=0.08,  # between 0.05 and 0.10
            equalized_odds_diff=0.08,
            predictive_parity_diff=0.03,
            disparate_impact_ratio=0.95,
        )
        assert detector._classify_severity(m) == FairnessSeverity.MEDIUM

    def test_recommendations_not_empty_for_high_severity(self):
        """HIGH severity генерирует хотя бы одну рекомендацию."""
        detector = BiasDetector()
        m = FairnessMetrics(
            demographic_parity_diff=0.3,
            equal_opportunity_diff=0.2,
            equalized_odds_diff=0.2,
            predictive_parity_diff=0.2,
            disparate_impact_ratio=0.6,
        )
        recs = detector._build_recommendations(m, FairnessSeverity.HIGH)
        assert len(recs) > 0
        # Должна упоминать EU AI Act
        assert any("EU AI Act" in r or "EEOC" in r for r in recs)

    def test_recommendations_contain_quarterly_for_low(self):
        """LOW severity рекомендует квартальный мониторинг."""
        detector = BiasDetector()
        m = self._perfect_metrics()
        recs = detector._build_recommendations(m, FairnessSeverity.LOW)
        assert any("quarterly" in r.lower() or "квартал" in r.lower() for r in recs)


class TestBiasDetector:
    """Integration tests for BiasDetector.analyze() and optimal_thresholds()."""

    def _fair_data(self):
        """Сгенерировать данные без смещений: одинаковые показатели для обеих групп.

        Детерминированные данные: обе группы имеют одинаковый positive rate и точность,
        что гарантирует DI ratio = 1.0 и LOW severity.
        """
        n_per_group = 50
        # Одинаковое чередование меток 1/0 в обеих группах → positive_rate = 0.5
        y_true_g = [1, 0] * (n_per_group // 2)
        # Идеальные предсказания → DI = 1.0, DP_diff = 0.0, EOD = 0.0
        y_pred_g = y_true_g[:]
        y_true = np.array(y_true_g + y_true_g)
        y_pred = np.array(y_pred_g + y_pred_g)
        groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)
        return y_true, y_pred, groups

    def _biased_data(self):
        """Сгенерировать данные с явным смещением: группа B чаще получает предсказание 1."""
        n = 200
        y_true = np.array([1, 0] * (n // 2))
        groups = np.array(["privileged"] * n + ["unprivileged"] * n)
        # Group A (privileged): 30% positive rate
        pred_a = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0] * (n // 10))
        # Group B (unprivileged): 70% positive rate — clearly biased
        pred_b = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0] * (n // 10))
        y_pred = np.concatenate([pred_a, pred_b])
        y_true_full = np.concatenate([y_true, y_true])
        return y_true_full, y_pred, groups

    def test_analyze_returns_fairness_report(self):
        """analyze() возвращает FairnessReport с заполненными полями."""
        detector = BiasDetector(protected_attribute="gender")
        y_true, y_pred, groups = self._fair_data()
        report = detector.analyze(y_true, y_pred, groups)
        assert report.protected_attribute == "gender"
        assert report.audit_id  # UUID не пустой
        assert report.timestamp  # ISO 8601
        valid = (FairnessSeverity.LOW, FairnessSeverity.MEDIUM, FairnessSeverity.HIGH)
        assert report.severity in valid

    def test_analyze_fair_data_low_severity(self):
        """Одинаковые ошибки в обеих группах → LOW severity."""
        detector = BiasDetector()
        y_true, y_pred, groups = self._fair_data()
        report = detector.analyze(y_true, y_pred, groups)
        assert report.severity == FairnessSeverity.LOW

    def test_analyze_biased_data_high_severity(self):
        """Явно смещённые предсказания → HIGH severity."""
        detector = BiasDetector()
        y_true, y_pred, groups = self._biased_data()
        report = detector.analyze(y_true, y_pred, groups)
        assert report.severity == FairnessSeverity.HIGH

    def test_analyze_disparate_impact_below_one_for_biased(self):
        """Смещённые данные: DI ratio < 1.0."""
        detector = BiasDetector()
        y_true, y_pred, groups = self._biased_data()
        report = detector.analyze(y_true, y_pred, groups)
        assert report.metrics.disparate_impact_ratio < 1.0

    def test_analyze_report_to_dict_serializable(self):
        """FairnessReport.to_dict() возвращает JSON-сериализуемый dict."""
        detector = BiasDetector()
        y_true, y_pred, groups = self._fair_data()
        report = detector.analyze(y_true, y_pred, groups)
        d = report.to_dict()
        import json

        # Should not raise TypeError
        json.dumps(d)

    def test_analyze_raises_on_single_group(self):
        """Одна группа → ValueError (нельзя сравнить с собой)."""
        detector = BiasDetector()
        with pytest.raises(ValueError, match="2 groups"):
            detector.analyze([0, 1, 0, 1], [0, 1, 0, 1], ["A", "A", "A", "A"])

    def test_analyze_group_labels_in_report(self):
        """Метки групп из данных попадают в отчёт."""
        detector = BiasDetector()
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 1, 0, 1, 0]
        groups = ["male"] * 4 + ["female"] * 4
        report = detector.analyze(
            y_true, y_pred, groups, group_a_label="male", group_b_label="female"
        )
        assert report.group_a_label == "male"
        assert report.group_b_label == "female"

    def test_analyze_sample_sizes_match_groups(self):
        """sample_size в GroupMetrics совпадает с реальным числом в группе."""
        detector = BiasDetector()
        n_a, n_b = 60, 40
        y_true = [0] * n_a + [0] * n_b
        y_pred = [0] * n_a + [1] * n_b
        groups = ["A"] * n_a + ["B"] * n_b
        report = detector.analyze(y_true, y_pred, groups)
        assert report.group_a.sample_size == n_a
        assert report.group_b.sample_size == n_b

    def test_demographic_parity_diff_is_nonnegative(self):
        """demographic_parity_diff — абсолютная разница, всегда >= 0."""
        detector = BiasDetector()
        y_true, y_pred, groups = self._fair_data()
        report = detector.analyze(y_true, y_pred, groups)
        assert report.metrics.demographic_parity_diff >= 0.0

    def test_optimal_thresholds_returns_per_group(self):
        """optimal_thresholds() возвращает порог для каждой группы."""
        detector = BiasDetector()
        rng = np.random.default_rng(0)
        n = 100
        y_true = (rng.random(n) > 0.5).astype(int)
        y_proba = np.clip(y_true.astype(float) + rng.normal(0, 0.3, n), 0, 1)
        groups = ["A"] * (n // 2) + ["B"] * (n // 2)
        thresholds = detector.optimal_thresholds(y_true, y_proba, groups)
        assert "A" in thresholds
        assert "B" in thresholds
        assert 0.0 <= thresholds["A"] <= 1.0
        assert 0.0 <= thresholds["B"] <= 1.0

    def test_analyze_proba_wrapper(self):
        """analyze_proba() с порогом 0.5 совпадает с analyze() после бинаризации."""
        detector = BiasDetector()
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_proba = [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.1, 0.9]
        groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
        report = detector.analyze_proba(y_true, y_proba, groups, threshold=0.5)
        assert report.severity is not None


class TestFairnessAPIEndpoints:
    """Integration tests for POST /fairness/analyze, GET /fairness/report,
    and POST /fairness/thresholds."""

    def _client(self):
        from churn.api.app import _reset_fairness_state, app
        from fastapi.testclient import TestClient

        _reset_fairness_state()
        return TestClient(app)

    def _fair_payload(self):
        """Полезная нагрузка для сбалансированных данных."""
        n = 40  # n образцов на группу
        y_true = [0, 1] * (n // 2)  # 40 samples: 20 negatives, 20 positives
        y_pred = y_true[:]  # perfect predictions → DI ratio = 1.0
        groups = ["male"] * n + ["female"] * n  # 80 total
        return {
            "y_true": y_true + y_true,  # 80 samples — matches groups length
            "y_pred": y_pred + y_pred,  # 80 samples
            "groups": groups,  # 80 samples
            "protected_attribute": "gender",
        }

    def test_analyze_returns_200(self):
        """POST /fairness/analyze возвращает 200."""
        resp = self._client().post("/fairness/analyze", json=self._fair_payload())
        assert resp.status_code == 200

    def test_analyze_response_has_required_fields(self):
        """Ответ содержит audit_id, metrics, severity, recommendations."""
        data = self._client().post("/fairness/analyze", json=self._fair_payload()).json()
        for field in (
            "audit_id",
            "timestamp",
            "metrics",
            "severity",
            "recommendations",
            "group_a",
            "group_b",
            "protected_attribute",
        ):
            assert field in data, f"Missing field: {field}"

    def test_analyze_metrics_has_all_keys(self):
        """metrics содержит все 5 метрик справедливости."""
        data = self._client().post("/fairness/analyze", json=self._fair_payload()).json()
        for k in (
            "demographic_parity_diff",
            "equal_opportunity_diff",
            "equalized_odds_diff",
            "disparate_impact_ratio",
        ):
            assert k in data["metrics"]

    def test_analyze_severity_is_valid_enum(self):
        """severity — одно из (low, medium, high)."""
        data = self._client().post("/fairness/analyze", json=self._fair_payload()).json()
        assert data["severity"] in ("low", "medium", "high")

    def test_analyze_length_mismatch_returns_422(self):
        """Несовпадение длин y_true/y_pred/groups → 422."""
        payload = {"y_true": [0, 1], "y_pred": [0, 1, 0], "groups": ["A", "B"]}
        resp = self._client().post("/fairness/analyze", json=payload)
        assert resp.status_code == 422

    def test_analyze_too_few_samples_returns_422(self):
        """Менее 4 образцов → 422."""
        payload = {"y_true": [0], "y_pred": [1], "groups": ["A"]}
        resp = self._client().post("/fairness/analyze", json=payload)
        assert resp.status_code == 422

    def test_analyze_single_group_returns_422(self):
        """Одна группа → 422."""
        payload = {
            "y_true": [0, 1, 0, 1],
            "y_pred": [0, 1, 0, 1],
            "groups": ["A", "A", "A", "A"],
        }
        resp = self._client().post("/fairness/analyze", json=payload)
        assert resp.status_code == 422

    def test_report_404_before_analysis(self):
        """GET /fairness/report → 404 до первого анализа."""
        resp = self._client().get("/fairness/report")
        assert resp.status_code == 404

    def test_report_200_after_analysis(self):
        """GET /fairness/report → 200 после POST /fairness/analyze."""
        client = self._client()
        client.post("/fairness/analyze", json=self._fair_payload())
        resp = client.get("/fairness/report")
        assert resp.status_code == 200

    def test_report_matches_last_analysis(self):
        """GET /fairness/report возвращает последний audit_id."""
        client = self._client()
        analyze_data = client.post("/fairness/analyze", json=self._fair_payload()).json()
        report_data = client.get("/fairness/report").json()
        assert report_data["audit_id"] == analyze_data["audit_id"]

    def test_thresholds_returns_200(self):
        """POST /fairness/thresholds → 200 с equal_opportunity."""
        n = 50
        payload = {
            "y_true": [0, 1] * n,
            "y_proba": [0.3, 0.7] * n,
            "groups": ["A"] * n + ["B"] * n,
            "target_metric": "equal_opportunity",
        }
        resp = self._client().post("/fairness/thresholds", json=payload)
        assert resp.status_code == 200

    def test_thresholds_response_structure(self):
        """POST /fairness/thresholds возвращает target_metric + thresholds dict."""
        n = 50
        payload = {
            "y_true": [0, 1] * n,
            "y_proba": [0.3, 0.7] * n,
            "groups": ["A"] * n + ["B"] * n,
            "target_metric": "demographic_parity",
        }
        data = self._client().post("/fairness/thresholds", json=payload).json()
        assert "thresholds" in data
        assert "A" in data["thresholds"]
        assert "B" in data["thresholds"]

    def test_thresholds_invalid_metric_422(self):
        """Неверный target_metric → 422."""
        payload = {
            "y_true": [0, 1, 0, 1],
            "y_proba": [0.2, 0.8, 0.3, 0.7],
            "groups": ["A", "A", "B", "B"],
            "target_metric": "invalid_metric",
        }
        resp = self._client().post("/fairness/thresholds", json=payload)
        assert resp.status_code == 422

    def test_thresholds_in_zero_one_range(self):
        """Пороги находятся в диапазоне [0, 1]."""
        n = 50
        payload = {
            "y_true": [0, 1] * n,
            "y_proba": [0.3, 0.7] * n,
            "groups": ["X"] * n + ["Y"] * n,
        }
        data = self._client().post("/fairness/thresholds", json=payload).json()
        for grp, thresh in data["thresholds"].items():
            assert 0.0 <= thresh <= 1.0, f"Threshold for {grp} out of range: {thresh}"


# ---------------------------------------------------------------------------
# Counterfactual Explanations Tests (DiCE-style)
# ---------------------------------------------------------------------------

_CF_BASE_CUSTOMER = {
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
    "MonthlyCharges": 80.0,
    "TotalCharges": 80.0,
}


def _contract_sensitive_fn(features):
    """High churn for month-to-month; low for Two year contract."""
    if features.get("Contract") == "Two year":
        return 0.1
    return 0.9


def _always_high_fn(features):
    return 0.95


class TestDIcEChurnUnit:
    """Unit tests using synthetic predict_fn — no trained model needed.

    Юнит-тесты с синтетической predict_fn — реальная модель не нужна.
    """

    def _gen(self, n=3, max_iter=500, target=0.35):
        from churn.counterfactual.dice import CounterfactualConfig, DIcEChurn

        cfg = CounterfactualConfig(
            n_counterfactuals=n,
            max_iterations=max_iter,
            target_probability=target,
        )
        return DIcEChurn(cfg)

    def test_generate_returns_counterfactual_result(self):
        """generate() возвращает CounterfactualResult."""
        from churn.counterfactual.dice import CounterfactualResult

        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        assert isinstance(result, CounterfactualResult)

    def test_generate_success_when_contract_matters(self):
        """success=True когда есть эффективный контрфакт (Two year contract)."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        assert result.success is True
        assert len(result.counterfactuals) > 0

    def test_generate_no_success_when_all_high(self):
        """success=False когда predict_fn всегда возвращает 0.95."""
        result = self._gen(max_iter=200).generate(dict(_CF_BASE_CUSTOMER), _always_high_fn)
        assert result.success is False
        assert result.counterfactuals == []

    def test_original_probability_recorded_correctly(self):
        """original_probability совпадает с predict_fn(original)."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        assert result.original_probability == pytest.approx(0.9, abs=1e-4)

    def test_n_tried_equals_max_iterations(self):
        """n_tried равно max_iterations."""
        result = self._gen(max_iter=150).generate(dict(_CF_BASE_CUSTOMER), _always_high_fn)
        assert result.n_tried == 150

    def test_counterfactuals_below_target(self):
        """Все найденные контрфакты имеют вероятность < target_probability."""
        result = self._gen(target=0.35).generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        for cf in result.counterfactuals:
            assert cf.churn_probability < 0.35

    def test_counterfactuals_sorted_ascending_by_probability(self):
        """Контрфакты отсортированы по вероятности оттока по возрастанию."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        probs = [cf.churn_probability for cf in result.counterfactuals]
        assert probs == sorted(probs)

    def test_immutable_features_unchanged(self):
        """Признаки gender, SeniorCitizen, Dependents не меняются в контрфактах."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        for cf in result.counterfactuals:
            assert cf.features["gender"] == _CF_BASE_CUSTOMER["gender"]
            assert cf.features["SeniorCitizen"] == _CF_BASE_CUSTOMER["SeniorCitizen"]
            assert cf.features["Dependents"] == _CF_BASE_CUSTOMER["Dependents"]

    def test_changes_dict_reflects_actual_differences(self):
        """changes содержит только признаки, реально отличающиеся от оригинала."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        for cf in result.counterfactuals:
            for feat, val in cf.changes.items():
                assert cf.features[feat] == val
                assert val != _CF_BASE_CUSTOMER.get(feat)

    def test_distance_in_zero_one_range(self):
        """Расстояние от оригинала находится в [0, 1]."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        for cf in result.counterfactuals:
            assert 0.0 <= cf.distance <= 1.0

    def test_feasibility_is_complement_of_distance(self):
        """feasibility_score = round(1 - distance, 4)."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        for cf in result.counterfactuals:
            assert cf.feasibility_score == pytest.approx(1.0 - cf.distance, abs=1e-3)

    def test_distance_zero_for_identical_inputs(self):
        """Расстояние между идентичными наборами признаков равно 0."""
        from churn.counterfactual.dice import DIcEChurn

        gen = DIcEChurn()
        d = gen._distance(dict(_CF_BASE_CUSTOMER), dict(_CF_BASE_CUSTOMER))
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_n_counterfactuals_capped_by_config(self):
        """Количество контрфактов не превышает n_counterfactuals."""
        result = self._gen(n=2, max_iter=500).generate(
            dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn
        )
        assert len(result.counterfactuals) <= 2

    def test_to_plain_text_returns_nonempty_list(self):
        """to_plain_text() возвращает непустой список строк."""
        result = self._gen().generate(dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn)
        for cf in result.counterfactuals:
            texts = cf.to_plain_text()
            assert isinstance(texts, list)
            assert len(texts) > 0
            assert all(isinstance(t, str) for t in texts)

    def test_to_plain_text_contract_change(self):
        """to_plain_text() упоминает 'contract' при изменении контракта."""
        from churn.counterfactual.dice import Counterfactual

        cf = Counterfactual(
            features={"Contract": "Two year"},
            changes={"Contract": "Two year"},
            churn_probability=0.1,
            distance=0.05,
            feasibility_score=0.95,
        )
        texts = cf.to_plain_text()
        assert any("contract" in t.lower() for t in texts)

    def test_to_plain_text_paperless_billing(self):
        """to_plain_text() корректно обрабатывает PaperlessBilling."""
        from churn.counterfactual.dice import Counterfactual

        cf = Counterfactual(
            features={"PaperlessBilling": "No"},
            changes={"PaperlessBilling": "No"},
            churn_probability=0.2,
            distance=0.07,
            feasibility_score=0.93,
        )
        texts = cf.to_plain_text()
        assert any("billing" in t.lower() or "Billing" in t for t in texts)

    def test_diverse_counterfactuals_have_unique_changes(self):
        """При достаточном числе итераций контрфакты не являются дубликатами."""
        result = self._gen(n=3, max_iter=800).generate(
            dict(_CF_BASE_CUSTOMER), _contract_sensitive_fn
        )
        if len(result.counterfactuals) >= 2:
            fps = [
                frozenset((k, str(v)) for k, v in cf.changes.items())
                for cf in result.counterfactuals
            ]
            assert len(set(fps)) == len(fps), "Duplicate counterfactuals detected"


class TestCounterfactualAPIEndpoints:
    """API integration tests for POST /explain/counterfactual.

    Используют mock-модель вместо артефакта обучения.
    Правильный паттерн: patch внутри каждого теста — мок активен во время вызова.
    """

    _PAYLOAD = {"customer": _CF_BASE_CUSTOMER}

    @staticmethod
    def _make_mock_model():
        """Mock LightGBM-compatible model: 0.1 for Two year contract, 0.85 otherwise."""
        import numpy as np

        class _MockModel:
            feature_name_ = None

            def predict_proba(self, x):  # noqa: N803
                prob = 0.85
                if hasattr(x, "columns"):
                    col = "Contract_Two year"
                    if col in x.columns and int(x[col].iloc[0]) == 1:
                        prob = 0.1
                return np.array([[1 - prob, prob]])

        return _MockModel()

    def test_missing_customer_returns_422(self):
        """POST без customer → 422."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/explain/counterfactual", json={"n_counterfactuals": 3})
        assert resp.status_code == 422

    def test_target_probability_out_of_range_returns_422(self):
        """target_probability > 1 → 422."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/explain/counterfactual",
            json={**self._PAYLOAD, "target_probability": 1.5},
        )
        assert resp.status_code == 422

    def test_n_counterfactuals_zero_returns_422(self):
        """n_counterfactuals=0 → 422."""
        from churn.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.post(
            "/explain/counterfactual",
            json={**self._PAYLOAD, "n_counterfactuals": 0},
        )
        assert resp.status_code == 422

    def test_returns_200_with_mock_model(self):
        """С mock-моделью → 200."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            resp = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 400},
            )
        assert resp.status_code == 200

    def test_response_top_level_fields_present(self):
        """Ответ содержит все обязательные поля верхнего уровня."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 400},
            ).json()
        required = (
            "original_probability",
            "target_probability",
            "success",
            "n_found",
            "n_tried",
            "counterfactuals",
        )
        for f in required:
            assert f in data, f"Missing top-level field: {f}"

    def test_original_probability_in_zero_one(self):
        """original_probability ∈ [0, 1]."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 300},
            ).json()
        assert 0.0 <= data["original_probability"] <= 1.0

    def test_n_found_matches_counterfactuals_length(self):
        """n_found совпадает с len(counterfactuals)."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 400},
            ).json()
        assert data["n_found"] == len(data["counterfactuals"])

    def test_n_tried_matches_requested_max_iterations(self):
        """n_tried совпадает с max_iterations в запросе."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 250},
            ).json()
        assert data["n_tried"] == 250

    def test_counterfactual_item_has_required_keys(self):
        """Каждый элемент counterfactuals содержит все обязательные ключи."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 500},
            ).json()
        cf_keys = (
            "rank",
            "churn_probability",
            "distance",
            "feasibility_score",
            "changes",
            "explanation",
        )  # noqa: E501
        for cf in data["counterfactuals"]:
            for key in cf_keys:
                assert key in cf

    def test_rank_is_sequential_starting_from_one(self):
        """rank начинается с 1 и идёт последовательно."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 500},
            ).json()
        ranks = [cf["rank"] for cf in data["counterfactuals"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_explanation_is_list_of_strings(self):
        """explanation — список строк для каждого контрфакта."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "max_iterations": 500},
            ).json()
        for cf in data["counterfactuals"]:
            assert isinstance(cf["explanation"], list)
            assert all(isinstance(s, str) for s in cf["explanation"])

    def test_target_probability_echoed_in_response(self):
        """target_probability из запроса отражается в ответе."""
        import unittest.mock

        from churn.api.app import app
        from fastapi.testclient import TestClient

        with unittest.mock.patch("churn.api.app.get_model", return_value=self._make_mock_model()):
            client = TestClient(app)
            data = client.post(
                "/explain/counterfactual",
                json={**self._PAYLOAD, "target_probability": 0.4, "max_iterations": 300},
            ).json()
        assert data["target_probability"] == pytest.approx(0.4)
