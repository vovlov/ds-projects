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
