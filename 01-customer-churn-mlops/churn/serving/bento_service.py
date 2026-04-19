"""
BentoML serving для Customer Churn модели.

Зачем BentoML поверх FastAPI?
FastAPI — transport layer (HTTP routing, validation).
BentoML — serving platform:
  - Adaptive batching: группирует concurrent запросы → лучший CPU-throughput
  - Built-in /healthz, /readyz, /metrics (Prometheus) endpoints
  - BentoML model store: версионирование артефактов отдельно от кода
  - One-command containerization: bentoml build && bentoml containerize

Паттерн: ChurnPredictor — transport-agnostic core.
FastAPI и BentoML сервисы используют один predictor → идентичные результаты.

Deploy: bentoml serve churn.serving.bento_service:ChurnService --reload
Build:  bentoml build && bentoml containerize churn_predictor:latest
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Graceful degradation: сервис работает и без установленного bentoml.
# В production-среде bentoml устанавливается отдельно через extras.
try:
    import bentoml

    BENTOML_AVAILABLE = True
except ImportError:
    BENTOML_AVAILABLE = False


def is_available() -> bool:
    """Check if bentoml is installed in the current environment."""
    return BENTOML_AVAILABLE


# ---------------------------------------------------------------------------
# Transport-agnostic data contracts
# ---------------------------------------------------------------------------


@dataclass
class ChurnInput:
    """Входные признаки для предсказания оттока / Input features for churn prediction.

    Зеркалит CustomerInput из FastAPI app, но как dataclass —
    не требует Pydantic и работает в любом transport layer (HTTP/gRPC/batch).
    """

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@dataclass
class ChurnPrediction:
    """Результат предсказания оттока / Churn prediction result.

    Добавляет model_version по сравнению с FastAPI PredictionResponse —
    важно для A/B тестирования и аудит-трейла в production.
    """

    churn_probability: float
    churn_prediction: bool
    risk_level: str
    model_version: str = "v1"


# ---------------------------------------------------------------------------
# Core predictor — transport-agnostic
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.pkl"


class ChurnPredictor:
    """
    Transport-agnostic churn prediction engine.

    Реализует prediction pipeline независимо от transport layer.
    FastAPI /predict и BentoML ChurnService.predict используют один predictor,
    гарантируя битовое совпадение результатов между endpoints.

    Lazy model loading: модель загружается при первом вызове predict(),
    а не при инициализации — совместимо с BentoML lifecycle management.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        model_version: str = "v1",
    ) -> None:
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._model_version = model_version
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy-load pickle model on first prediction call."""
        if self._model is None:
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model artifact not found at {self._model_path}. "
                    "Run train.py first to generate the artifact."
                )
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)
        return self._model

    @staticmethod
    def _classify_risk(proba: float) -> str:
        """Map churn probability to business risk tier."""
        if proba >= 0.7:
            return "high"
        if proba >= 0.4:
            return "medium"
        return "low"

    def predict(self, inp: ChurnInput) -> ChurnPrediction:
        """Predict churn probability for a single customer.

        Воспроизводит preprocessing pipeline из FastAPI /predict endpoint:
        add_features() → one-hot encoding → column alignment.
        Полная идентичность результатов между FastAPI и BentoML endpoints.
        """
        import polars as pl

        from ..data.load import CATEGORICAL_FEATURES, add_features

        model = self._load_model()

        row: dict[str, Any] = {
            "customerID": "bento-request",
            "gender": inp.gender,
            "SeniorCitizen": inp.SeniorCitizen,
            "Partner": inp.Partner,
            "Dependents": inp.Dependents,
            "tenure": inp.tenure,
            "PhoneService": inp.PhoneService,
            "MultipleLines": inp.MultipleLines,
            "InternetService": inp.InternetService,
            "OnlineSecurity": inp.OnlineSecurity,
            "OnlineBackup": inp.OnlineBackup,
            "DeviceProtection": inp.DeviceProtection,
            "TechSupport": inp.TechSupport,
            "StreamingTV": inp.StreamingTV,
            "StreamingMovies": inp.StreamingMovies,
            "Contract": inp.Contract,
            "PaperlessBilling": inp.PaperlessBilling,
            "PaymentMethod": inp.PaymentMethod,
            "MonthlyCharges": inp.MonthlyCharges,
            "TotalCharges": inp.TotalCharges,
            "Churn": 0,
        }

        df = pl.DataFrame([row])
        df = add_features(df)

        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns] + ["TenureGroup"]
        df_enc = df.to_dummies(columns=cat_cols)
        feature_cols = [c for c in df_enc.columns if c not in ("customerID", "Churn")]
        X = df_enc.select(feature_cols).to_pandas()

        # Align feature columns with what model was trained on
        if hasattr(model, "feature_name_"):
            model_cols = model.feature_name_
            for col in model_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[model_cols]

        proba = float(model.predict_proba(X)[0][1])

        return ChurnPrediction(
            churn_probability=round(proba, 4),
            churn_prediction=proba >= 0.5,
            risk_level=self._classify_risk(proba),
            model_version=self._model_version,
        )

    def predict_batch(self, inputs: list[ChurnInput]) -> list[ChurnPrediction]:
        """Batch prediction for multiple customers.

        Используется BentoML adaptive batching: клиентские запросы
        группируются в батчи до max_batch_size или max_latency_ms.
        Throughput растёт линейно без изменений кода predictor'а.
        """
        return [self.predict(inp) for inp in inputs]


# ---------------------------------------------------------------------------
# BentoML model store helper
# ---------------------------------------------------------------------------


def save_to_bentoml(
    model_path: Path | None = None,
    model_name: str = "churn_model",
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Save trained churn model to BentoML local model store.

    BentoML model store версионирует артефакты независимо от кода —
    можно откатиться к предыдущей версии без git revert.

    Returns:
        Model tag string (e.g. 'churn_model:a1b2c3d4') или None если
        bentoml не установлен или артефакт отсутствует.
    """
    if not BENTOML_AVAILABLE:
        return None

    path = model_path or _DEFAULT_MODEL_PATH
    if not path.exists():
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)

    saved = bentoml.picklable_model.save_model(
        model_name,
        model,
        signatures={"predict_proba": {"batchable": True, "batch_dim": 0}},
        metadata=metadata
        or {
            "task": "binary_classification",
            "dataset": "telco-churn",
            "metrics": {"roc_auc": "see MLflow for latest"},
        },
    )
    return str(saved.tag)


# ---------------------------------------------------------------------------
# BentoML service — conditionally defined when bentoml is installed.
# ---------------------------------------------------------------------------

if BENTOML_AVAILABLE:

    @bentoml.service(  # type: ignore[misc]
        name="churn_predictor",
        traffic={"timeout": 10, "max_concurrency": 32},
    )
    class ChurnService:
        """
        BentoML production service для предсказания оттока клиентов.

        Преимущества перед чистым FastAPI:
        - Adaptive batching: /predict_batch группирует concurrent запросы
          в батчи до 32 строк → лучший CPU throughput при retention-кампаниях
        - BentoML model store: артефакты версионированы, откат без git revert
        - Built-in health/readiness checks и Prometheus /metrics
        - One-command Docker: bentoml build && bentoml containerize churn_predictor

        Usage:
            bentoml serve churn.serving.bento_service:ChurnService --reload
            bentoml build && bentoml containerize churn_predictor:latest
        """

        def __init__(self) -> None:
            self.predictor = ChurnPredictor()

        @bentoml.api()  # type: ignore[misc]
        def predict(self, inp: ChurnInput) -> ChurnPrediction:
            """Single-customer churn prediction."""
            return self.predictor.predict(inp)

        @bentoml.api(batchable=True, max_batch_size=32, max_latency_ms=100)  # type: ignore[misc]
        def predict_batch(self, inputs: list[ChurnInput]) -> list[ChurnPrediction]:
            """Batch prediction for retention campaigns.

            Используй для bulk-скоринга (ночные батчи) или
            когда ожидается высокий concurrent трафик.
            """
            return self.predictor.predict_batch(inputs)

        @bentoml.api()  # type: ignore[misc]
        def health(self) -> dict[str, str]:
            """Liveness probe endpoint."""
            return {"status": "healthy", "service": "churn_predictor"}
