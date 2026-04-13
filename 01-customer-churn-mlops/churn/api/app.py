"""FastAPI prediction service for customer churn model."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data.load import CATEGORICAL_FEATURES, add_features
from ..retraining.trigger import PSI_YELLOW

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using CatBoost model",
    version="1.0.0",
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.pkl"

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(500, "Model not found. Run training first.")
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


class CustomerInput(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[844.2])


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


# ---------------------------------------------------------------------------
# Retraining notification endpoint (receives alerts from Project 10)
# ---------------------------------------------------------------------------


class DriftAlertPayload(BaseModel):
    """Payload от системы алертинга Project 10 (Data Quality Platform).
    Payload from Project 10's drift alerting system.

    Project 10 отправляет этот запрос на /retraining/notify когда обнаруживает
    дрейф в признаках, которые использует churn-модель.

    Project 10 sends this to /retraining/notify when it detects drift
    in features used by the churn model.
    """

    severity: str = Field(
        ...,
        description="Серьёзность дрейфа / Drift severity: ok | warning | critical",
    )
    features_drifted: list[str] = Field(
        default_factory=list,
        description="Список признаков с дрейфом / Features with detected drift",
    )
    max_psi: float = Field(
        ...,
        ge=0.0,
        description="Максимальный PSI среди всех признаков / Max PSI across features",
    )
    columns_checked: int = Field(default=0, description="Сколько столбцов проверено")
    columns_with_drift: int = Field(default=0, description="Сколько столбцов с дрейфом")
    timestamp: str = Field(..., description="ISO-8601 timestamp алерта")
    source: str = Field(
        default="data-quality-platform",
        description="Источник алерта / Alert source",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Полный drift-отчёт для аудит-трейла / Full drift report",
    )


class RetrainingDecision(BaseModel):
    """Решение о переобучении модели оттока / Churn model retraining decision."""

    decision: str = Field(..., description="'retrain' или 'skip'")
    reason: str = Field(..., description="Человекочитаемое объяснение решения")
    severity: str = Field(..., description="Серьёзность из входящего алерта")
    max_psi: float = Field(..., description="PSI из входящего алерта")
    triggered_by: str = Field(..., description="Источник алерта")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/retraining/notify", response_model=RetrainingDecision)
def retraining_notify(payload: DriftAlertPayload) -> RetrainingDecision:
    """Принять алерт о дрейфе от Data Quality Platform (Project 10).
    Accept drift alert from Data Quality Platform (Project 10).

    Конечная точка замыкает цикл мониторинга:
      Quality Platform обнаруживает дрейф → отправляет сюда POST →
      этот эндпоинт решает: переобучать модель или нет.

    Closes the monitoring loop:
      Quality Platform detects drift → POSTs here →
      this endpoint decides whether to trigger retraining.

    Логика принятия решения (OR):
    - severity == "critical" (PSI >= 0.25): немедленное переобучение
    - severity == "warning" AND max_psi >= PSI_YELLOW (0.2): переобучение
    - иначе: пропустить (drift есть, но ниже порога действия)

    Decision logic (OR):
    - severity == "critical" (PSI >= 0.25): immediate retrain
    - severity == "warning" AND max_psi >= PSI_YELLOW (0.2): retrain
    - otherwise: skip
    """
    # Критический дрейф: переобучать всегда
    # Critical drift: always retrain
    if payload.severity == "critical":
        return RetrainingDecision(
            decision="retrain",
            reason=(
                f"Critical drift detected by {payload.source}: "
                f"max_psi={payload.max_psi:.4f} >= 0.25 "
                f"in features={payload.features_drifted}"
            ),
            severity=payload.severity,
            max_psi=payload.max_psi,
            triggered_by=payload.source,
        )

    # Умеренный дрейф: переобучать если PSI >= BCBS-порог (0.2)
    # Moderate drift: retrain if PSI >= BCBS threshold (0.2)
    if payload.severity == "warning" and payload.max_psi >= PSI_YELLOW:
        return RetrainingDecision(
            decision="retrain",
            reason=(
                f"Warning drift from {payload.source}: "
                f"max_psi={payload.max_psi:.4f} >= {PSI_YELLOW} (BCBS threshold) "
                f"in features={payload.features_drifted}"
            ),
            severity=payload.severity,
            max_psi=payload.max_psi,
            triggered_by=payload.source,
        )

    # Дрейф есть но ниже порога действия — мониторим, не переобучаем
    # Drift below action threshold — monitor but don't retrain yet
    return RetrainingDecision(
        decision="skip",
        reason=(
            f"Drift from {payload.source} below retraining threshold: "
            f"severity={payload.severity}, max_psi={payload.max_psi:.4f} < {PSI_YELLOW}"
        ),
        severity=payload.severity,
        max_psi=payload.max_psi,
        triggered_by=payload.source,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    model = get_model()

    # Reconstruct a single-row DataFrame matching the training pipeline
    row = {
        "customerID": "api-request",
        "gender": customer.gender,
        "SeniorCitizen": customer.SeniorCitizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "tenure": customer.tenure,
        "PhoneService": customer.PhoneService,
        "MultipleLines": customer.MultipleLines,
        "InternetService": customer.InternetService,
        "OnlineSecurity": customer.OnlineSecurity,
        "OnlineBackup": customer.OnlineBackup,
        "DeviceProtection": customer.DeviceProtection,
        "TechSupport": customer.TechSupport,
        "StreamingTV": customer.StreamingTV,
        "StreamingMovies": customer.StreamingMovies,
        "Contract": customer.Contract,
        "PaperlessBilling": customer.PaperlessBilling,
        "PaymentMethod": customer.PaymentMethod,
        "MonthlyCharges": customer.MonthlyCharges,
        "TotalCharges": customer.TotalCharges,
        "Churn": 0,
    }
    df = pl.DataFrame([row])
    df = add_features(df)

    # One-hot encode categoricals (same as LightGBM training)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns] + ["TenureGroup"]
    df_enc = df.to_dummies(columns=cat_cols)
    feature_cols = [c for c in df_enc.columns if c not in ("customerID", "Churn")]
    X = df_enc.select(feature_cols).to_pandas()

    # Align columns with model (add missing, drop extra)
    if hasattr(model, "feature_name_"):
        model_cols = model.feature_name_
        for col in model_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[model_cols]

    proba = float(model.predict_proba(X)[0][1])
    prediction = proba >= 0.5

    if proba >= 0.7:
        risk = "high"
    elif proba >= 0.4:
        risk = "medium"
    else:
        risk = "low"

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_level=risk,
    )
