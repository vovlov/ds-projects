"""FastAPI prediction service for customer churn model."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..ab_testing.experiment import ABExperiment, VariantConfig
from ..data.load import CATEGORICAL_FEATURES, add_features
from ..evaluation.model_comparison import (
    ModelResult,
    compare_models,
    generate_json_report,
    generate_markdown_report,
)
from ..retraining.trigger import PSI_YELLOW

logger = logging.getLogger(__name__)

# Глобальный эксперимент: control (v1) vs treatment (v2).
# В production заменяется на Redis-backed store для горизонтального масштабирования.
_experiment = ABExperiment(
    variants=[
        VariantConfig("control", 0.5, "v1", "current production model"),
        VariantConfig("treatment", 0.5, "v2", "challenger model"),
    ]
)

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


# ---------------------------------------------------------------------------
# A/B Testing endpoints
# ---------------------------------------------------------------------------


class ABPredictRequest(BaseModel):
    """Запрос на предсказание с A/B роутингом / A/B routed prediction request."""

    customer_id: str = Field(
        ..., description="Уникальный ID клиента для детерминированного роутинга"
    )
    customer: CustomerInput


class ABPredictionResponse(BaseModel):
    """Ответ с A/B мета-данными / Prediction response with A/B metadata."""

    churn_probability: float
    churn_prediction: bool
    risk_level: str
    variant: str = Field(..., description="Вариант A/B эксперимента (control/treatment)")
    model_version: str = Field(..., description="Версия модели обслуживающей вариант")


class OutcomeRequest(BaseModel):
    """Запись фактического исхода для клиента / Record ground-truth outcome."""

    customer_id: str = Field(..., description="ID клиента из предыдущего /ab/predict запроса")
    actual_churn: bool = Field(..., description="Фактически ли клиент ушёл")


class OutcomeResponse(BaseModel):
    """Ответ на запись исхода / Outcome recording response."""

    recorded: bool
    customer_id: str
    message: str


class ABResultsResponse(BaseModel):
    """Статистические результаты эксперимента / Experiment statistical results."""

    status: str
    winner: str | None
    p_value_rate: float | None = Field(None, description="Chi-squared p-value for high-risk rate")
    p_value_prob: float | None = Field(None, description="Welch t-test p-value for churn prob")
    control_n: int
    treatment_n: int
    control_high_risk_rate: float
    treatment_high_risk_rate: float
    relative_effect_pct: float | None
    recommendation: str
    scipy_available: bool


@app.post("/ab/predict", response_model=ABPredictionResponse)
def ab_predict(request: ABPredictRequest) -> ABPredictionResponse:
    """Предсказать отток с детерминированным A/B роутингом по customer_id.

    Predict churn with deterministic A/B routing by customer_id.

    Один и тот же customer_id всегда попадает в один вариант —
    нет switching noise при повторных обращениях клиента.
    Предсказание записывается для последующего статистического анализа.
    """
    model = get_model()
    customer = request.customer

    row: dict[str, Any] = {
        "customerID": request.customer_id,
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
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns] + ["TenureGroup"]
    df_enc = df.to_dummies(columns=cat_cols)
    feature_cols = [c for c in df_enc.columns if c not in ("customerID", "Churn")]
    X = df_enc.select(feature_cols).to_pandas()

    if hasattr(model, "feature_name_"):
        model_cols = model.feature_name_
        for col in model_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[model_cols]

    proba = float(model.predict_proba(X)[0][1])
    risk = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"

    # Детерминированный роутинг и запись результата
    variant = _experiment.route(request.customer_id)
    _experiment.record_prediction(request.customer_id, variant, round(proba, 4), risk)

    # Версия модели берётся из конфигурации варианта
    variant_cfg = next((v for v in _experiment._variants if v.name == variant), None)
    model_version = variant_cfg.model_version if variant_cfg else "v1"

    return ABPredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= 0.5,
        risk_level=risk,
        variant=variant,
        model_version=model_version,
    )


@app.post("/ab/outcome", response_model=OutcomeResponse)
def ab_outcome(request: OutcomeRequest) -> OutcomeResponse:
    """Записать фактический исход оттока для клиента.

    Record actual churn outcome for a customer.
    Ground truth поступает с задержкой (дни/недели после предсказания) —
    этот endpoint замыкает петлю обратной связи для outcome-based анализа.
    """
    recorded = _experiment.record_outcome(request.customer_id, request.actual_churn)
    return OutcomeResponse(
        recorded=recorded,
        customer_id=request.customer_id,
        message=(
            "Outcome recorded successfully."
            if recorded
            else f"No pending prediction found for customer_id='{request.customer_id}'."
        ),
    )


@app.get("/ab/results", response_model=ABResultsResponse)
def ab_results() -> ABResultsResponse:
    """Получить статистические результаты текущего A/B эксперимента.

    Get statistical results of the current A/B experiment.

    Возвращает p-value, победителя и рекомендацию.
    Для значимых выводов нужно >= 385 предсказаний на вариант.
    """
    result = _experiment.compute_results()
    return ABResultsResponse(
        status=result.status,
        winner=result.winner,
        p_value_rate=result.p_value_rate,
        p_value_prob=result.p_value_prob,
        control_n=result.control_stats.n_predictions,
        treatment_n=result.treatment_stats.n_predictions,
        control_high_risk_rate=result.control_stats.high_risk_rate,
        treatment_high_risk_rate=result.treatment_stats.high_risk_rate,
        relative_effect_pct=result.relative_effect,
        recommendation=result.recommendation,
        scipy_available=result.scipy_available,
    )


@app.get("/ab/status")
def ab_status() -> dict[str, Any]:
    """Краткая сводка состояния эксперимента для дашбордов.

    Quick experiment status overview for monitoring dashboards.
    """
    return _experiment.get_status_summary()


@app.post("/ab/reset")
def ab_reset() -> dict[str, str]:
    """Сбросить эксперимент: удалить все собранные предсказания.

    Reset experiment — clears all collected predictions.
    Используется при запуске нового эксперимента.
    """
    _experiment.reset()
    return {"status": "reset", "message": "Experiment data cleared. Ready for new experiment."}


# ---------------------------------------------------------------------------
# Model comparison endpoints
# ---------------------------------------------------------------------------

# In-memory cache for the last comparison report (replaced on each POST).
# In production, this would be persisted to MLflow or a database.
_last_comparison_report: dict[str, Any] | None = None


class ModelResultInput(BaseModel):
    """Input schema for a single model's metrics.
    Входные данные метрик одной модели.
    """

    name: str = Field(..., description="Уникальное имя модели / Unique model name")
    roc_auc: float = Field(..., ge=0.0, le=1.0, description="ROC AUC на тестовой выборке")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1-score (macro)")
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    training_time_sec: float = Field(default=0.0, ge=0.0)
    params: dict[str, Any] = Field(default_factory=dict)
    feature_importances: dict[str, float] = Field(default_factory=dict)
    run_id: str | None = Field(default=None, description="MLflow run ID")


class CompareModelsRequest(BaseModel):
    """Request body for POST /compare/models."""

    models: list[ModelResultInput] = Field(
        ...,
        min_length=1,
        description="Список результатов моделей для сравнения / List of model results to compare",
    )
    format: str = Field(
        default="json",
        description="Формат отчёта: 'json' или 'markdown' / Report format: 'json' or 'markdown'",
    )


@app.post("/compare/models")
def compare_models_endpoint(request: CompareModelsRequest) -> dict[str, Any]:
    """Сравнить несколько обученных моделей и вернуть отчёт с ранжированием.

    Compare multiple trained models and return a ranking report.

    Ранжирует по ROC AUC (primary) и F1 (tiebreaker). Победитель считается
    значимым, если разница в AUC >= 0.02 (Hanley & McNeil 1982, ~2 SE AUC).
    Отчёт кэшируется и доступен через GET /compare/report.

    Ranks by ROC AUC (primary) and F1 (tiebreaker). Winner is declared
    significant when AUC gap >= 0.02. Report is cached for GET /compare/report.
    """
    global _last_comparison_report

    results = [
        ModelResult(
            name=m.name,
            roc_auc=m.roc_auc,
            f1_score=m.f1_score,
            precision=m.precision,
            recall=m.recall,
            training_time_sec=m.training_time_sec,
            params=m.params,
            feature_importances=m.feature_importances,
            run_id=m.run_id,
        )
        for m in request.models
    ]

    report = compare_models(results)
    json_report = generate_json_report(report)
    _last_comparison_report = json_report

    if request.format == "markdown":
        return {
            **json_report,
            "markdown": generate_markdown_report(report),
        }

    return json_report


@app.get("/compare/report")
def get_comparison_report() -> dict[str, Any]:
    """Вернуть последний закэшированный отчёт сравнения моделей.

    Return the last cached model comparison report.

    Отчёт создаётся через POST /compare/models и сохраняется до следующего вызова.
    Report is created via POST /compare/models and persists until next call.
    """
    if _last_comparison_report is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=404,
            detail="No comparison report available. POST to /compare/models first.",
        )
    return _last_comparison_report


@app.get("/compare/leaderboard")
def get_leaderboard() -> dict[str, Any]:
    """Вернуть только таблицу лидеров из последнего отчёта.

    Return only the leaderboard from the last comparison report.
    Удобно для Grafana-дашбордов и мониторинга / Useful for Grafana dashboards.
    """
    if _last_comparison_report is None:
        return {"leaderboard": [], "message": "No comparison data yet. POST to /compare/models."}
    return {
        "leaderboard": _last_comparison_report.get("leaderboard", []),
        "timestamp": _last_comparison_report.get("timestamp"),
    }
