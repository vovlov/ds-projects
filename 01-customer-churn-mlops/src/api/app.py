"""FastAPI prediction service for customer churn model."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data.load import CATEGORICAL_FEATURES, add_features

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


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


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
