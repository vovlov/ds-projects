"""FastAPI prediction service for customer churn model."""

from __future__ import annotations

import pickle
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

    total_charges = customer.TotalCharges
    tenure = customer.tenure
    monthly = customer.MonthlyCharges

    features = [
        customer.gender,
        customer.Partner,
        customer.Dependents,
        customer.PhoneService,
        customer.MultipleLines,
        customer.InternetService,
        customer.OnlineSecurity,
        customer.OnlineBackup,
        customer.DeviceProtection,
        customer.TechSupport,
        customer.StreamingTV,
        customer.StreamingMovies,
        customer.Contract,
        customer.PaperlessBilling,
        customer.PaymentMethod,
        customer.SeniorCitizen,
        tenure,
        monthly,
        total_charges,
        total_charges / (tenure + 1),  # AvgMonthlySpend
        monthly * tenure,  # ExpectedTotalCharges
        "new" if tenure <= 12 else "mid" if tenure <= 36 else "long",  # TenureGroup
        sum(
            [
                customer.OnlineSecurity == "Yes",
                customer.OnlineBackup == "Yes",
                customer.DeviceProtection == "Yes",
                customer.TechSupport == "Yes",
                customer.StreamingTV == "Yes",
                customer.StreamingMovies == "Yes",
            ]
        ),  # NumServices
    ]

    proba = float(model.predict_proba([features])[0][1])
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
