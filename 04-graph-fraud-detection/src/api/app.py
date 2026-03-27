"""FastAPI endpoint for fraud detection scoring."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..data.dataset import generate_synthetic_transactions, get_feature_matrix
from ..models.baseline.tabular import train_baseline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph Fraud Detection API",
    description="Score transactions for fraud using CatBoost baseline",
    version="1.0.0",
)

_model = None
_trained = False


def _ensure_model():
    global _model, _trained
    if not _trained:
        logger.info("Training baseline model on synthetic data...")
        data = generate_synthetic_transactions(n_nodes=500, n_transactions=2000, fraud_rate=0.08)
        X, y = get_feature_matrix(data)
        result = train_baseline(X, y)
        _model = result["model"]
        _trained = True
        logger.info(f"Model trained: F1={result['f1_score']:.4f}, AUC={result['roc_auc']:.4f}")
    return _model


class TransactionInput(BaseModel):
    avg_amount: float = Field(..., ge=0, examples=[150.0])
    n_transactions: int = Field(..., ge=0, examples=[5])
    account_age_days: float = Field(..., ge=0, examples=[180.0])


class FraudScore(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _trained}


@app.post("/score", response_model=FraudScore)
def score_transaction(txn: TransactionInput):
    model = _ensure_model()
    features = np.array([[txn.avg_amount, txn.n_transactions, txn.account_age_days]])
    proba = float(model.predict_proba(features)[0][1])

    if proba >= 0.7:
        risk = "high"
    elif proba >= 0.3:
        risk = "medium"
    else:
        risk = "low"

    return FraudScore(
        fraud_probability=round(proba, 4),
        is_fraud=proba >= 0.5,
        risk_level=risk,
    )


@app.post("/score/batch")
def score_batch(transactions: list[TransactionInput]):
    model = _ensure_model()
    features = np.array(
        [[t.avg_amount, t.n_transactions, t.account_age_days] for t in transactions]
    )
    probas = model.predict_proba(features)[:, 1]

    return [
        {
            "fraud_probability": round(float(p), 4),
            "is_fraud": float(p) >= 0.5,
            "risk_level": "high" if p >= 0.7 else "medium" if p >= 0.3 else "low",
        }
        for p in probas
    ]
