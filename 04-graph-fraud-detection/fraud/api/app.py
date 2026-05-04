"""FastAPI endpoint for fraud detection scoring."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..data.dataset import generate_synthetic_transactions, get_feature_matrix
from ..models.baseline.tabular import train_baseline
from ..models.temporal import (
    TEMPORAL_FEATURE_NAMES,
    NodeTemporalFeatures,
    TemporalConfig,
    TemporalFeatureExtractor,
    explain_temporal_features,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph Fraud Detection API",
    description="Score transactions for fraud using CatBoost baseline + temporal graph features",
    version="2.0.0",
)

_model = None
_temporal_model = None
_trained = False
_temporal_trained = False

# Глобальный экстрактор — единый для всех запросов
_extractor = TemporalFeatureExtractor(TemporalConfig(time_window=30.0))


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


def _ensure_temporal_model():
    """Обучить модель на base + temporal признаках.

    Temporal-признаки добавляют ~3-8% AUC на реальных датасетах (Elliptic benchmark).
    """
    global _temporal_model, _temporal_trained
    if not _temporal_trained:
        logger.info("Training temporal model (base + 6 temporal graph features)...")
        data = generate_synthetic_transactions(n_nodes=500, n_transactions=2000, fraud_rate=0.08)
        X, y = get_feature_matrix(data)
        X_aug = _extractor.augment_features(X, data)
        result = train_baseline(X_aug, y)
        _temporal_model = result["model"]
        _temporal_trained = True
        logger.info(
            f"Temporal model trained: F1={result['f1_score']:.4f}, AUC={result['roc_auc']:.4f}"
        )
    return _temporal_model


class TransactionInput(BaseModel):
    avg_amount: float = Field(..., ge=0, examples=[150.0])
    n_transactions: int = Field(..., ge=0, examples=[5])
    account_age_days: float = Field(..., ge=0, examples=[180.0])


class GraphTransactionInput(BaseModel):
    """Входные данные для temporal-обогащённой оценки мошенничества.

    Temporal-признаки клиент вычисляет самостоятельно (feature store),
    что соответствует production-паттерну: offline store → online serving.
    """

    avg_amount: float = Field(..., ge=0, examples=[150.0])
    n_transactions: int = Field(..., ge=0, examples=[5])
    account_age_days: float = Field(..., ge=0, examples=[180.0])
    # Temporal graph features (предвычислены feature store)
    velocity_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Доля транзакций в окне 30д")
    burst_score: float = Field(0.0, ge=0.0, description="CV временны́х интервалов (нерегулярность)")
    amount_hhi: float = Field(0.0, ge=0.0, le=1.0, description="Herfindahl Index сумм")
    recent_amount_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Доля объёма в окне 30д")
    neighbor_fraud_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Доля соседей-мошенников")
    hub_proximity: float = Field(0.0, ge=0.0, description="log(1 + средняя степень соседей)")


class FraudScore(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class GraphFraudScore(BaseModel):
    """Результат temporal-обогащённой оценки с объяснениями."""

    fraud_probability: float
    is_fraud: bool
    risk_level: str
    temporal_flags: dict[str, str]  # объяснения подозрительных temporal-признаков
    feature_contributions: dict[str, float]  # значения temporal-признаков


def _risk_level(proba: float) -> str:
    if proba >= 0.7:
        return "high"
    elif proba >= 0.3:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _trained,
        "temporal_model_loaded": _temporal_trained,
    }


@app.post("/score", response_model=FraudScore)
def score_transaction(txn: TransactionInput):
    """Базовая оценка по 3 табличным признакам."""
    model = _ensure_model()
    features = np.array([[txn.avg_amount, txn.n_transactions, txn.account_age_days]])
    proba = float(model.predict_proba(features)[0][1])
    return FraudScore(
        fraud_probability=round(proba, 4),
        is_fraud=proba >= 0.5,
        risk_level=_risk_level(proba),
    )


@app.post("/score/batch")
def score_batch(transactions: list[TransactionInput]):
    """Пакетная базовая оценка транзакций."""
    model = _ensure_model()
    features = np.array(
        [[t.avg_amount, t.n_transactions, t.account_age_days] for t in transactions]
    )
    probas = model.predict_proba(features)[:, 1]

    return [
        {
            "fraud_probability": round(float(p), 4),
            "is_fraud": float(p) >= 0.5,
            "risk_level": _risk_level(float(p)),
        }
        for p in probas
    ]


@app.post("/score/graph", response_model=GraphFraudScore)
def score_with_temporal(txn: GraphTransactionInput):
    """Temporal-обогащённая оценка: base (3) + temporal graph features (6).

    Temporal-признаки улавливают паттерны, невидимые в статических признаках:
    burst activity, концентрацию сумм, сетевое окружение мошенников.

    В production эти признаки предвычисляются feature store (Feast/Tecton)
    и подаются готовыми — latency не зависит от размера графа.
    """
    model = _ensure_temporal_model()

    temporal_feat = NodeTemporalFeatures(
        velocity_ratio=txn.velocity_ratio,
        burst_score=txn.burst_score,
        amount_hhi=txn.amount_hhi,
        recent_amount_ratio=txn.recent_amount_ratio,
        neighbor_fraud_ratio=txn.neighbor_fraud_ratio,
        hub_proximity=txn.hub_proximity,
    )

    base_features = np.array([[txn.avg_amount, txn.n_transactions, txn.account_age_days]])
    temporal_vec = temporal_feat.to_array().reshape(1, -1)
    full_features = np.hstack([base_features, temporal_vec])

    proba = float(model.predict_proba(full_features)[0][1])

    explanations = explain_temporal_features(temporal_feat)
    contributions = {
        name: float(val)
        for name, val in zip(TEMPORAL_FEATURE_NAMES, temporal_feat.to_array(), strict=True)
    }

    return GraphFraudScore(
        fraud_probability=round(proba, 4),
        is_fraud=proba >= 0.5,
        risk_level=_risk_level(proba),
        temporal_flags=explanations,
        feature_contributions=contributions,
    )
