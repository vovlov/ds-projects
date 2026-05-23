"""FastAPI endpoint for fraud detection scoring."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data.dataset import generate_synthetic_transactions, get_feature_matrix
from ..models.baseline.tabular import train_baseline
from ..models.calibration import CalibrationResult, FraudCalibrator
from ..models.community import CommunityConfig, DetectionResult, FraudRingDetector
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
    version="2.1.0",
)

_model = None
_temporal_model = None
_trained = False
_temporal_trained = False

# Глобальный экстрактор — единый для всех запросов
_extractor = TemporalFeatureExtractor(TemporalConfig(time_window=30.0))

# Калибратор вероятностей (опциональный)
_calibrator: FraudCalibrator | None = None
_calibration_result: CalibrationResult | None = None

# Детектор мошеннических колец (singleton — не хранит состояние между запросами)
_ring_detector = FraudRingDetector()
_last_detection: DetectionResult | None = None


def _reset_calibrator() -> None:
    """Сбросить глобальный калибратор (для тестовой изоляции)."""
    global _calibrator, _calibration_result
    _calibrator = None
    _calibration_result = None


def _reset_ring_detector() -> None:
    """Сбросить последний результат детекции (для тестовой изоляции)."""
    global _last_detection
    _last_detection = None


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
    calibrated_probability: float | None = None  # None если калибратор не обучен


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


class CalibrateRequest(BaseModel):
    """Параметры обучения калибратора на новых данных."""

    method: str = Field(
        "isotonic",
        description="'platt' (sigmoid, малые датасеты) или 'isotonic' (>1000 samples)",
    )
    n_bins: int = Field(10, ge=2, le=50)


class CalibrateResponse(BaseModel):
    method: str
    n_calibration_samples: int
    ece_before: float
    ece_after: float
    ece_improvement: float
    brier_score: float
    mce: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _trained,
        "temporal_model_loaded": _temporal_trained,
        "calibration_fitted": _calibrator is not None and _calibrator.fitted,
    }


@app.post("/calibrate", response_model=CalibrateResponse)
def calibrate_model(req: CalibrateRequest):
    """Обучить калибратор на синтетических данных (hold-out 20%).

    В production сюда передаются реальные hold-out данные с ground truth.
    Калибровка критична для установки порогов блокировки:
    P(fraud) = 0.7 должна означать 70% реальных мошенников, не 90% и не 50%.
    """
    global _calibrator, _calibration_result

    if req.method not in ("platt", "isotonic"):
        raise HTTPException(status_code=422, detail="method must be 'platt' or 'isotonic'")

    model = _ensure_model()

    # Hold-out калибровочный набор (отдельный от обучающего)
    cal_data = generate_synthetic_transactions(n_nodes=300, n_transactions=1500, fraud_rate=0.10)
    X_cal, y_cal = get_feature_matrix(cal_data)
    raw_scores = model.predict_proba(X_cal)[:, 1]

    _calibrator = FraudCalibrator(method=req.method, n_bins=req.n_bins)  # type: ignore[arg-type]
    result = _calibrator.fit(raw_scores, y_cal)
    _calibration_result = result

    improvement = _calibrator.ece_improvement() or 0.0
    raw_ece = result.ece + improvement  # ece = cal_ece, raw_ece = ece + improvement

    logger.info(
        f"Calibration done: method={req.method}, "
        f"ECE {raw_ece:.4f} → {result.ece:.4f} (Δ={improvement:.4f})"
    )

    return CalibrateResponse(
        method=result.method,
        n_calibration_samples=result.n_calibration_samples,
        ece_before=round(raw_ece, 6),
        ece_after=round(result.ece, 6),
        ece_improvement=round(improvement, 6),
        brier_score=round(result.brier_score, 6),
        mce=round(result.mce, 6),
    )


@app.get("/calibration/metrics")
def calibration_metrics() -> dict[str, Any]:
    """Метрики текущего калибратора + данные reliability diagram (для Grafana/фронтенда)."""
    if _calibration_result is None:
        raise HTTPException(
            status_code=404,
            detail="No calibrator fitted. Call POST /calibrate first.",
        )
    return _calibration_result.to_dict()


@app.post("/score", response_model=FraudScore)
def score_transaction(txn: TransactionInput):
    """Базовая оценка по 3 табличным признакам.

    Если калибратор обучен (POST /calibrate), возвращает также calibrated_probability —
    более надёжную оценку P(fraud) для бизнес-порогов блокировки.
    """
    model = _ensure_model()
    features = np.array([[txn.avg_amount, txn.n_transactions, txn.account_age_days]])
    proba = float(model.predict_proba(features)[0][1])

    cal_proba: float | None = None
    if _calibrator is not None and _calibrator.fitted:
        cal_proba = round(float(_calibrator.calibrate(np.array([proba]))[0]), 4)

    return FraudScore(
        fraud_probability=round(proba, 4),
        is_fraud=proba >= 0.5,
        risk_level=_risk_level(proba),
        calibrated_probability=cal_proba,
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


class CommunityNodeInput(BaseModel):
    node_id: str
    is_fraud: bool | None = Field(None, description="None если метка неизвестна")


class CommunityEdgeInput(BaseModel):
    from_id: str
    to_id: str


class CommunityDetectRequest(BaseModel):
    """Граф транзакций для детекции мошеннических колец.

    Пример: 100 аккаунтов + 300 транзакций → алгоритм находит 5-10 сообществ,
    часть из которых — скоординированные fraud rings.
    """

    nodes: list[CommunityNodeInput] = Field(..., min_length=1)
    edges: list[CommunityEdgeInput] = Field(default_factory=list)
    fraud_ratio_high: float = Field(0.3, ge=0.0, le=1.0, description="Порог для risk_level=high")
    min_ring_size: int = Field(2, ge=1, description="Минимальный размер подозрительного кольца")


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


@app.post("/community/detect")
def detect_fraud_rings(req: CommunityDetectRequest) -> dict:
    """Обнаружить мошеннические кольца через Label Propagation.

    Алгоритм находит плотно связанные сообщества аккаунтов. Сообщества с высокой долей
    known fraud → потенциальные fraud rings (организованное мошенничество).

    Fraud rings (>30% known fraudsters) автоматически помечаются risk_level='high'
    и возвращаются в suspicious_rings для приоритетного расследования.
    """
    global _last_detection

    config = CommunityConfig(
        fraud_ratio_high=req.fraud_ratio_high,
        min_ring_size=req.min_ring_size,
    )
    detector = FraudRingDetector(config)

    node_ids = [n.node_id for n in req.nodes]
    edges = [(e.from_id, e.to_id) for e in req.edges]
    fraud_labels = {n.node_id: n.is_fraud for n in req.nodes if n.is_fraud is not None}

    result = detector.detect(node_ids, edges, fraud_labels)
    _last_detection = result

    logger.info(
        f"Community detection: {result.n_communities} communities, "
        f"{len(result.suspicious_rings)} suspicious rings, "
        f"converged={result.converged} in {result.n_iterations} iter"
    )

    return result.to_dict()


@app.get("/community/stats")
def community_stats() -> dict:
    """Статистика последней детекции колец (для Grafana / дашбордов).

    Возвращает агрегированные метрики без полного списка узлов —
    подходит для мониторинга без утечки PII.
    """
    if _last_detection is None:
        raise HTTPException(
            status_code=404,
            detail="No community detection run yet. Call POST /community/detect first.",
        )

    d = _last_detection
    nodes_in_suspicious = sum(c.size for c in d.suspicious_rings)

    return {
        "n_communities": d.n_communities,
        "n_suspicious_rings": len(d.suspicious_rings),
        "total_nodes_analyzed": d.total_nodes,
        "nodes_in_suspicious_rings": nodes_in_suspicious,
        "suspicious_coverage_ratio": (
            round(nodes_in_suspicious / d.total_nodes, 4) if d.total_nodes > 0 else 0.0
        ),
        "converged": d.converged,
        "n_iterations": d.n_iterations,
    }
