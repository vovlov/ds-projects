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
from ..models.centrality import (
    CENTRALITY_FEATURE_NAMES,
    CentralityConfig,
    CentralityExtractResult,
    CentralityFeatureExtractor,
    explain_centrality_features,
)
from ..models.community import CommunityConfig, DetectionResult, FraudRingDetector
from ..models.lime import LIMEConfig, LIMEExplainer
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

# Centrality extractor (stateless, создаётся один раз)
_centrality_extractor = CentralityFeatureExtractor(CentralityConfig())
_last_centrality: CentralityExtractResult | None = None

# LIME-объяснитель (создаётся лениво при первом вызове /explain/lime)
_lime_explainer: LIMEExplainer | None = None


def _reset_calibrator() -> None:
    """Сбросить глобальный калибратор (для тестовой изоляции)."""
    global _calibrator, _calibration_result
    _calibrator = None
    _calibration_result = None


def _reset_ring_detector() -> None:
    """Сбросить последний результат детекции (для тестовой изоляции)."""
    global _last_detection
    _last_detection = None


def _reset_centrality() -> None:
    """Сбросить последний centrality-результат (для тестовой изоляции)."""
    global _last_centrality
    _last_centrality = None


def _reset_lime_explainer() -> None:
    """Сбросить LIME-объяснитель (для тестовой изоляции)."""
    global _lime_explainer
    _lime_explainer = None


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
        "centrality_last_run": _last_centrality is not None,
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


# ─── Centrality endpoints ───────────────────────────────────────────────────


class CentralityNodeInput(BaseModel):
    node_id: str
    is_fraud: bool | None = Field(None, description="Известная метка (для контекста)")


class CentralityEdgeInput(BaseModel):
    from_id: str
    to_id: str


class CentralityRequest(BaseModel):
    """Граф транзакций для вычисления centrality-признаков.

    Типичное применение: вычислить признаки один раз (offline feature store),
    затем подавать предвычисленные значения в POST /score/graph.
    """

    nodes: list[CentralityNodeInput] = Field(..., min_length=1)
    edges: list[CentralityEdgeInput] = Field(default_factory=list)
    top_k: int = Field(5, ge=1, le=100, description="Сколько top-risk узлов вернуть")


class NodeCentralityResponse(BaseModel):
    node_id: str
    pagerank: float
    in_degree_centrality: float
    out_degree_centrality: float
    betweenness_approx: float
    clustering_coefficient: float
    k_core_number: float
    risk_flags: dict[str, str]
    is_fraud: bool | None = None


class CentralityResponse(BaseModel):
    n_nodes: int
    n_edges: int
    max_pagerank: float
    max_k_core_raw: int
    nodes: list[NodeCentralityResponse]


@app.post("/centrality/compute", response_model=CentralityResponse)
def compute_centrality(req: CentralityRequest) -> CentralityResponse:
    """Вычислить graph centrality признаки для всех узлов.

    Признаки (6 штук):
    - pagerank: influence-score (Brin & Page 1998) — деньги «текут» к крупным хабам.
    - in/out_degree_centrality: нормированный вход/выход [0,1].
    - betweenness_approx: роль посредника/координатора (k-BFS аппроксимация).
    - clustering_coefficient: плотность треугольников в окрестности.
    - k_core_number: глубина встроенности в core сети (Malliaros et al. 2020).

    Возвращает top_k узлов по PageRank + risk_flags для каждого.
    В production признаки записываются в feature store и подаются в POST /score/graph.
    """
    global _last_centrality

    node_ids = [n.node_id for n in req.nodes]
    fraud_map = {n.node_id: n.is_fraud for n in req.nodes}
    edges = [(e.from_id, e.to_id) for e in req.edges]

    result = _centrality_extractor.extract(node_ids, edges)
    _last_centrality = result

    logger.info(
        f"Centrality computed: {result.n_nodes} nodes, {result.n_edges} edges, "
        f"max_PR={result.max_pagerank:.4f}, max_k_core={result.max_k_core_raw}"
    )

    # Порог PageRank — top 10% считаем «высоким» (не абсолютный, а относительный)
    all_pr = [result.features[nid].pagerank for nid in node_ids]
    pr_threshold = float(np.percentile(all_pr, 90)) if all_pr else 0.01
    pr_threshold = max(pr_threshold, 1.0 / max(result.n_nodes, 1))  # минимум 1/N

    # Сортировать по PageRank убыванию, взять top_k
    sorted_nodes = sorted(node_ids, key=lambda nid: result.features[nid].pagerank, reverse=True)
    top_nodes = sorted_nodes[: req.top_k]

    node_responses = [
        NodeCentralityResponse(
            node_id=nid,
            pagerank=round(result.features[nid].pagerank, 6),
            in_degree_centrality=round(result.features[nid].in_degree_centrality, 4),
            out_degree_centrality=round(result.features[nid].out_degree_centrality, 4),
            betweenness_approx=round(result.features[nid].betweenness_approx, 4),
            clustering_coefficient=round(result.features[nid].clustering_coefficient, 4),
            k_core_number=round(result.features[nid].k_core_number, 4),
            risk_flags=explain_centrality_features(
                result.features[nid],
                pr_threshold=pr_threshold,
            ),
            is_fraud=fraud_map.get(nid),
        )
        for nid in top_nodes
    ]

    return CentralityResponse(
        n_nodes=result.n_nodes,
        n_edges=result.n_edges,
        max_pagerank=round(result.max_pagerank, 6),
        max_k_core_raw=result.max_k_core_raw,
        nodes=node_responses,
    )


@app.get("/centrality/info")
def centrality_info() -> dict:
    """Описание centrality-признаков и их применение к fraud detection.

    Используется для документации, onboarding аналитиков и аудита (EU AI Act).
    """
    return {
        "feature_names": CENTRALITY_FEATURE_NAMES,
        "algorithms": {
            "pagerank": "Power iteration (Brin & Page 1998), d=0.85, converges in ~50 iter",
            "degree_centrality": "Normalized in/out degree: k / (N-1)",
            "betweenness_approx": "k-BFS approximation (Brandes 2001), O(k·(V+E))",
            "clustering_coefficient": "Local triangles: 2·t(v) / (k·(k-1))",
            "k_core_number": "Iterative peeling, normalized by max k-core",
        },
        "fraud_patterns": {
            "high_pagerank": "Money mule aggregators receive from many sources → high PR",
            "high_in_degree": "Accounts receiving many transfers (potential layering hub)",
            "high_betweenness": "Coordinators routing funds between sub-rings",
            "high_clustering": "Tight-knit fraud ring members (many shared neighbors)",
            "high_k_core": "Deeply embedded nodes — removal-resistant fraud infrastructure",
        },
        "compliance": {
            "EU_AI_Act_Article_13": "explain_centrality_features() provides human-readable flags",
            "feature_count": len(CENTRALITY_FEATURE_NAMES),
            "external_dependencies": "none (numpy + stdlib only)",
        },
    }


# ─── LIME Explainability endpoints ──────────────────────────────────────────

_BASE_FEATURE_NAMES = ["avg_amount", "n_transactions", "account_age_days"]


def _get_lime_explainer(n_perturbations: int = 500) -> LIMEExplainer:
    """Получить или создать LIME-объяснитель для базовых (3) признаков."""
    global _lime_explainer
    if _lime_explainer is None:
        _lime_explainer = LIMEExplainer(
            feature_names=_BASE_FEATURE_NAMES,
            config=LIMEConfig(n_perturbations=n_perturbations, seed=42),
        )
    return _lime_explainer


class LIMERequest(BaseModel):
    """Транзакция для LIME-объяснения предсказания мошенничества."""

    avg_amount: float = Field(..., ge=0, examples=[500.0])
    n_transactions: int = Field(..., ge=0, examples=[15])
    account_age_days: float = Field(..., ge=0, examples=[10.0])
    n_perturbations: int = Field(
        500, ge=50, le=2000, description="Число точек для аппроксимации (больше = точнее)"
    )


class LIMEFeatureResponse(BaseModel):
    feature_name: str
    value: float
    contribution: float
    direction: str


class LIMEResponse(BaseModel):
    """Объяснение одного предсказания через LIME."""

    prediction: float
    local_prediction: float
    local_fidelity: float
    intercept: float
    top_features: list[LIMEFeatureResponse]
    n_perturbations: int
    method: str


@app.post("/explain/lime", response_model=LIMEResponse)
def explain_lime(req: LIMERequest) -> LIMEResponse:
    """Объяснить предсказание мошенничества через LIME.

    Строит локальную линейную аппроксимацию чёрного ящика в окрестности
    данной транзакции. Возвращает top признаки по абсолютному вкладу:
    - contribution > 0: признак увеличивает P(fraud)
    - contribution < 0: признак снижает P(fraud)
    - local_fidelity ≈ 1.0: локальная модель точно описывает оригинальную.

    Соответствие EU AI Act Article 13 (право на объяснение автоматических решений).
    """
    model = _ensure_model()

    def predict_fn(x: np.ndarray) -> np.ndarray:
        return model.predict_proba(x)

    explainer = _get_lime_explainer(n_perturbations=req.n_perturbations)
    instance = np.array([req.avg_amount, req.n_transactions, req.account_age_days], dtype=float)
    explanation = explainer.explain(instance, predict_fn=predict_fn)

    logger.info(
        f"LIME: P(fraud)={explanation.prediction:.4f}, "
        f"fidelity={explanation.local_fidelity:.4f}, "
        f"top_feature={explanation.top_features[0].feature_name if explanation.top_features else 'n/a'}"  # noqa: E501
    )

    return LIMEResponse(
        prediction=explanation.prediction,
        local_prediction=explanation.local_prediction,
        local_fidelity=explanation.local_fidelity,
        intercept=explanation.intercept,
        top_features=[
            LIMEFeatureResponse(
                feature_name=f.feature_name,
                value=f.value,
                contribution=f.contribution,
                direction=f.direction,
            )
            for f in explanation.top_features
        ],
        n_perturbations=explanation.n_perturbations,
        method=explanation.method,
    )


@app.get("/explain/info")
def explain_info() -> dict:
    """Обзор методов объяснимости, реализованных в проекте.

    Используется для документации, onboarding аналитиков и аудита (EU AI Act Article 13).
    """
    return {
        "methods": {
            "lime": {
                "endpoint": "POST /explain/lime",
                "type": "local, model-agnostic",
                "algorithm": "Weighted Ridge regression on Gaussian perturbations",
                "output": "feature contributions (signed) + local_fidelity",
                "complexity": "O(n_perturbations × n_features)",
                "reference": "Ribeiro et al. 2016 KDD (arxiv:1602.04938)",
            },
            "temporal_flags": {
                "endpoint": "POST /score/graph",
                "type": "local, rule-based",
                "algorithm": "Threshold rules on temporal graph features",
                "output": "human-readable risk flags per temporal feature",
                "reference": "EU AI Act Article 13",
            },
            "centrality_flags": {
                "endpoint": "POST /centrality/compute",
                "type": "local, rule-based",
                "algorithm": "Percentile thresholds on graph centrality features",
                "output": "human-readable risk flags per structural feature",
                "reference": "Malliaros et al. 2020 IEEE TKDE",
            },
        },
        "compliance": {
            "EU_AI_Act_Article_13": (
                "All three methods provide human-readable explanations "
                "for individual predictions as required by Article 13."
            ),
            "external_dependencies": "none (numpy + stdlib only for LIME)",
        },
        "lime_config": {
            "n_perturbations_default": 500,
            "kernel_width": 0.75,
            "n_features_in_explanation": len(_BASE_FEATURE_NAMES),
            "ridge_alpha": 1.0,
        },
    }
