"""FastAPI endpoint for anomaly detection with Prometheus metrics and MMD drift detection."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..metrics.prometheus_exporter import AnomalyMetrics, is_available
from ..models.detector import MultiMetricDetector
from ..models.isolation import IsolationConfig, IsolationForestDetector
from ..models.isolation import is_available as isolation_available
from ..models.lstm_autoencoder import LSTMConfig, create_autoencoder
from ..models.mahalanobis import MahalanobisConfig, MahalanobisDetector

app = FastAPI(
    title="Realtime Anomaly Detection API",
    description="Detect anomalies in metric time series with Prometheus observability, "
    "MMD drift detection, LSTM/ESN autoencoder serving, Isolation Forest with "
    "feature-level explainability, and Mahalanobis Distance for correlated metrics",
    version="5.0.0",
)

detector = MultiMetricDetector(window_size=50, threshold_sigma=3.0)

# Инициализируем метрики только если prometheus_client доступен.
# Graceful degradation: без либы API работает в прежнем режиме.
_metrics: AnomalyMetrics | None = None
if is_available():
    _metrics = AnomalyMetrics()
    _metrics.set_model_config(
        threshold_sigma=detector.detector.threshold_sigma,
        window_size=detector.detector.window_size,
    )

# Глобальный drift trigger — инициализируется lazy при первом вызове /drift/check.
# Lazy init нужен чтобы bootstrap не замедлял старт сервиса.
_drift_trigger = None
_last_drift_result: dict | None = None
_DRIFT_FEATURES = ["cpu", "latency", "requests"]

# LSTM/ESN автоэнкодер — инициализируется при /lstm/train.
# Глобальный стейт: один model per service (production pattern).
_lstm_model = create_autoencoder()
_lstm_train_result: dict | None = None

# Isolation Forest — инициализируется при /isolation/train.
_isolation_model = IsolationForestDetector()
_isolation_train_result: dict | None = None

# Mahalanobis Distance — инициализируется при /mahalanobis/train.
_mahalanobis_model = MahalanobisDetector()
_mahalanobis_train_result: dict | None = None


class MetricPoint(BaseModel):
    timestamp: float
    cpu: float = Field(..., ge=0, le=100)
    latency: float = Field(..., ge=0)
    requests: float = Field(..., ge=0)


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float


class DriftCheckRequest(BaseModel):
    """Запрос на проверку дрейфа данных через MMD.

    reference: Эталонные данные — матрица [cpu, latency, requests] за «нормальный» период.
    current: Текущие данные — тот же формат за мониторируемый период.

    Формат: список точек, каждая точка = [cpu, latency, requests].
    """

    reference: list[list[float]] = Field(
        ...,
        description="Reference (training) data: list of [cpu, latency, requests] points",
        min_length=10,
    )
    current: list[list[float]] = Field(
        ...,
        description="Current (production) data: list of [cpu, latency, requests] points",
        min_length=5,
    )
    alpha: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Significance level for MMD bootstrap threshold",
    )
    n_bootstrap: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Bootstrap iterations (more = more accurate threshold)",
    )


class DriftCheckResponse(BaseModel):
    """Ответ MMD drift detection."""

    mmd_statistic: float
    threshold: float
    is_drift: bool
    p_value: float
    gamma: float
    reference_size: int
    current_size: int
    features: list[str]
    audit_id: str
    timestamp: str
    reason: str
    should_retrain: bool
    triggered_by: str


@app.get("/health")
def health() -> dict:
    """Health check + краткая статистика метрик."""
    response: dict = {"status": "healthy", "prometheus": is_available()}
    if _metrics is not None:
        response["stats"] = _metrics.get_summary()
    if _last_drift_result is not None:
        response["last_drift"] = {
            "is_drift": _last_drift_result["is_drift"],
            "mmd_statistic": _last_drift_result["mmd_statistic"],
            "timestamp": _last_drift_result["timestamp"],
        }
    response["isolation_fitted"] = _isolation_model.is_fitted
    response["mahalanobis_fitted"] = _mahalanobis_model.is_fitted
    return response


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """
    Prometheus /metrics endpoint.

    Возвращает метрики в формате text exposition (совместим с Prometheus scrape).
    Если prometheus_client не установлен — возвращает пустую строку.
    """
    if _metrics is None:
        return "# prometheus_client not available\n"
    return _metrics.get_metrics_text()


@app.post("/detect", response_model=list[AnomalyResponse])
def detect_anomalies(points: list[MetricPoint]) -> list[AnomalyResponse]:
    """Detect anomalies in a batch of metric points."""
    data = {
        "cpu": np.array([p.cpu for p in points]),
        "latency": np.array([p.latency for p in points]),
        "requests": np.array([p.requests for p in points]),
    }

    # Измеряем latency детекции и обновляем Prometheus-метрики
    if _metrics is not None:
        with _metrics.track_detection():
            result = detector.detect(data)
    else:
        result = detector.detect(data)

    responses = [
        AnomalyResponse(
            is_anomaly=bool(result.predictions[i]),
            score=float(result.scores[i]),
            threshold=result.threshold,
        )
        for i in range(len(points))
    ]

    # Записываем метрики запроса: кол-во аномалий + их Z-score
    if _metrics is not None:
        anomaly_scores = [
            float(result.scores[i]) for i in range(len(points)) if result.predictions[i]
        ]
        _metrics.record_request(
            n_points=len(points),
            n_anomalies=len(anomaly_scores),
            scores=anomaly_scores,
        )

    return responses


@app.post("/drift/check", response_model=DriftCheckResponse)
def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Проверить дрейф данных между эталонной и текущей выборкой через MMD.

    MMD (Maximum Mean Discrepancy) — многомерный непараметрический тест.
    Анализирует совместное распределение [cpu, latency, requests], ловит
    дрейф в корреляциях между метриками (чего PSI не умеет).

    Bootstrap-порог гарантирует контроль ошибки I рода на уровне alpha.

    Возвращает audit_id для сквозной трассировки (EU AI Act compliance).
    """
    global _drift_trigger, _last_drift_result

    from ..retraining.trigger import AnomalyRetrainingTrigger

    reference = np.array(request.reference)
    current = np.array(request.current)

    # Lazy-инициализация триггера с переданными эталонными данными.
    # Пересоздаём если эталон изменился (разный размер = новый reference период).
    # Это правильно: в production reference обновляется после каждого retraining.
    _drift_trigger = AnomalyRetrainingTrigger(
        reference_data=reference,
        features=_DRIFT_FEATURES,
        alpha=request.alpha,
        n_bootstrap=request.n_bootstrap,
    )

    retrain_result = _drift_trigger.evaluate(current)
    drift = retrain_result.drift_result

    # Сохраняем последний результат для /health endpoint
    _last_drift_result = {
        "is_drift": drift.is_drift,
        "mmd_statistic": drift.mmd_statistic,
        "timestamp": drift.timestamp,
        "audit_id": drift.audit_id,
    }

    return DriftCheckResponse(
        mmd_statistic=drift.mmd_statistic,
        threshold=drift.threshold,
        is_drift=drift.is_drift,
        p_value=drift.p_value,
        gamma=drift.gamma,
        reference_size=drift.reference_size,
        current_size=drift.current_size,
        features=drift.features,
        audit_id=drift.audit_id,
        timestamp=drift.timestamp,
        reason=drift.reason,
        should_retrain=retrain_result.should_retrain,
        triggered_by=retrain_result.triggered_by,
    )


@app.get("/drift/status")
def drift_status() -> dict:
    """Вернуть статус последней проверки дрейфа.

    Полезно для Grafana/Prometheus alerting: опросить статус
    без повторного вычисления MMD.
    """
    if _last_drift_result is None:
        return {
            "status": "no_check_performed",
            "message": "Call POST /drift/check first to initialize drift monitoring",
        }

    return {
        "status": "drift_detected" if _last_drift_result["is_drift"] else "stable",
        **_last_drift_result,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LSTM / ESN Autoencoder endpoints
# ──────────────────────────────────────────────────────────────────────────────


class LSTMTrainRequest(BaseModel):
    """Обучить ESN-автоэнкодер на нормальных данных.

    data: временной ряд нормального поведения — список точек [cpu, latency, requests].
    Рекомендуется минимум 200 точек (~30 мин при 10s-интервале).
    """

    data: list[list[float]] = Field(
        ...,
        description="Normal time series: list of [cpu, latency, requests] points",
        min_length=50,
    )
    reservoir_size: int = Field(default=100, ge=20, le=500)
    window_size: int = Field(default=30, ge=10, le=100)
    anomaly_percentile: float = Field(default=95.0, ge=80.0, le=99.9)


class LSTMTrainResponse(BaseModel):
    n_samples: int
    n_windows: int
    train_mse: float
    threshold: float
    model_config_: dict = Field(alias="model_config")

    model_config = {"populate_by_name": True}


class LSTMDetectRequest(BaseModel):
    """Обнаружить аномалии через ESN-автоэнкодер.

    data: список точек [cpu, latency, requests] для анализа.
    """

    data: list[list[float]] = Field(
        ...,
        description="Time series to analyze: list of [cpu, latency, requests] points",
        min_length=2,
    )


class LSTMDetectResponse(BaseModel):
    scores: list[float]
    predictions: list[int]
    threshold: float
    n_anomalies: int
    anomaly_rate: float
    model: str


@app.post("/lstm/train", response_model=LSTMTrainResponse)
def lstm_train(request: LSTMTrainRequest) -> LSTMTrainResponse:
    """Обучить ESN-автоэнкодер на нормальных данных.

    Echo State Network захватывает сложные временные паттерны (кросс-метрические
    корреляции, burst-recovery циклы), которые Z-score пропускает.

    Результат: порог аномальности (percentile reconstruction error на train data).
    """
    global _lstm_model, _lstm_train_result

    cfg = LSTMConfig(
        reservoir_size=request.reservoir_size,
        window_size=request.window_size,
        anomaly_percentile=request.anomaly_percentile,
    )
    _lstm_model = create_autoencoder(cfg)

    X = np.array(request.data, dtype=float)
    result = _lstm_model.fit(X)

    _lstm_train_result = {
        "n_samples": result.n_samples,
        "n_windows": result.n_windows,
        "train_mse": result.train_mse,
        "threshold": result.threshold,
    }

    return LSTMTrainResponse(
        n_samples=result.n_samples,
        n_windows=result.n_windows,
        train_mse=result.train_mse,
        threshold=result.threshold,
        model_config=_lstm_model.get_config(),
    )


@app.post("/lstm/detect", response_model=LSTMDetectResponse)
def lstm_detect(request: LSTMDetectRequest) -> LSTMDetectResponse:
    """Обнаружить аномалии через ESN-автоэнкодер.

    Требует предварительного обучения: POST /lstm/train.
    Возвращает reconstruction error score для каждой точки + бинарные предсказания.
    """
    if not _lstm_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Call POST /lstm/train with normal data first.",
        )

    X = np.array(request.data, dtype=float)
    result = _lstm_model.detect(X)

    n_anomalies = int(result.predictions.sum())
    anomaly_rate = n_anomalies / len(result.predictions) if len(result.predictions) > 0 else 0.0

    return LSTMDetectResponse(
        scores=[float(s) for s in result.scores],
        predictions=[int(p) for p in result.predictions],
        threshold=result.threshold,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        model=result.model,
    )


@app.get("/lstm/status")
def lstm_status() -> dict:
    """Статус ESN-модели: обучена / не обучена, конфигурация, метрики.

    Полезно для health-check перед inference.
    """
    config = _lstm_model.get_config()
    status = {
        "fitted": _lstm_model.is_fitted,
        "model_config": config,
    }
    if _lstm_train_result is not None:
        status["train_metrics"] = _lstm_train_result
    else:
        status["message"] = "Model not trained. Call POST /lstm/train to initialize."
    return status


# ──────────────────────────────────────────────────────────────────────────────
# Isolation Forest endpoints
# ──────────────────────────────────────────────────────────────────────────────


class IsolationTrainRequest(BaseModel):
    """Обучить Isolation Forest на нормальных данных.

    data: список точек [cpu, latency, requests] — нормальное поведение инфраструктуры.
    Рекомендуется минимум 100 точек для надёжного построения деревьев.
    contamination: ожидаемая доля аномалий в production (0.01–0.20).
    n_estimators: количество деревьев изоляции (больше = стабильнее, медленнее).
    """

    data: list[list[float]] = Field(
        ...,
        description="Normal time series: list of [cpu, latency, requests] points",
        min_length=20,
    )
    contamination: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Expected fraction of anomalies in production data",
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of isolation trees",
    )


class IsolationTrainResponse(BaseModel):
    n_samples: int
    n_features: int
    contamination: float
    n_trees: int
    avg_path_length_normal: float
    sklearn_available: bool


class IsolationPointResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    path_length: float
    feature_contributions: dict[str, float]
    top_feature: str


class IsolationDetectRequest(BaseModel):
    """Обнаружить аномалии через Isolation Forest.

    data: список точек [cpu, latency, requests] для анализа.
    Требует предварительного обучения: POST /isolation/train.
    """

    data: list[list[float]] = Field(
        ...,
        description="Metric points to analyze: list of [cpu, latency, requests]",
        min_length=1,
    )


class IsolationDetectResponse(BaseModel):
    results: list[IsolationPointResult]
    n_anomalies: int
    anomaly_rate: float
    top_anomalous_feature: str | None


@app.post("/isolation/train", response_model=IsolationTrainResponse)
def isolation_train(request: IsolationTrainRequest) -> IsolationTrainResponse:
    """Обучить Isolation Forest на нормальных метриках инфраструктуры.

    Isolation Forest строит ансамбль деревьев случайных разбиений.
    Аномальные точки изолируются быстрее (меньше разбиений = короче путь).
    Преимущество над Z-score: ловит многомерные аномалии (CPU spike + latency
    spike одновременно), где унивариатный Z-score пропускает паттерн.

    Требуется sklearn; graceful degradation: 503 если недоступен.
    """
    global _isolation_model, _isolation_train_result

    if not isolation_available():
        raise HTTPException(
            status_code=503,
            detail="scikit-learn not available; cannot train IsolationForest",
        )

    cfg = IsolationConfig(
        n_estimators=request.n_estimators,
        contamination=request.contamination,
    )
    _isolation_model = IsolationForestDetector(cfg)

    X = np.array(request.data, dtype=float)
    result = _isolation_model.fit(X)

    _isolation_train_result = {
        "n_samples": result.n_samples,
        "n_features": result.n_features,
        "contamination": result.contamination,
        "n_trees": result.n_trees,
        "avg_path_length_normal": result.avg_path_length_normal,
    }

    return IsolationTrainResponse(
        n_samples=result.n_samples,
        n_features=result.n_features,
        contamination=result.contamination,
        n_trees=result.n_trees,
        avg_path_length_normal=result.avg_path_length_normal,
        sklearn_available=True,
    )


@app.post("/isolation/detect", response_model=IsolationDetectResponse)
def isolation_detect(request: IsolationDetectRequest) -> IsolationDetectResponse:
    """Обнаружить аномалии и объяснить вклад каждого признака.

    Возвращает для каждой точки:
    - is_anomaly: флаг аномалии
    - anomaly_score: нормализованная оценка [0, 1]
    - feature_contributions: вклад каждого признака (сумма = 1)
    - top_feature: главная причина аномалии (важно для SRE diagonistics)

    Требует предварительного обучения: POST /isolation/train.
    """
    if not _isolation_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Call POST /isolation/train with normal data first.",
        )

    X = np.array(request.data, dtype=float)
    results = _isolation_model.detect(X)

    point_results = [
        IsolationPointResult(
            is_anomaly=r.is_anomaly,
            anomaly_score=r.anomaly_score,
            path_length=r.path_length,
            feature_contributions=r.feature_contributions,
            top_feature=r.top_feature,
        )
        for r in results
    ]

    n_anomalies = sum(1 for r in results if r.is_anomaly)
    anomaly_rate = n_anomalies / len(results) if results else 0.0

    # Наиболее часто встречающийся top_feature среди аномалий
    anomaly_features = [r.top_feature for r in results if r.is_anomaly]
    top_anomalous_feature: str | None = None
    if anomaly_features:
        top_anomalous_feature = max(set(anomaly_features), key=anomaly_features.count)

    return IsolationDetectResponse(
        results=point_results,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        top_anomalous_feature=top_anomalous_feature,
    )


@app.get("/isolation/status")
def isolation_status() -> dict:
    """Статус Isolation Forest: обучена / не обучена, sklearn доступен.

    Полезно для health-check и мониторинга состояния модели.
    """
    status: dict = {
        "fitted": _isolation_model.is_fitted,
        "sklearn_available": isolation_available(),
    }
    if _isolation_train_result is not None:
        status["train_metrics"] = _isolation_train_result
    else:
        status["message"] = "Model not trained. Call POST /isolation/train to initialize."
    return status


# ──────────────────────────────────────────────────────────────────────────────
# Mahalanobis Distance endpoints
# ──────────────────────────────────────────────────────────────────────────────


class MahalanobisTrainRequest(BaseModel):
    """Обучить Mahalanobis детектор на нормальных метриках инфраструктуры.

    data: список точек [cpu, latency, requests] нормального поведения.
    threshold_percentile: перцентиль дистанций на train данных для порога (97.5 = 2.5% FPR).
    regularization: коэффициент ε регуляризации ковариационной матрицы (ε·I).
    """

    data: list[list[float]] = Field(
        ...,
        description="Normal metric points: list of [cpu, latency, requests]",
        min_length=10,
    )
    threshold_percentile: float = Field(
        default=97.5,
        ge=80.0,
        le=99.9,
        description="Percentile of train distances used as anomaly threshold",
    )
    regularization: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1.0,
        description="Regularisation coefficient added to covariance diagonal",
    )


class MahalanobisTrainResponse(BaseModel):
    n_samples: int
    n_features: int
    mean: list[float]
    condition_number: float
    threshold: float


class MahalanobisPointResult(BaseModel):
    is_anomaly: bool
    mahalanobis_distance: float
    anomaly_score: float
    threshold: float
    feature_contributions: dict[str, float]
    top_feature: str


class MahalanobisDetectRequest(BaseModel):
    """Обнаружить аномалии через Mahalanobis расстояние.

    data: список точек [cpu, latency, requests].
    Требует предварительного обучения: POST /mahalanobis/train.
    """

    data: list[list[float]] = Field(
        ...,
        description="Metric points to analyze: list of [cpu, latency, requests]",
        min_length=1,
    )


class MahalanobisDetectResponse(BaseModel):
    results: list[MahalanobisPointResult]
    n_anomalies: int
    anomaly_rate: float
    top_anomalous_feature: str | None
    mean_distance: float


@app.post("/mahalanobis/train", response_model=MahalanobisTrainResponse)
def mahalanobis_train(request: MahalanobisTrainRequest) -> MahalanobisTrainResponse:
    """Обучить Mahalanobis детектор на нормальных метриках инфраструктуры.

    Mahalanobis расстояние учитывает корреляционную структуру метрик:
    CPU spike + proportional latency spike = нормально (ожидаемая корреляция).
    CPU=10% + latency=500ms = аномально (нарушение нормальной корреляции).

    Преимущества над Z-score: многомерный анализ корреляций.
    Преимущества над Isolation Forest: явная параметрическая модель ковариации.
    Pure numpy — нет внешних зависимостей.
    """
    global _mahalanobis_model, _mahalanobis_train_result

    cfg = MahalanobisConfig(
        threshold_percentile=request.threshold_percentile,
        regularization=request.regularization,
    )
    _mahalanobis_model = MahalanobisDetector(cfg)

    X = np.array(request.data, dtype=float)
    result = _mahalanobis_model.fit(X)

    _mahalanobis_train_result = {
        "n_samples": result.n_samples,
        "n_features": result.n_features,
        "mean": result.mean,
        "condition_number": result.condition_number,
        "threshold": result.threshold,
    }

    return MahalanobisTrainResponse(
        n_samples=result.n_samples,
        n_features=result.n_features,
        mean=result.mean,
        condition_number=result.condition_number,
        threshold=result.threshold,
    )


@app.post("/mahalanobis/detect", response_model=MahalanobisDetectResponse)
def mahalanobis_detect(request: MahalanobisDetectRequest) -> MahalanobisDetectResponse:
    """Обнаружить аномалии и объяснить вклад каждого признака.

    Для каждой точки возвращает:
    - is_anomaly: флаг аномалии
    - mahalanobis_distance: сырое расстояние Махаланобиса
    - anomaly_score: нормализованный score [0, 1]
    - feature_contributions: вклад каждого признака (сумма = 1)
    - top_feature: главная причина аномалии

    Требует предварительного обучения: POST /mahalanobis/train.
    """
    if not _mahalanobis_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Call POST /mahalanobis/train with normal data first.",
        )

    X = np.array(request.data, dtype=float)
    results = _mahalanobis_model.detect(X)

    point_results = [
        MahalanobisPointResult(
            is_anomaly=r.is_anomaly,
            mahalanobis_distance=r.mahalanobis_distance,
            anomaly_score=r.anomaly_score,
            threshold=r.threshold,
            feature_contributions=r.feature_contributions,
            top_feature=r.top_feature,
        )
        for r in results
    ]

    n_anomalies = sum(1 for r in results if r.is_anomaly)
    anomaly_rate = n_anomalies / len(results) if results else 0.0
    mean_distance = float(np.mean([r.mahalanobis_distance for r in results]))

    anomaly_features = [r.top_feature for r in results if r.is_anomaly]
    top_anomalous_feature: str | None = None
    if anomaly_features:
        top_anomalous_feature = max(set(anomaly_features), key=anomaly_features.count)

    return MahalanobisDetectResponse(
        results=point_results,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        top_anomalous_feature=top_anomalous_feature,
        mean_distance=mean_distance,
    )


def _reset_mahalanobis_for_tests() -> None:
    """Сбросить глобальное состояние Mahalanobis детектора для изоляции тестов."""
    global _mahalanobis_model, _mahalanobis_train_result
    _mahalanobis_model = MahalanobisDetector()
    _mahalanobis_train_result = None


@app.get("/mahalanobis/status")
def mahalanobis_status() -> dict:
    """Статус Mahalanobis детектора: обучен / не обучен, порог.

    Полезно для health-check и Grafana мониторинга состояния модели.
    """
    status: dict = {"fitted": _mahalanobis_model.is_fitted}
    if _mahalanobis_train_result is not None:
        status["train_metrics"] = _mahalanobis_train_result
    else:
        status["message"] = "Model not trained. Call POST /mahalanobis/train to initialize."
    return status
