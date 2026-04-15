"""FastAPI endpoint for anomaly detection with Prometheus metrics and MMD drift detection."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..metrics.prometheus_exporter import AnomalyMetrics, is_available
from ..models.detector import MultiMetricDetector

app = FastAPI(
    title="Realtime Anomaly Detection API",
    description="Detect anomalies in metric time series with Prometheus observability "
    "and MMD drift detection",
    version="3.0.0",
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
