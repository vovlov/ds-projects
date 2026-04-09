"""FastAPI endpoint for anomaly detection with Prometheus metrics."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..metrics.prometheus_exporter import AnomalyMetrics, is_available
from ..models.detector import MultiMetricDetector

app = FastAPI(
    title="Realtime Anomaly Detection API",
    description="Detect anomalies in metric time series with Prometheus observability",
    version="2.0.0",
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


class MetricPoint(BaseModel):
    timestamp: float
    cpu: float = Field(..., ge=0, le=100)
    latency: float = Field(..., ge=0)
    requests: float = Field(..., ge=0)


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float


@app.get("/health")
def health() -> dict:
    """Health check + краткая статистика метрик."""
    response: dict = {"status": "healthy", "prometheus": is_available()}
    if _metrics is not None:
        response["stats"] = _metrics.get_summary()
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
    import numpy as np

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
