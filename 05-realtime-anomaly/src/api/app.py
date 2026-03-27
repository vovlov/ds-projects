"""FastAPI endpoint for anomaly detection."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..models.detector import MultiMetricDetector

app = FastAPI(
    title="Realtime Anomaly Detection API",
    description="Detect anomalies in metric time series",
    version="1.0.0",
)

detector = MultiMetricDetector(window_size=50, threshold_sigma=3.0)


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
def health():
    return {"status": "healthy"}


@app.post("/detect", response_model=list[AnomalyResponse])
def detect_anomalies(points: list[MetricPoint]):
    """Detect anomalies in a batch of metric points."""
    import numpy as np

    data = {
        "cpu": np.array([p.cpu for p in points]),
        "latency": np.array([p.latency for p in points]),
        "requests": np.array([p.requests for p in points]),
    }

    result = detector.detect(data)

    return [
        AnomalyResponse(
            is_anomaly=bool(result.predictions[i]),
            score=float(result.scores[i]),
            threshold=result.threshold,
        )
        for i in range(len(points))
    ]
