"""
FastAPI service for document classification.

The model is trained lazily on first request using synthetic data and the
sklearn baseline.  In production you'd load a pre-trained ONNX model
instead (see src/models/cnn.py for the export function).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.dataset import generate_synthetic_documents, get_feature_matrix
from src.models.classifier import predict, train_classifier

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Scanner API",
    description="Classify scanned insurance documents by type.",
    version="0.1.0",
)

# module-level state -- populated on first classify request
_state: dict[str, Any] = {}


class DocumentFeatures(BaseModel):
    """Feature vector extracted from a scanned document image."""

    aspect_ratio: float = Field(..., ge=0.1, le=3.0)
    brightness: float = Field(..., ge=0.0, le=1.0)
    text_density: float = Field(..., ge=0.0, le=1.0)
    edge_density: float = Field(..., ge=0.0, le=1.0)
    file_size_kb: float = Field(..., ge=1.0, le=5000.0)


class PredictionResponse(BaseModel):
    doc_type: str
    confidence: float
    probabilities: dict[str, float]


def _ensure_model() -> None:
    """Train the sklearn model if it hasn't been loaded yet."""
    if "model" in _state:
        return
    logger.info("First request -- training baseline model on synthetic data...")
    data = generate_synthetic_documents(n=500)
    X, y, le = get_feature_matrix(data)
    result = train_classifier(X, y, label_encoder=le)
    _state["model"] = result["model"]
    _state["label_encoder"] = result["label_encoder"]
    logger.info("Model ready (accuracy=%.3f)", result["accuracy"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=PredictionResponse)
def classify(features: DocumentFeatures) -> PredictionResponse:
    """Classify a single document based on its extracted features."""
    _ensure_model()

    feature_array = np.array(
        [
            [
                features.aspect_ratio,
                features.brightness,
                features.text_density,
                features.edge_density,
                features.file_size_kb,
            ]
        ],
        dtype=np.float32,
    )

    results = predict(
        _state["model"],
        feature_array,
        label_encoder=_state["label_encoder"],
    )
    if not results:
        raise HTTPException(status_code=500, detail="Prediction failed")

    r = results[0]
    return PredictionResponse(
        doc_type=r["doc_type"],
        confidence=r["confidence"],
        probabilities=r["probabilities"],
    )
