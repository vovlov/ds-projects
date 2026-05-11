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

from scanner.data.dataset import generate_synthetic_documents, get_feature_matrix
from scanner.models.classifier import predict, train_classifier
from scanner.preprocessing.quality import QualityMetrics, assess_quality

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Scanner API",
    description="Classify scanned insurance documents by type.",
    version="0.2.0",
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


class PixelMatrix(BaseModel):
    """Grayscale pixel matrix sent as a 2-D list of integers [0-255]."""

    pixels: list[list[int]] = Field(
        ...,
        description="2-D array of uint8 grayscale pixels (rows × cols)",
    )


class QualityAssessmentResponse(BaseModel):
    blur_score: float
    brightness_score: float
    contrast_score: float
    noise_level: float
    skew_angle_deg: float
    overall_score: float
    is_acceptable: bool
    rejection_reason: str | None = None


class ClassifyWithGateRequest(BaseModel):
    """Two-stage: quality check → classification (if quality passes)."""

    quality_pixels: list[list[int]] = Field(
        ..., description="Grayscale pixel matrix for quality gate"
    )
    features: DocumentFeatures


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


def _pixels_to_array(pixels: list[list[int]]) -> np.ndarray:
    """Convert nested list to uint8 ndarray with basic shape validation."""
    if not pixels or not pixels[0]:
        raise HTTPException(status_code=422, detail="Pixel matrix must not be empty")
    row_len = len(pixels[0])
    if any(len(r) != row_len for r in pixels):
        raise HTTPException(status_code=422, detail="Pixel matrix rows must have equal length")
    return np.array(pixels, dtype=np.uint8)


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


@app.post("/quality/assess", response_model=QualityAssessmentResponse)
def assess_document_quality(request: PixelMatrix) -> QualityAssessmentResponse:
    """Assess scan quality and decide whether it is usable for classification.

    Rejects low-quality scans early to prevent silent model failures —
    critical for STP (straight-through processing) in insurance pipelines.
    """
    arr = _pixels_to_array(request.pixels)
    metrics: QualityMetrics = assess_quality(arr)
    return QualityAssessmentResponse(**metrics.to_dict())


@app.post("/classify/gated")
def classify_with_quality_gate(request: ClassifyWithGateRequest) -> dict[str, Any]:
    """Two-stage pipeline: quality gate → classification.

    Returns quality metrics plus classification result (or None if rejected).
    Field ``gated=True`` signals that classification was skipped.
    """
    arr = _pixels_to_array(request.quality_pixels)
    metrics: QualityMetrics = assess_quality(arr)

    result: dict[str, Any] = {"quality": metrics.to_dict(), "gated": not metrics.is_acceptable}

    if not metrics.is_acceptable:
        result["classification"] = None
        return result

    _ensure_model()
    feature_array = np.array(
        [
            [
                request.features.aspect_ratio,
                request.features.brightness,
                request.features.text_density,
                request.features.edge_density,
                request.features.file_size_kb,
            ]
        ],
        dtype=np.float32,
    )
    predictions = predict(
        _state["model"],
        feature_array,
        label_encoder=_state["label_encoder"],
    )
    if not predictions:
        raise HTTPException(status_code=500, detail="Prediction failed")

    result["classification"] = predictions[0]
    return result
