"""
Sklearn-based document classifier -- the lightweight baseline.

This module is the *always-available* path: it runs on any machine with
scikit-learn (no GPU, no torch).  For production quality on raw images
you'd swap this for an EfficientNet-V2 CNN trained in Docker (see
src/models/cnn.py), but for structured features extracted from a
preprocessing pipeline a Random Forest already gets 90%+ accuracy.

Production CNN approach (for reference):
    1. EfficientNet-V2-S pretrained on ImageNet, final FC replaced with
       5-class head.
    2. Fine-tune on 10k labelled document scans (224x224, augmented with
       rotation / perspective / brightness jitter).
    3. Export to ONNX for fast CPU inference behind the FastAPI service.
    4. See Dockerfile and src/models/cnn.py for the training code.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a Random Forest and return model + evaluation metrics.

    Returns a dict with keys: model, accuracy, f1_macro, f1_per_class,
    confusion_matrix, classification_report, label_encoder.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_per = f1_score(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)

    # human-readable report (uses label names if encoder is provided)
    target_names = None
    if label_encoder is not None:
        target_names = list(label_encoder.classes_)
    report = classification_report(y_test, y_pred, target_names=target_names)

    logger.info("Baseline accuracy: %.3f | F1 macro: %.3f", acc, f1_macro)
    logger.info("\n%s", report)

    return {
        "model": clf,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per,
        "confusion_matrix": cm,
        "classification_report": report,
        "label_encoder": label_encoder,
    }


def predict(
    model: RandomForestClassifier,
    features: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> list[dict[str, Any]]:
    """Classify one or more documents given their feature vectors.

    Returns a list of dicts: [{doc_type, confidence, probabilities}, ...].
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    probs = model.predict_proba(features)
    preds = model.predict(features)

    results = []
    for idx in range(len(preds)):
        class_idx = int(preds[idx])
        label = (
            label_encoder.inverse_transform([class_idx])[0]
            if label_encoder is not None
            else str(class_idx)
        )
        results.append(
            {
                "doc_type": label,
                "confidence": float(probs[idx].max()),
                "probabilities": {
                    (
                        label_encoder.inverse_transform([i])[0]
                        if label_encoder is not None
                        else str(i)
                    ): float(p)
                    for i, p in enumerate(probs[idx])
                },
            }
        )
    return results
