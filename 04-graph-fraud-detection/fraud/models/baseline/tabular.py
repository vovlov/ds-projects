"""CatBoost tabular baseline for fraud detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train_baseline(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Train CatBoost baseline on tabular features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model": model,
        "f1_score": f1,
        "roc_auc": auc,
        "report": report,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
