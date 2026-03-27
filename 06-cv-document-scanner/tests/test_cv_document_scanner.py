"""
Tests for the document scanner project.

Covers data generation, classifier training, API endpoints, and the CNN
availability check.  Everything runs without PyTorch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.data.dataset import (
    DOC_TYPES,
    FEATURE_COLS,
    generate_synthetic_documents,
    get_feature_matrix,
)
from src.models.classifier import predict, train_classifier
from src.models.cnn import is_available as cnn_is_available

# ---- fixtures ----


@pytest.fixture(scope="module")
def synth_data():
    return generate_synthetic_documents(n=500)


@pytest.fixture(scope="module")
def feature_matrix(synth_data):
    return get_feature_matrix(synth_data)


@pytest.fixture(scope="module")
def trained(feature_matrix):
    X, y, le = feature_matrix
    return train_classifier(X, y, label_encoder=le)


@pytest.fixture(scope="module")
def api_client():
    # import here so module-level state doesn't leak across fixtures
    from src.api.app import app

    return TestClient(app)


# ===========================================================================
# TestData
# ===========================================================================


class TestData:
    def test_row_count(self, synth_data):
        # 500 samples, evenly divided among 5 classes -> 500
        assert synth_data.shape[0] == 500

    def test_columns_present(self, synth_data):
        for col in FEATURE_COLS + ["doc_type"]:
            assert col in synth_data.columns

    def test_doc_types(self, synth_data):
        unique = set(synth_data["doc_type"].to_list())
        assert unique == set(DOC_TYPES)

    def test_class_balance(self, synth_data):
        counts = synth_data.group_by("doc_type").len()["len"].to_list()
        assert all(c == 100 for c in counts), "Expected 100 samples per class"

    def test_feature_ranges(self, synth_data):
        # aspect ratio should be positive
        assert synth_data["aspect_ratio"].min() > 0
        # brightness and densities in [0, 1]
        for col in ("brightness", "text_density", "edge_density"):
            assert synth_data[col].min() >= 0.0
            assert synth_data[col].max() <= 1.0

    def test_feature_matrix_shape(self, feature_matrix):
        X, y, le = feature_matrix
        assert X.shape == (500, 5)
        assert y.shape == (500,)

    def test_label_encoder_classes(self, feature_matrix):
        _, _, le = feature_matrix
        assert set(le.classes_) == set(DOC_TYPES)


# ===========================================================================
# TestClassifier
# ===========================================================================


class TestClassifier:
    def test_train_runs(self, trained):
        assert "model" in trained
        assert "accuracy" in trained

    def test_accuracy_above_threshold(self, trained):
        # synthetic data is designed to be separable -- should exceed 50% easily
        assert trained["accuracy"] > 0.5

    def test_f1_macro_reasonable(self, trained):
        assert trained["f1_macro"] > 0.5

    def test_confusion_matrix_shape(self, trained):
        cm = trained["confusion_matrix"]
        assert cm.shape == (5, 5)

    def test_predict_single(self, trained):
        model = trained["model"]
        le = trained["label_encoder"]
        # a receipt-like feature vector
        sample = np.array([[0.35, 0.78, 0.45, 0.30, 120.0]], dtype=np.float32)
        results = predict(model, sample, label_encoder=le)
        assert len(results) == 1
        assert results[0]["doc_type"] in DOC_TYPES
        assert 0.0 <= results[0]["confidence"] <= 1.0

    def test_predict_batch(self, trained):
        model = trained["model"]
        le = trained["label_encoder"]
        batch = np.random.default_rng(0).random((10, 5)).astype(np.float32)
        results = predict(model, batch, label_encoder=le)
        assert len(results) == 10


# ===========================================================================
# TestAPI
# ===========================================================================


class TestAPI:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_classify_valid(self, api_client):
        payload = {
            "aspect_ratio": 0.77,
            "brightness": 0.82,
            "text_density": 0.60,
            "edge_density": 0.20,
            "file_size_kb": 350.0,
        }
        resp = api_client.post("/classify", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] in DOC_TYPES
        assert 0.0 <= body["confidence"] <= 1.0

    def test_classify_returns_probabilities(self, api_client):
        payload = {
            "aspect_ratio": 1.58,
            "brightness": 0.55,
            "text_density": 0.25,
            "edge_density": 0.50,
            "file_size_kb": 250.0,
        }
        resp = api_client.post("/classify", json=payload)
        probs = resp.json()["probabilities"]
        # should have one entry per class
        assert len(probs) == len(DOC_TYPES)

    def test_classify_invalid_input(self, api_client):
        # brightness out of range
        payload = {
            "aspect_ratio": 0.77,
            "brightness": 5.0,
            "text_density": 0.60,
            "edge_density": 0.20,
            "file_size_kb": 350.0,
        }
        resp = api_client.post("/classify", json=payload)
        assert resp.status_code == 422  # validation error


# ===========================================================================
# TestCNN
# ===========================================================================


class TestCNN:
    def test_is_available_returns_bool(self):
        result = cnn_is_available()
        assert isinstance(result, bool)

    def test_import_does_not_crash(self):
        """Even without torch the module should import cleanly."""
        import src.models.cnn as cnn_mod  # noqa: F811

        assert hasattr(cnn_mod, "TORCH_AVAILABLE")
