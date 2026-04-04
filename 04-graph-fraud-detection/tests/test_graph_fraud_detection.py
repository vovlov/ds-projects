"""Tests for Graph Fraud Detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fraud.data.dataset import (
    generate_synthetic_transactions,
    get_edge_index,
    get_feature_matrix,
)
from fraud.data.elliptic import (
    ELLIPTIC_N_FEATURES,
    generate_mock_elliptic,
    get_labeled_split,
    load_elliptic_dataset,
)
from fraud.models.baseline.tabular import train_baseline
from fraud.models.baseline.vae import is_available as vae_available


class TestDataGeneration:
    def test_generate_data_shape(self):
        data = generate_synthetic_transactions(n_nodes=100, n_transactions=500)
        assert len(data["nodes"]) == 100
        assert len(data["edges"]) == 500

    def test_generate_data_fraud_rate(self):
        data = generate_synthetic_transactions(n_nodes=1000, fraud_rate=0.1, seed=42)
        n_fraud = sum(n["is_fraud"] for n in data["nodes"])
        assert 50 < n_fraud < 200

    def test_feature_matrix(self):
        data = generate_synthetic_transactions(n_nodes=100)
        X, y = get_feature_matrix(data)
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_edge_index(self):
        data = generate_synthetic_transactions(n_nodes=100, n_transactions=500)
        edge_index = get_edge_index(data)
        assert edge_index.shape == (2, 500)
        assert edge_index.min() >= 0
        assert edge_index.max() < 100

    def test_no_self_loops(self):
        data = generate_synthetic_transactions(n_nodes=100, n_transactions=500)
        edge_index = get_edge_index(data)
        assert not np.any(edge_index[0] == edge_index[1])

    def test_deterministic(self):
        data1 = generate_synthetic_transactions(seed=42)
        data2 = generate_synthetic_transactions(seed=42)
        assert data1["nodes"] == data2["nodes"]

    def test_node_features_positive(self):
        data = generate_synthetic_transactions(n_nodes=100)
        for node in data["nodes"]:
            assert node["avg_amount"] > 0
            assert node["n_transactions"] >= 0
            assert node["account_age_days"] >= 0


class TestBaseline:
    def test_train_baseline_runs(self):
        data = generate_synthetic_transactions(n_nodes=200, fraud_rate=0.1)
        X, y = get_feature_matrix(data)
        result = train_baseline(X, y)
        assert "f1_score" in result
        assert "roc_auc" in result
        assert 0 <= result["f1_score"] <= 1
        assert 0 <= result["roc_auc"] <= 1

    def test_baseline_predictions_shape(self):
        data = generate_synthetic_transactions(n_nodes=200, fraud_rate=0.1)
        X, y = get_feature_matrix(data)
        result = train_baseline(X, y, test_size=0.2)
        assert len(result["y_pred"]) == int(200 * 0.2)
        assert len(result["y_proba"]) == int(200 * 0.2)


class TestEllipticDataset:
    def test_mock_returns_expected_keys(self):
        data = generate_mock_elliptic(n_nodes=200, n_edges=300, seed=0)
        for key in ("node_ids", "features", "labels", "edges", "is_mock"):
            assert key in data
        assert data["is_mock"] is True

    def test_mock_feature_shape(self):
        data = generate_mock_elliptic(n_nodes=200, n_edges=300, seed=0)
        assert data["features"].shape == (200, ELLIPTIC_N_FEATURES)

    def test_mock_edge_shape(self):
        data = generate_mock_elliptic(n_nodes=200, n_edges=300, seed=0)
        # n_edges может быть немного меньше из-за удаления self-loops
        assert data["edges"].shape[0] == 2
        assert data["edges"].shape[1] <= 300

    def test_mock_labels_valid_values(self):
        data = generate_mock_elliptic(n_nodes=200, seed=0)
        # Метки: 1 (illicit) и 2 (licit)
        assert set(np.unique(data["labels"])).issubset({1, 2})

    def test_no_self_loops(self):
        data = generate_mock_elliptic(n_nodes=200, n_edges=400, seed=0)
        src, dst = data["edges"][0], data["edges"][1]
        assert not np.any(src == dst)

    def test_load_elliptic_fallback_to_mock(self):
        # data_dir=None → mock
        data = load_elliptic_dataset(data_dir=None)
        assert data["is_mock"] is True

    def test_load_elliptic_nonexistent_dir(self, tmp_path):
        # Несуществующая директория → mock
        data = load_elliptic_dataset(data_dir=tmp_path / "no_such_dir")
        assert data["is_mock"] is True

    def test_get_labeled_split_binary_labels(self):
        data = generate_mock_elliptic(n_nodes=300, seed=7)
        X, y = get_labeled_split(data)
        # Метки должны быть бинарными
        assert set(np.unique(y)).issubset({0, 1})

    def test_get_labeled_split_feature_shape(self):
        data = generate_mock_elliptic(n_nodes=300, seed=7)
        X, y = get_labeled_split(data)
        assert X.shape[1] == ELLIPTIC_N_FEATURES
        assert X.shape[0] == y.shape[0]

    def test_get_labeled_split_excludes_unknown(self):
        data = generate_mock_elliptic(n_nodes=300, seed=7)
        # Mock содержит только 1/2, но добавим unknown=0 вручную
        data["labels"][0] = 0
        X, y = get_labeled_split(data)
        # Все метки должны быть 0 или 1 (без unknown)
        assert len(X) == (data["labels"] > 0).sum()


class TestVAEModule:
    def test_is_available_returns_bool(self):
        result = vae_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not vae_available(), reason="PyTorch not available")
    def test_vae_train_returns_expected_keys(self):
        from fraud.models.baseline.vae import train_vae

        rng = np.random.default_rng(42)
        n = 300
        X = rng.standard_normal((n, 10)).astype(np.float32)
        y = np.array([0] * 240 + [1] * 60, dtype=np.int32)

        result = train_vae(X, y, epochs=5, hidden_dim=32, latent_dim=8)
        expected_keys = (
            "model",
            "scaler",
            "threshold",
            "f1_score",
            "roc_auc",
            "y_test",
            "y_pred",
            "y_score",
        )
        for key in expected_keys:
            assert key in result

    @pytest.mark.skipif(not vae_available(), reason="PyTorch not available")
    def test_vae_metrics_in_valid_range(self):
        from fraud.models.baseline.vae import train_vae

        rng = np.random.default_rng(0)
        n = 300
        X = rng.standard_normal((n, 10)).astype(np.float32)
        y = np.array([0] * 240 + [1] * 60, dtype=np.int32)

        result = train_vae(X, y, epochs=5, hidden_dim=32, latent_dim=8)
        assert 0.0 <= result["f1_score"] <= 1.0
        assert 0.0 <= result["roc_auc"] <= 1.0

    @pytest.mark.skipif(not vae_available(), reason="PyTorch not available")
    def test_vae_prediction_shape(self):
        from fraud.models.baseline.vae import train_vae

        rng = np.random.default_rng(1)
        n = 300
        X = rng.standard_normal((n, 10)).astype(np.float32)
        y = np.array([0] * 240 + [1] * 60, dtype=np.int32)

        result = train_vae(X, y, epochs=5, test_size=0.2, hidden_dim=32, latent_dim=8)
        expected_test_size = int(n * 0.2)
        assert len(result["y_pred"]) == expected_test_size
        assert len(result["y_score"]) == expected_test_size

    @pytest.mark.skipif(vae_available(), reason="Only runs when PyTorch is absent")
    def test_vae_raises_without_torch(self):
        from fraud.models.baseline.vae import FraudVAE

        with pytest.raises(RuntimeError, match="PyTorch"):
            FraudVAE(input_dim=10)


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_score_endpoint(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score",
            json={
                "avg_amount": 150.0,
                "n_transactions": 5,
                "account_age_days": 180.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert 0 <= data["fraud_probability"] <= 1

    def test_score_batch_endpoint(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score/batch",
            json=[
                {"avg_amount": 150.0, "n_transactions": 5, "account_age_days": 180.0},
                {"avg_amount": 5000.0, "n_transactions": 20, "account_age_days": 5.0},
            ],
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2
