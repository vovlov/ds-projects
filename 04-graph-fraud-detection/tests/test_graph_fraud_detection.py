"""Tests for Graph Fraud Detection."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (
    generate_synthetic_transactions,
    get_edge_index,
    get_feature_matrix,
)
from src.models.baseline.tabular import train_baseline


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


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_score_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

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
        from src.api.app import app

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
