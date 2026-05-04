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
from fraud.models.temporal import (
    TEMPORAL_FEATURE_NAMES,
    NodeTemporalFeatures,
    TemporalConfig,
    TemporalFeatureExtractor,
    explain_temporal_features,
)


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


class TestTemporalFeatures:
    """Тесты для TemporalFeatureExtractor и NodeTemporalFeatures."""

    def _make_data(self, n_nodes: int = 50, n_transactions: int = 200) -> dict:
        return generate_synthetic_transactions(
            n_nodes=n_nodes, n_transactions=n_transactions, fraud_rate=0.1, seed=42
        )

    def test_node_temporal_features_defaults(self):
        feat = NodeTemporalFeatures()
        assert feat.velocity_ratio == 0.0
        assert feat.burst_score == 0.0
        assert feat.amount_hhi == 0.0
        assert feat.recent_amount_ratio == 0.0
        assert feat.neighbor_fraud_ratio == 0.0
        assert feat.hub_proximity == 0.0

    def test_node_temporal_features_to_array_shape(self):
        feat = NodeTemporalFeatures(velocity_ratio=0.5, burst_score=1.2)
        arr = feat.to_array()
        assert arr.shape == (6,)
        assert arr.dtype == np.float32

    def test_node_temporal_features_to_array_values(self):
        feat = NodeTemporalFeatures(
            velocity_ratio=0.8,
            burst_score=2.5,
            amount_hhi=0.4,
            recent_amount_ratio=0.6,
            neighbor_fraud_ratio=0.15,
            hub_proximity=1.3,
        )
        arr = feat.to_array()
        assert arr[0] == pytest.approx(0.8, abs=1e-5)
        assert arr[1] == pytest.approx(2.5, abs=1e-5)
        assert arr[4] == pytest.approx(0.15, abs=1e-5)

    def test_feature_names_count(self):
        assert len(TEMPORAL_FEATURE_NAMES) == 6

    def test_extract_output_shape(self):
        data = self._make_data(n_nodes=50)
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert result.shape == (50, 6)
        assert result.dtype == np.float32

    def test_augment_features_shape(self):
        data = self._make_data(n_nodes=50)
        X, _ = get_feature_matrix(data)
        extractor = TemporalFeatureExtractor()
        X_aug = extractor.augment_features(X, data)
        assert X_aug.shape == (50, 9)  # 3 base + 6 temporal

    def test_velocity_ratio_in_unit_interval(self):
        data = self._make_data()
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert (result[:, 0] >= 0.0).all()
        assert (result[:, 0] <= 1.0).all()

    def test_burst_score_non_negative(self):
        data = self._make_data()
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert (result[:, 1] >= 0.0).all()

    def test_amount_hhi_in_unit_interval(self):
        data = self._make_data()
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert (result[:, 2] >= 0.0).all()
        assert (result[:, 2] <= 1.0 + 1e-5).all()

    def test_neighbor_fraud_ratio_in_unit_interval(self):
        data = self._make_data()
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert (result[:, 4] >= 0.0).all()
        assert (result[:, 4] <= 1.0 + 1e-5).all()

    def test_hub_proximity_non_negative(self):
        data = self._make_data()
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert (result[:, 5] >= 0.0).all()

    def test_isolated_node_returns_zeros(self):
        # Узел без рёбер → все velocity-признаки = 0
        data = {
            "nodes": [
                {
                    "id": 0,
                    "avg_amount": 100.0,
                    "n_transactions": 1,
                    "account_age_days": 100.0,
                    "is_fraud": 0,
                }
            ],
            "edges": [],
        }
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)
        assert result.shape == (1, 6)
        # Без рёбер velocity_ratio = 0
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_custom_config(self):
        config = TemporalConfig(time_window=7.0, decay_factor=0.5, min_edges_for_burst=3)
        extractor = TemporalFeatureExtractor(config=config)
        data = self._make_data()
        result = extractor.extract(data)
        assert result.shape[1] == 6

    def test_compute_node_features_returns_dataclass(self):
        data = self._make_data()
        extractor = TemporalFeatureExtractor()
        node_id = data["nodes"][0]["id"]
        feat = extractor.compute_node_features(data, node_id)
        assert isinstance(feat, NodeTemporalFeatures)

    def test_explain_temporal_features_high_risk(self):
        feat = NodeTemporalFeatures(
            velocity_ratio=0.9,
            burst_score=4.0,
            amount_hhi=0.8,
            neighbor_fraud_ratio=0.5,
        )
        explanations = explain_temporal_features(feat)
        assert "velocity_ratio" in explanations
        assert "HIGH" in explanations["velocity_ratio"]
        assert "burst_score" in explanations
        assert "HIGH" in explanations["burst_score"]

    def test_explain_temporal_features_low_risk(self):
        feat = NodeTemporalFeatures(
            velocity_ratio=0.1,
            burst_score=0.3,
            amount_hhi=0.1,
            neighbor_fraud_ratio=0.02,
        )
        explanations = explain_temporal_features(feat)
        assert "LOW" in explanations["velocity_ratio"]
        assert "LOW" in explanations["burst_score"]

    def test_fraud_nodes_higher_neighbor_fraud_ratio(self):
        # Мошеннические узлы образуют кластеры → сосед-мошенник чаще
        data = generate_synthetic_transactions(
            n_nodes=200, n_transactions=1000, fraud_rate=0.15, seed=99
        )
        extractor = TemporalFeatureExtractor()
        result = extractor.extract(data)

        fraud_mask = np.array([n["is_fraud"] for n in data["nodes"]], dtype=bool)
        legit_mask = ~fraud_mask

        if fraud_mask.sum() > 0 and legit_mask.sum() > 0:
            fraud_neighbor_ratio = result[fraud_mask, 4].mean()
            legit_neighbor_ratio = result[legit_mask, 4].mean()
            # Мошенники должны иметь в среднем больше соседей-мошенников
            assert fraud_neighbor_ratio >= legit_neighbor_ratio


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_shows_temporal_model_field(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert "temporal_model_loaded" in resp.json()

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

    def test_score_graph_endpoint_returns_200(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score/graph",
            json={
                "avg_amount": 500.0,
                "n_transactions": 8,
                "account_age_days": 15.0,
                "velocity_ratio": 0.85,
                "burst_score": 3.2,
                "amount_hhi": 0.6,
                "recent_amount_ratio": 0.9,
                "neighbor_fraud_ratio": 0.4,
                "hub_proximity": 1.5,
            },
        )
        assert resp.status_code == 200

    def test_score_graph_response_has_temporal_flags(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score/graph",
            json={
                "avg_amount": 500.0,
                "n_transactions": 8,
                "account_age_days": 15.0,
                "velocity_ratio": 0.9,
                "burst_score": 4.0,
                "amount_hhi": 0.8,
                "recent_amount_ratio": 0.95,
                "neighbor_fraud_ratio": 0.5,
                "hub_proximity": 2.0,
            },
        )
        data = resp.json()
        assert "temporal_flags" in data
        assert "feature_contributions" in data

    def test_score_graph_response_feature_contributions_keys(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score/graph",
            json={
                "avg_amount": 150.0,
                "n_transactions": 3,
                "account_age_days": 200.0,
                "velocity_ratio": 0.1,
                "burst_score": 0.2,
                "amount_hhi": 0.1,
                "recent_amount_ratio": 0.1,
                "neighbor_fraud_ratio": 0.0,
                "hub_proximity": 0.5,
            },
        )
        data = resp.json()
        for name in TEMPORAL_FEATURE_NAMES:
            assert name in data["feature_contributions"]

    def test_score_graph_probability_in_range(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score/graph",
            json={
                "avg_amount": 200.0,
                "n_transactions": 4,
                "account_age_days": 90.0,
                "velocity_ratio": 0.5,
                "burst_score": 1.5,
                "amount_hhi": 0.3,
                "recent_amount_ratio": 0.5,
                "neighbor_fraud_ratio": 0.1,
                "hub_proximity": 1.0,
            },
        )
        data = resp.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_score_graph_risk_level_valid(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/score/graph",
            json={
                "avg_amount": 150.0,
                "n_transactions": 5,
                "account_age_days": 180.0,
                "velocity_ratio": 0.0,
                "burst_score": 0.0,
                "amount_hhi": 0.0,
                "recent_amount_ratio": 0.0,
                "neighbor_fraud_ratio": 0.0,
                "hub_proximity": 0.0,
            },
        )
        assert resp.json()["risk_level"] in ("low", "medium", "high")
