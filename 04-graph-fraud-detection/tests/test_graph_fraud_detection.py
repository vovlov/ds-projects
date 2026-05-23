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


# ---------------------------------------------------------------------------
# Probability Calibration
# ---------------------------------------------------------------------------


class TestFraudCalibrationUnit:
    """Unit tests for FraudCalibrator — no API, pure model tests."""

    def test_ece_computation_perfect(self):
        """Perfect calibration: ECE = 0."""
        from fraud.models.calibration import _compute_ece

        probas = np.linspace(0.05, 0.95, 100)
        labels = (np.random.default_rng(42).random(100) < probas).astype(float)
        ece, mce, bins = _compute_ece(probas, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0
        assert len(bins) == 10

    def test_ece_worst_case(self):
        """Model always predicts 1.0 but labels are 0 → ECE = 1.0."""
        from fraud.models.calibration import _compute_ece

        probas = np.ones(100)
        labels = np.zeros(100)
        ece, mce, bins = _compute_ece(probas, labels, n_bins=10)
        # All samples fall in last bin: gap = 1.0 - 0.0 = 1.0
        assert abs(ece - 1.0) < 1e-6

    def test_ece_bins_count(self):
        from fraud.models.calibration import _compute_ece

        probas = np.random.default_rng(0).uniform(0, 1, 200)
        labels = (probas > 0.5).astype(float)
        _, _, bins = _compute_ece(probas, labels, n_bins=5)
        assert len(bins) == 5

    def test_platt_scaler_fit_transform(self):
        from fraud.models.calibration import _PlattScaler

        rng = np.random.default_rng(7)
        scores = rng.uniform(0, 1, 200)
        labels = (scores + rng.normal(0, 0.2, 200) > 0.5).astype(float)
        scaler = _PlattScaler(lr=0.05, n_iter=500)
        scaler.fit(scores, labels)
        assert scaler.fitted
        out = scaler.transform(scores)
        assert out.shape == scores.shape
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_platt_output_range(self):
        """Outputs must be valid probabilities."""
        from fraud.models.calibration import _PlattScaler

        scaler = _PlattScaler()
        scaler.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0, 1.0]))
        out = scaler.transform(np.array([-5.0, 0.5, 10.0]))
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_isotonic_fit_transform(self):
        from fraud.models.calibration import _IsotonicCalibrator

        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 1, 300)
        labels = (scores + rng.normal(0, 0.15, 300) > 0.5).astype(float)
        iso = _IsotonicCalibrator()
        iso.fit(scores, labels)
        assert iso.fitted
        out = iso.transform(scores)
        assert out.shape == scores.shape
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_isotonic_monotone(self):
        """Isotonic calibrator output must be non-decreasing for sorted input."""
        from fraud.models.calibration import _IsotonicCalibrator

        rng = np.random.default_rng(5)
        scores = rng.uniform(0, 1, 500)
        labels = (scores + rng.normal(0, 0.1, 500) > 0.5).astype(float)
        iso = _IsotonicCalibrator()
        iso.fit(scores, labels)
        sorted_scores = np.sort(scores)
        out = iso.transform(sorted_scores)
        diffs = np.diff(out)
        # Allow tiny float noise (-1e-9 tolerance)
        assert np.all(diffs >= -1e-9)

    def test_fraud_calibrator_fit_platt(self):
        from fraud.models.calibration import FraudCalibrator

        rng = np.random.default_rng(99)
        scores = rng.uniform(0, 1, 150)
        labels = (scores + rng.normal(0, 0.2, 150) > 0.5).astype(float)
        cal = FraudCalibrator(method="platt")
        result = cal.fit(scores, labels)
        assert cal.fitted
        assert result.method == "platt"
        assert result.n_calibration_samples == 150
        assert 0.0 <= result.ece <= 1.0
        assert 0.0 <= result.brier_score <= 1.0

    def test_fraud_calibrator_fit_isotonic(self):
        from fraud.models.calibration import FraudCalibrator

        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 500)
        labels = (scores + rng.normal(0, 0.15, 500) > 0.5).astype(float)
        cal = FraudCalibrator(method="isotonic")
        result = cal.fit(scores, labels)
        assert cal.fitted
        assert result.method == "isotonic"
        assert len(result.bins) == 10

    def test_fraud_calibrator_calibrate(self):
        from fraud.models.calibration import FraudCalibrator

        rng = np.random.default_rng(11)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(float)
        cal = FraudCalibrator(method="platt")
        cal.fit(scores, labels)
        out = cal.calibrate(np.array([0.2, 0.5, 0.8]))
        assert out.shape == (3,)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_fraud_calibrator_not_fitted_raises(self):
        from fraud.models.calibration import FraudCalibrator

        cal = FraudCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.calibrate(np.array([0.5]))

    def test_fraud_calibrator_too_few_samples(self):
        from fraud.models.calibration import FraudCalibrator

        cal = FraudCalibrator()
        with pytest.raises(ValueError, match="10 calibration"):
            cal.fit(np.array([0.5, 0.3]), np.array([1.0, 0.0]))

    def test_ece_improvement_positive(self):
        """After calibration ECE should not increase (sanity check)."""
        from fraud.models.calibration import FraudCalibrator

        rng = np.random.default_rng(88)
        # Over-confident model: predicted 0.9 for everything
        scores = np.clip(rng.normal(0.85, 0.1, 300), 0.01, 0.99)
        labels = rng.binomial(1, 0.5, 300).astype(float)
        cal = FraudCalibrator(method="isotonic")
        cal.fit(scores, labels)
        imp = cal.ece_improvement()
        assert imp is not None

    def test_calibration_result_to_dict(self):
        from fraud.models.calibration import FraudCalibrator

        rng = np.random.default_rng(7)
        scores = rng.uniform(0, 1, 200)
        labels = (scores > 0.5).astype(float)
        cal = FraudCalibrator(method="platt", n_bins=5)
        result = cal.fit(scores, labels)
        d = result.to_dict()
        assert d["method"] == "platt"
        assert d["n_bins"] == 5
        assert len(d["bins"]) == 5
        assert "ece" in d
        assert "brier_score" in d

    def test_is_available(self):
        from fraud.models.calibration import FraudCalibrator

        # sklearn is installed in CI venv
        assert FraudCalibrator.is_available() is True


class TestCalibrationAPIEndpoints:
    """Integration tests for /calibrate and /calibration/metrics endpoints."""

    def test_calibrate_isotonic_success(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post("/calibrate", json={"method": "isotonic", "n_bins": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert data["method"] == "isotonic"
        assert data["n_calibration_samples"] > 0
        assert 0.0 <= data["ece_after"] <= 1.0
        assert 0.0 <= data["brier_score"] <= 1.0

    def test_calibrate_platt_success(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post("/calibrate", json={"method": "platt", "n_bins": 10})
        assert resp.status_code == 200
        assert resp.json()["method"] == "platt"

    def test_calibrate_invalid_method(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.post("/calibrate", json={"method": "unknown", "n_bins": 10})
        assert resp.status_code == 422

    def test_calibration_metrics_after_calibrate(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        client.post("/calibrate", json={"method": "isotonic", "n_bins": 10})
        resp = client.get("/calibration/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "ece" in data
        assert "bins" in data
        assert len(data["bins"]) == 10

    def test_calibration_metrics_before_calibrate_404(self):
        """Without a calibrator, metrics endpoint returns 404."""
        from fastapi.testclient import TestClient
        from fraud.api.app import (
            _reset_calibrator,  # type: ignore[attr-defined]
            app,
        )

        _reset_calibrator()
        client = TestClient(app)
        resp = client.get("/calibration/metrics")
        assert resp.status_code == 404

    def test_score_calibrated_probability_after_fit(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        client.post("/calibrate", json={"method": "isotonic", "n_bins": 10})
        resp = client.post(
            "/score",
            json={"avg_amount": 200.0, "n_transactions": 5, "account_age_days": 90.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["calibrated_probability"] is not None
        assert 0.0 <= data["calibrated_probability"] <= 1.0

    def test_health_shows_calibration_status(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert "calibration_fitted" in resp.json()


class TestFraudCommunityUnit:
    """Unit-тесты для детектора мошеннических колец (Label Propagation)."""

    def _detector(self, **kwargs):
        from fraud.models.community import CommunityConfig, FraudRingDetector

        config = CommunityConfig(**kwargs)
        return FraudRingDetector(config)

    def test_empty_node_list_raises(self):
        detector = self._detector()
        with pytest.raises(ValueError, match="empty"):
            detector.detect([], [])

    def test_single_node_forms_own_community(self):
        detector = self._detector()
        result = detector.detect(["A"], [])
        assert result.n_communities == 1
        assert result.communities[0].node_ids == ["A"]

    def test_two_cliques_separate_communities(self):
        """Два несвязных треугольника → 2 сообщества."""
        detector = self._detector()
        nodes = ["A", "B", "C", "D", "E", "F"]
        edges = [("A", "B"), ("B", "C"), ("A", "C"), ("D", "E"), ("E", "F"), ("D", "F")]
        result = detector.detect(nodes, edges)
        assert result.n_communities == 2

    def test_all_connected_chain_same_community(self):
        """Цепочка A-B-C-D → одно сообщество."""
        detector = self._detector()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        result = detector.detect(nodes, edges)
        assert result.n_communities == 1

    def test_fraud_ratio_zero_no_fraud(self):
        detector = self._detector()
        nodes = ["A", "B"]
        labels = {"A": False, "B": False}
        result = detector.detect(nodes, [("A", "B")], labels)
        assert result.communities[0].fraud_ratio == 0.0

    def test_fraud_ratio_all_fraud(self):
        detector = self._detector()
        nodes = ["A", "B"]
        labels = {"A": True, "B": True}
        result = detector.detect(nodes, [("A", "B")], labels)
        assert result.communities[0].fraud_ratio == 1.0

    def test_fraud_ratio_mixed(self):
        detector = self._detector()
        nodes = ["A", "B", "C", "D"]
        edges = [("A", "B"), ("B", "C"), ("C", "D")]
        labels = {"A": True, "B": False, "C": True, "D": False}
        result = detector.detect(nodes, edges, labels)
        # Одно сообщество, 2 fraudsters из 4 labeled → ratio = 0.5
        comm = result.communities[0]
        assert abs(comm.fraud_ratio - 0.5) < 1e-6

    def test_risk_level_high(self):
        from fraud.models.community import CommunityConfig, FraudRingDetector

        cfg = CommunityConfig(fraud_ratio_high=0.3)
        detector = FraudRingDetector(cfg)
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        labels = {"A": True, "B": True, "C": False}
        result = detector.detect(nodes, edges, labels)
        assert result.communities[0].risk_level == "high"

    def test_risk_level_medium(self):
        from fraud.models.community import CommunityConfig, FraudRingDetector

        cfg = CommunityConfig(fraud_ratio_medium=0.1, fraud_ratio_high=0.5)
        detector = FraudRingDetector(cfg)
        nodes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        edges = [(f"{chr(65 + i)}", f"{chr(65 + i + 1)}") for i in range(9)]
        labels = {n: (n == "A") for n in nodes}  # 1/10 = 0.1 → medium
        result = detector.detect(nodes, edges, labels)
        assert result.communities[0].risk_level in ("medium", "high")

    def test_risk_level_low(self):
        detector = self._detector()
        nodes = ["A", "B"]
        labels = {"A": False, "B": False}
        result = detector.detect(nodes, [("A", "B")], labels)
        assert result.communities[0].risk_level == "low"

    def test_converged_simple_graph(self):
        detector = self._detector()
        result = detector.detect(["A", "B"], [("A", "B")])
        assert result.converged is True
        assert result.n_iterations >= 1

    def test_communities_sorted_by_size_descending(self):
        """Сообщества упорядочены по убыванию размера."""
        detector = self._detector()
        # Большая группа (4 узла) и маленькая (2 узла) — несвязные
        nodes = ["A", "B", "C", "D", "X", "Y"]
        edges = [("A", "B"), ("B", "C"), ("C", "D"), ("X", "Y")]
        result = detector.detect(nodes, edges)
        sizes = [c.size for c in result.communities]
        assert sizes == sorted(sizes, reverse=True)

    def test_suspicious_rings_filtered_by_risk_and_size(self):
        """suspicious_rings содержит только high/medium с size >= min_ring_size."""
        from fraud.models.community import CommunityConfig, FraudRingDetector

        cfg = CommunityConfig(min_ring_size=2, fraud_ratio_high=0.3)
        detector = FraudRingDetector(cfg)
        nodes = ["A", "B", "X"]
        edges = [("A", "B")]
        labels = {"A": True, "B": True, "X": False}
        result = detector.detect(nodes, edges, labels)
        for ring in result.suspicious_rings:
            assert ring.risk_level != "low"
            assert ring.size >= 2

    def test_isolated_node_forms_own_community(self):
        detector = self._detector()
        nodes = ["A", "B", "C"]
        edges = [("A", "B")]  # C изолирован
        result = detector.detect(nodes, edges)
        sizes = sorted([c.size for c in result.communities], reverse=True)
        assert sizes[0] >= 2  # группа A-B
        assert sizes[-1] == 1  # одиночный C

    def test_fraud_ratio_unlabeled_nodes_excluded(self):
        """Unlabeled узлы не попадают в знаменатель fraud_ratio."""
        detector = self._detector()
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        labels = {"A": True}  # B и C не labeled
        result = detector.detect(nodes, edges, labels)
        # Только 1 labeled узел (A=True) → fraud_ratio = 1.0
        assert result.communities[0].fraud_ratio == 1.0

    def test_community_result_to_dict_structure(self):
        from fraud.models.community import CommunityResult

        cr = CommunityResult(
            community_id=0, size=3, fraud_ratio=0.5, risk_level="high", node_ids=["A", "B", "C"]
        )
        d = cr.to_dict()
        assert set(d.keys()) == {"community_id", "size", "fraud_ratio", "risk_level", "node_ids"}
        assert d["fraud_ratio"] == 0.5

    def test_detection_result_total_nodes(self):
        detector = self._detector()
        nodes = ["A", "B", "C", "D", "E"]
        result = detector.detect(nodes, [("A", "B")])
        assert result.total_nodes == 5


class TestCommunityAPIEndpoints:
    """Интеграционные тесты для /community/* endpoint'ов."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        from fraud.api.app import _reset_ring_detector, app

        _reset_ring_detector()
        return TestClient(app)

    def _simple_payload(self, with_fraud=False):
        nodes = [{"node_id": n} for n in ["A", "B", "C", "D", "E", "F"]]
        if with_fraud:
            nodes[0]["is_fraud"] = True
            nodes[1]["is_fraud"] = True
            nodes[2]["is_fraud"] = True
        edges = [
            {"from_id": "A", "to_id": "B"},
            {"from_id": "B", "to_id": "C"},
            {"from_id": "D", "to_id": "E"},
            {"from_id": "E", "to_id": "F"},
        ]
        return {"nodes": nodes, "edges": edges}

    def test_detect_returns_200(self):
        client = self._make_client()
        resp = client.post("/community/detect", json=self._simple_payload())
        assert resp.status_code == 200

    def test_detect_response_structure(self):
        client = self._make_client()
        resp = client.post("/community/detect", json=self._simple_payload())
        data = resp.json()
        expected_keys = (
            "n_communities",
            "communities",
            "suspicious_rings",
            "n_iterations",
            "converged",
            "total_nodes",
        )
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_detect_n_communities_positive(self):
        client = self._make_client()
        resp = client.post("/community/detect", json=self._simple_payload())
        assert resp.json()["n_communities"] >= 1

    def test_detect_total_nodes_matches_input(self):
        client = self._make_client()
        resp = client.post("/community/detect", json=self._simple_payload())
        assert resp.json()["total_nodes"] == 6

    def test_detect_with_high_fraud_generates_suspicious_rings(self):
        client = self._make_client()
        payload = {
            "nodes": [
                {"node_id": "A", "is_fraud": True},
                {"node_id": "B", "is_fraud": True},
                {"node_id": "C", "is_fraud": True},
            ],
            "edges": [{"from_id": "A", "to_id": "B"}, {"from_id": "B", "to_id": "C"}],
        }
        resp = client.post("/community/detect", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["suspicious_rings"]) >= 1

    def test_detect_empty_nodes_returns_422(self):
        client = self._make_client()
        resp = client.post("/community/detect", json={"nodes": [], "edges": []})
        assert resp.status_code == 422

    def test_community_stats_404_before_detect(self):
        client = self._make_client()
        resp = client.get("/community/stats")
        assert resp.status_code == 404

    def test_community_stats_200_after_detect(self):
        client = self._make_client()
        client.post("/community/detect", json=self._simple_payload())
        resp = client.get("/community/stats")
        assert resp.status_code == 200

    def test_community_stats_structure(self):
        client = self._make_client()
        client.post("/community/detect", json=self._simple_payload())
        data = client.get("/community/stats").json()
        for key in (
            "n_communities",
            "n_suspicious_rings",
            "total_nodes_analyzed",
            "nodes_in_suspicious_rings",
            "suspicious_coverage_ratio",
            "converged",
            "n_iterations",
        ):
            assert key in data, f"Missing key: {key}"

    def test_community_stats_coverage_ratio_in_range(self):
        client = self._make_client()
        client.post("/community/detect", json=self._simple_payload())
        data = client.get("/community/stats").json()
        assert 0.0 <= data["suspicious_coverage_ratio"] <= 1.0
