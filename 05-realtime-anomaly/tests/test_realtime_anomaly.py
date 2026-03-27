"""Tests for Realtime Anomaly Detection."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generator import generate_timeseries, to_windows
from src.models.detector import MultiMetricDetector, StatisticalDetector


class TestDataGenerator:
    def test_generate_shape(self):
        data = generate_timeseries(n_points=1000)
        assert len(data["timestamps"]) == 1000
        assert len(data["cpu"]) == 1000
        assert len(data["latency"]) == 1000
        assert len(data["requests"]) == 1000
        assert len(data["labels"]) == 1000

    def test_anomaly_rate(self):
        data = generate_timeseries(n_points=5000, anomaly_rate=0.05)
        anomaly_ratio = data["labels"].mean()
        assert 0.01 < anomaly_ratio < 0.15  # roughly around 5%

    def test_cpu_range(self):
        data = generate_timeseries(n_points=1000)
        assert data["cpu"].min() >= 0
        assert data["cpu"].max() <= 100

    def test_latency_positive(self):
        data = generate_timeseries(n_points=1000)
        assert (data["latency"] > 0).all()

    def test_requests_non_negative(self):
        data = generate_timeseries(n_points=1000)
        assert (data["requests"] >= 0).all()

    def test_deterministic(self):
        d1 = generate_timeseries(seed=42)
        d2 = generate_timeseries(seed=42)
        np.testing.assert_array_equal(d1["cpu"], d2["cpu"])

    def test_labels_binary(self):
        data = generate_timeseries(n_points=1000)
        assert set(np.unique(data["labels"])).issubset({0, 1})


class TestWindows:
    def test_window_shape(self):
        data = generate_timeseries(n_points=200)
        X, y = to_windows(data, window_size=30, stride=1)
        assert X.shape[1] == 30  # window size
        assert X.shape[2] == 3  # features (cpu, latency, requests)
        assert len(y) == X.shape[0]

    def test_window_labels_binary(self):
        data = generate_timeseries(n_points=200)
        _, y = to_windows(data, window_size=30)
        assert set(np.unique(y)).issubset({0, 1})


class TestStatisticalDetector:
    def test_detect_returns_result(self):
        detector = StatisticalDetector(window_size=20, threshold_sigma=2.0)
        series = np.random.randn(200)
        result = detector.detect(series)
        assert len(result.scores) == 200
        assert len(result.predictions) == 200
        assert result.threshold == 2.0

    def test_detect_finds_spike(self):
        detector = StatisticalDetector(window_size=20, threshold_sigma=3.0)
        series = np.random.randn(200) * 0.5
        series[150] = 50  # big spike
        result = detector.detect(series)
        assert result.predictions[150] == 1

    def test_normal_data_few_anomalies(self):
        detector = StatisticalDetector(window_size=50, threshold_sigma=3.0)
        series = np.random.randn(1000)
        result = detector.detect(series)
        anomaly_rate = result.predictions.mean()
        assert anomaly_rate < 0.05  # should be very few


class TestMultiMetricDetector:
    def test_detect_synthetic_data(self):
        data = generate_timeseries(n_points=1000, anomaly_rate=0.05)
        detector = MultiMetricDetector(window_size=50, threshold_sigma=3.0)
        result = detector.detect(data)
        assert len(result.scores) == 1000
        assert len(result.predictions) == 1000

    def test_detect_finds_some_anomalies(self):
        data = generate_timeseries(n_points=2000, anomaly_rate=0.05)
        detector = MultiMetricDetector(window_size=50, threshold_sigma=2.5)
        result = detector.detect(data)
        # Should find at least some anomalies
        assert result.predictions.sum() > 0


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_detect_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        points = [
            {"timestamp": i, "cpu": 45.0, "latency": 50.0, "requests": 1000.0} for i in range(60)
        ]
        # Inject anomaly
        points[55]["cpu"] = 99.0
        points[55]["latency"] = 500.0

        resp = client.post("/detect", json=points)
        assert resp.status_code == 200
        assert len(resp.json()) == 60


class TestAlerting:
    def test_create_alert(self):
        from src.alerting.webhook import create_alert

        alert = create_alert(
            timestamp=1000.0,
            cpu=95.0,
            latency=500.0,
            requests=50.0,
            score=5.0,
            threshold=3.0,
        )
        assert alert.severity in ("warning", "critical")
        assert alert.metric_name in ("cpu", "latency", "requests")

    def test_format_alert_payload(self):
        from src.alerting.webhook import Alert, format_alert_payload

        alert = Alert(
            timestamp=1000.0,
            metric_name="cpu",
            value=95.0,
            score=5.0,
            threshold=3.0,
            severity="warning",
        )
        payload = format_alert_payload(alert)
        assert "message" in payload
        assert "severity" in payload
        assert payload["severity"] == "warning"

    def test_send_alert_logs(self):
        from src.alerting.webhook import Alert, send_alert

        alert = Alert(
            timestamp=1000.0,
            metric_name="latency",
            value=500.0,
            score=8.0,
            threshold=3.0,
            severity="critical",
        )
        # Without webhook URL, should just log
        result = send_alert(alert, webhook_url=None)
        assert result is True
