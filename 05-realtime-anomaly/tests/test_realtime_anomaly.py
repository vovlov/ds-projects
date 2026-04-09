"""Tests for Realtime Anomaly Detection."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.data.generator import generate_timeseries, to_windows
from anomaly.models.detector import MultiMetricDetector, StatisticalDetector


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
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_detect_endpoint(self):
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

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


class TestPrometheusExporter:
    """Тесты Prometheus-экспортера метрик."""

    def test_is_available(self):
        from anomaly.metrics.prometheus_exporter import is_available

        # Просто проверяем что функция работает (значение зависит от окружения)
        assert isinstance(is_available(), bool)

    def test_metrics_init(self):
        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)
        assert m is not None

    def test_set_model_config(self):
        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)
        m.set_model_config(threshold_sigma=3.0, window_size=50)

        summary = m.get_summary()
        assert summary["threshold_sigma"] == 3.0
        assert summary["window_size"] == 50.0

    def test_record_request(self):
        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)
        m.record_request(n_points=100, n_anomalies=3, scores=[3.5, 4.1, 5.0])

        summary = m.get_summary()
        assert summary["requests_total"] == 1
        assert summary["points_total"] == 100
        assert summary["anomalies_total"] == 3
        assert summary["anomaly_rate"] == 0.03

    def test_anomaly_rate_zero_when_no_points(self):
        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)
        summary = m.get_summary()
        assert summary["anomaly_rate"] == 0.0

    def test_track_detection_context_manager(self):
        import time

        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)

        with m.track_detection():
            time.sleep(0.001)  # минимальная задержка для проверки

        # Гистограмма должна записать хотя бы одно наблюдение
        # (проверяем через text output — содержит _bucket)
        text = m.get_metrics_text()
        assert "anomaly_detector_detection_seconds" in text

    def test_get_metrics_text_format(self):
        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)
        m.set_model_config(threshold_sigma=3.0, window_size=50)
        m.record_request(n_points=10, n_anomalies=1, scores=[4.2])

        text = m.get_metrics_text()
        # Проверяем что все ключевые метрики присутствуют
        assert "anomaly_detector_requests_total" in text
        assert "anomaly_detector_points_total" in text
        assert "anomaly_detector_anomalies_total" in text
        assert "anomaly_detector_anomaly_score" in text
        assert "anomaly_detector_detection_seconds" in text
        assert "anomaly_detector_threshold" in text
        assert "anomaly_detector_window_size" in text

    def test_multiple_requests_accumulate(self):
        from anomaly.metrics.prometheus_exporter import AnomalyMetrics, is_available

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        from prometheus_client import CollectorRegistry

        registry = CollectorRegistry()
        m = AnomalyMetrics(registry=registry)

        m.record_request(n_points=50, n_anomalies=2, scores=[3.1, 4.5])
        m.record_request(n_points=30, n_anomalies=1, scores=[5.0])

        summary = m.get_summary()
        assert summary["requests_total"] == 2
        assert summary["points_total"] == 80
        assert summary["anomalies_total"] == 3


class TestAPIWithMetrics:
    """Тесты API-эндпоинтов с метриками."""

    def test_health_includes_prometheus_status(self):
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "prometheus" in data
        assert isinstance(data["prometheus"], bool)

    def test_metrics_endpoint_accessible(self):
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Должен вернуть text/plain (Prometheus format)
        assert "text/plain" in resp.headers["content-type"]

    def test_detect_updates_metrics(self):
        from anomaly.api.app import app, is_available
        from fastapi.testclient import TestClient

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        client = TestClient(app)
        points = [
            {"timestamp": i, "cpu": 45.0, "latency": 50.0, "requests": 1000.0} for i in range(60)
        ]
        points[55]["cpu"] = 99.0

        client.post("/detect", json=points)

        # После запроса /metrics должен содержать обновлённые данные
        resp = client.get("/metrics")
        assert "anomaly_detector_points_total" in resp.text

    def test_health_stats_after_detect(self):
        from anomaly.api.app import app, is_available
        from fastapi.testclient import TestClient

        if not is_available():
            import pytest

            pytest.skip("prometheus_client not installed")

        client = TestClient(app)
        points = [
            {"timestamp": i, "cpu": 45.0, "latency": 50.0, "requests": 1000.0} for i in range(60)
        ]
        client.post("/detect", json=points)

        resp = client.get("/health")
        data = resp.json()
        if "stats" in data:
            assert data["stats"]["points_total"] >= 60


class TestAlerting:
    def test_create_alert(self):
        from anomaly.alerting.webhook import create_alert

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
        from anomaly.alerting.webhook import Alert, format_alert_payload

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
        from anomaly.alerting.webhook import Alert, send_alert

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
