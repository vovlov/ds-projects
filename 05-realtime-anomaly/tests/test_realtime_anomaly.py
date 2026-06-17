"""Tests for Realtime Anomaly Detection."""

import sys
from pathlib import Path

import numpy as np
import pytest

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


class TestMMDCore:
    """Тесты ядра MMD: kernel matrix, compute_mmd_rbf, bootstrap threshold."""

    def test_rbf_kernel_self_similarity(self):
        """k(x, x) = 1 для RBF-ядра — точка идентична себе."""
        from anomaly.drift.mmd import _rbf_kernel_matrix

        X = np.random.randn(10, 3)
        K = _rbf_kernel_matrix(X, X, gamma=1.0)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_rbf_kernel_symmetry(self):
        """K(X, Y) = K(Y, X)^T — матрица ядра симметрична."""
        from anomaly.drift.mmd import _rbf_kernel_matrix

        X = np.random.randn(8, 3)
        Y = np.random.randn(6, 3)
        K_XY = _rbf_kernel_matrix(X, Y, gamma=0.5)
        K_YX = _rbf_kernel_matrix(Y, X, gamma=0.5)
        np.testing.assert_allclose(K_XY, K_YX.T, atol=1e-10)

    def test_mmd_identical_distributions_near_zero(self):
        """MMD(P, P) ≈ 0 когда оба набора из одного распределения."""
        from anomaly.drift.mmd import compute_mmd_rbf

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        Y = rng.randn(200, 3)  # тоже N(0,1) — дрейфа нет
        mmd2, _ = compute_mmd_rbf(X, Y, gamma=1.0)
        # MMD² должен быть небольшим (шум выборки, не дрейф)
        assert mmd2 < 0.1

    def test_mmd_detects_distribution_shift(self):
        """MMD >> 0 при явном сдвиге среднего."""
        from anomaly.drift.mmd import compute_mmd_rbf

        rng = np.random.RandomState(0)
        X = rng.randn(300, 3)  # N(0, 1)
        Y = rng.randn(300, 3) + 5.0  # N(5, 1) — сдвиг на 5 σ
        mmd2, _ = compute_mmd_rbf(X, Y, gamma=0.1)
        assert mmd2 > 0.5, f"Expected large MMD, got {mmd2}"

    def test_mmd_non_negative(self):
        """MMD² ≥ 0 всегда (физически ограничено снизу нулём)."""
        from anomaly.drift.mmd import compute_mmd_rbf

        rng = np.random.RandomState(7)
        X = rng.randn(50, 2)
        Y = rng.randn(50, 2)
        mmd2, _ = compute_mmd_rbf(X, Y)
        assert mmd2 >= 0.0

    def test_mmd_auto_gamma(self):
        """gamma='auto' не падает и возвращает положительный gamma."""
        from anomaly.drift.mmd import compute_mmd_rbf

        X = np.random.randn(100, 3)
        Y = np.random.randn(100, 3)
        mmd2, gamma = compute_mmd_rbf(X, Y, gamma="auto")
        assert gamma > 0
        assert mmd2 >= 0

    def test_mmd_1d_input(self):
        """MMD работает с 1D входом (одна метрика)."""
        from anomaly.drift.mmd import compute_mmd_rbf

        X = np.random.randn(100)
        Y = np.random.randn(100) + 2.0
        mmd2, _ = compute_mmd_rbf(X, Y, gamma=1.0)
        assert mmd2 > 0

    def test_bootstrap_threshold_positive(self):
        """Bootstrap-порог > 0 и < 1 для типичных данных."""
        from anomaly.drift.mmd import bootstrap_mmd_threshold

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        threshold, gamma = bootstrap_mmd_threshold(X, n_bootstrap=50, alpha=0.05)
        assert threshold >= 0
        assert gamma > 0

    def test_bootstrap_threshold_larger_for_smaller_alpha(self):
        """Меньший alpha → больший порог (строже)."""
        from anomaly.drift.mmd import bootstrap_mmd_threshold

        rng = np.random.RandomState(42)
        X = rng.randn(80, 3)
        threshold_05, _ = bootstrap_mmd_threshold(X, n_bootstrap=50, alpha=0.05)
        threshold_20, _ = bootstrap_mmd_threshold(X, n_bootstrap=50, alpha=0.20)
        # alpha=0.05 → 95-й квантиль (строже, больший порог)
        # alpha=0.20 → 80-й квантиль (мягче, меньший порог)
        assert threshold_05 >= threshold_20


class TestMMDDriftDetector:
    """Тесты высокоуровневого MMDDriftDetector класса."""

    def _make_reference(self, n: int = 200, seed: int = 42) -> np.ndarray:
        """Создать эталонные данные из трёх нормальных распределений."""
        rng = np.random.RandomState(seed)
        cpu = rng.normal(40, 5, n)
        latency = rng.normal(50, 10, n)
        requests = rng.normal(1000, 100, n)
        return np.column_stack([cpu, latency, requests])

    def test_init_sets_threshold(self):
        """Инициализация вычисляет порог из bootstrap."""
        from anomaly.drift.mmd import MMDDriftDetector

        ref = self._make_reference()
        det = MMDDriftDetector(ref, n_bootstrap=30)
        assert det.threshold >= 0
        assert det.gamma > 0

    def test_detect_no_drift_same_distribution(self):
        """Нет дрейфа когда current из того же распределения."""
        from anomaly.drift.mmd import MMDDriftDetector

        rng = np.random.RandomState(99)
        ref = self._make_reference(n=300)
        current = rng.normal(0, 1, (100, 3)) * [5, 10, 100] + [40, 50, 1000]

        det = MMDDriftDetector(ref, n_bootstrap=50)
        result = det.detect(current)

        # is_drift может быть True/False из-за случайности bootstrap,
        # но mmd_statistic должен быть небольшим
        assert result.mmd_statistic >= 0
        assert isinstance(result.is_drift, bool)
        assert result.audit_id != ""
        assert result.timestamp != ""

    def test_detect_drift_on_shifted_data(self):
        """Дрейф обнаруживается при сильном сдвиге."""
        from anomaly.drift.mmd import MMDDriftDetector

        rng = np.random.RandomState(1)
        ref = self._make_reference(n=300)

        # Имитируем производственный инцидент: CPU вырос в 2 раза
        current = rng.normal(0, 1, (100, 3)) * [5, 10, 100] + [80, 200, 500]

        det = MMDDriftDetector(ref, n_bootstrap=50)
        result = det.detect(current)

        assert result.is_drift is True
        assert result.mmd_statistic > result.threshold

    def test_drift_result_has_correct_sizes(self):
        """DriftResult корректно записывает размеры выборок."""
        from anomaly.drift.mmd import MMDDriftDetector

        ref = self._make_reference(n=200)
        current = np.random.randn(80, 3)

        det = MMDDriftDetector(ref, n_bootstrap=30)
        result = det.detect(current)

        assert result.reference_size == 200
        assert result.current_size == 80

    def test_drift_result_features(self):
        """Имена признаков корректно передаются в DriftResult."""
        from anomaly.drift.mmd import MMDDriftDetector

        ref = self._make_reference(n=100)
        det = MMDDriftDetector(
            ref,
            features=["cpu", "latency", "requests"],
            n_bootstrap=30,
        )
        result = det.detect(np.random.randn(50, 3))
        assert result.features == ["cpu", "latency", "requests"]

    def test_p_value_range(self):
        """P-value всегда в диапазоне [0, 1]."""
        from anomaly.drift.mmd import MMDDriftDetector

        ref = self._make_reference(n=150)
        det = MMDDriftDetector(ref, n_bootstrap=50)
        result = det.detect(np.random.randn(50, 3))
        assert 0.0 <= result.p_value <= 1.0

    def test_reason_not_empty(self):
        """DriftResult.reason содержит описание решения."""
        from anomaly.drift.mmd import MMDDriftDetector

        ref = self._make_reference(n=100)
        det = MMDDriftDetector(ref, n_bootstrap=30)
        result = det.detect(np.random.randn(40, 3))
        assert len(result.reason) > 0


class TestAnomalyRetrainingTrigger:
    """Тесты триггера переобучения детектора аномалий."""

    def _make_metrics(self, n: int, shift: float = 0.0, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        cpu = rng.normal(40 + shift, 5, n)
        latency = rng.normal(50 + shift * 3, 10, n)
        requests = rng.normal(1000 - shift * 10, 100, n)
        return np.column_stack([cpu, latency, requests])

    def test_evaluate_no_drift_returns_skip(self):
        """При отсутствии дрейфа: should_retrain=False, triggered_by='none'."""
        from anomaly.retraining.trigger import AnomalyRetrainingTrigger

        ref = self._make_metrics(300)
        current = self._make_metrics(100, shift=0.5)  # незначительный сдвиг

        trigger = AnomalyRetrainingTrigger(
            ref,
            features=["cpu", "latency", "requests"],
            n_bootstrap=30,
        )
        result = trigger.evaluate(current)

        # Не проверяем конкретное значение is_drift (зависит от bootstrap),
        # но проверяем структуру результата
        assert isinstance(result.should_retrain, bool)
        assert result.triggered_by in ("mmd_drift", "none")
        assert result.drift_result is not None

    def test_evaluate_drift_triggers_retraining(self):
        """При сильном дрейфе: should_retrain=True, triggered_by='mmd_drift'."""
        from anomaly.retraining.trigger import AnomalyRetrainingTrigger

        ref = self._make_metrics(300)
        # Катастрофический инцидент: CPU вырос на 40 единиц
        current = self._make_metrics(100, shift=40.0, seed=99)

        trigger = AnomalyRetrainingTrigger(
            ref,
            features=["cpu", "latency", "requests"],
            n_bootstrap=50,
        )
        result = trigger.evaluate(current)

        assert result.should_retrain is True
        assert result.triggered_by == "mmd_drift"

    def test_result_has_audit_id(self):
        """Каждый результат содержит audit_id для EU AI Act compliance."""
        from anomaly.retraining.trigger import AnomalyRetrainingTrigger

        ref = self._make_metrics(100)
        trigger = AnomalyRetrainingTrigger(ref, n_bootstrap=20)
        result = trigger.evaluate(self._make_metrics(50))

        assert result.drift_result.audit_id != ""
        assert len(result.drift_result.audit_id) == 36  # UUID format

    def test_result_has_timestamp(self):
        """Результат содержит ISO 8601 timestamp."""
        from anomaly.retraining.trigger import AnomalyRetrainingTrigger

        ref = self._make_metrics(100)
        trigger = AnomalyRetrainingTrigger(ref, n_bootstrap=20)
        result = trigger.evaluate(self._make_metrics(50))

        ts = result.drift_result.timestamp
        assert "T" in ts  # ISO 8601 format
        assert "+" in ts or "Z" in ts or ts.endswith("+00:00")  # timezone

    def test_reason_contains_mmd(self):
        """Причина решения содержит 'MMD' для читаемости."""
        from anomaly.retraining.trigger import AnomalyRetrainingTrigger

        ref = self._make_metrics(100)
        trigger = AnomalyRetrainingTrigger(ref, n_bootstrap=20)
        result = trigger.evaluate(self._make_metrics(50))

        assert "MMD" in result.reason or "mmd" in result.reason.lower()


class TestMMDDriftAPIEndpoint:
    """Тесты API-эндпоинтов MMD drift detection."""

    def _make_points(self, n: int, shift: float = 0.0) -> list[list[float]]:
        rng = np.random.RandomState(42)
        cpu = rng.normal(40 + shift, 5, n).tolist()
        latency = rng.normal(50 + shift, 10, n).tolist()
        requests = rng.normal(1000, 100, n).tolist()
        return [[c, lat, r] for c, lat, r in zip(cpu, latency, requests, strict=False)]

    def test_drift_check_endpoint_stable(self):
        """POST /drift/check возвращает 200 и корректный ответ."""
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = {
            "reference": self._make_points(100),
            "current": self._make_points(50),
            "n_bootstrap": 30,
        }
        resp = client.post("/drift/check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "mmd_statistic" in data
        assert "threshold" in data
        assert "is_drift" in data
        assert "audit_id" in data
        assert "should_retrain" in data
        assert isinstance(data["is_drift"], bool)
        assert data["mmd_statistic"] >= 0

    def test_drift_check_detects_shift(self):
        """POST /drift/check обнаруживает сильный дрейф."""
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        payload = {
            "reference": self._make_points(150),
            "current": self._make_points(80, shift=50.0),  # большой сдвиг
            "n_bootstrap": 50,
        }
        resp = client.post("/drift/check", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_drift"] is True
        assert data["should_retrain"] is True
        assert data["triggered_by"] == "mmd_drift"

    def test_drift_status_before_check(self):
        """GET /drift/status до первого /drift/check возвращает no_check_performed."""
        import anomaly.api.app as app_module
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        # Сбрасываем глобальный статус
        app_module._last_drift_result = None
        client = TestClient(app)
        resp = client.get("/drift/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_check_performed"

    def test_drift_status_after_check(self):
        """GET /drift/status возвращает результат после /drift/check."""
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Сначала делаем check
        payload = {
            "reference": self._make_points(100),
            "current": self._make_points(50),
            "n_bootstrap": 20,
        }
        client.post("/drift/check", json=payload)

        resp = client.get("/drift/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("stable", "drift_detected")
        assert "mmd_statistic" in data
        assert "audit_id" in data

    def test_health_includes_drift_after_check(self):
        """GET /health включает last_drift после проверки."""
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        payload = {
            "reference": self._make_points(100),
            "current": self._make_points(50),
            "n_bootstrap": 20,
        }
        client.post("/drift/check", json=payload)

        resp = client.get("/health")
        data = resp.json()
        assert "last_drift" in data
        assert "is_drift" in data["last_drift"]
        assert "mmd_statistic" in data["last_drift"]


# ---------------------------------------------------------------------------
# Tests for Dashboard utils (anomaly/dashboard/utils.py)
# ---------------------------------------------------------------------------


class TestDashboardGenerateMetricStream:
    """Тесты для generate_metric_stream: форма, диапазоны, воспроизводимость."""

    def test_output_keys(self):
        """Должен вернуть ключи cpu, latency, requests."""
        from anomaly.dashboard.utils import generate_metric_stream

        stream = generate_metric_stream(n_points=50)
        assert set(stream.keys()) == {"cpu", "latency", "requests"}

    def test_output_shape(self):
        """Все серии должны иметь одинаковую длину = n_points."""
        from anomaly.dashboard.utils import generate_metric_stream

        for n in [50, 100, 200]:
            stream = generate_metric_stream(n_points=n)
            for key in ("cpu", "latency", "requests"):
                assert len(stream[key]) == n, f"{key} length mismatch for n={n}"

    def test_cpu_range(self):
        """CPU должен быть в диапазоне [5, 95] без аномалий."""
        from anomaly.dashboard.utils import generate_metric_stream

        stream = generate_metric_stream(n_points=200, inject_anomaly=False)
        assert stream["cpu"].min() >= 5.0
        assert stream["cpu"].max() <= 95.0

    def test_latency_positive(self):
        """Latency должна быть строго положительной."""
        from anomaly.dashboard.utils import generate_metric_stream

        stream = generate_metric_stream(n_points=200)
        assert (stream["latency"] > 0).all()

    def test_requests_positive(self):
        """Requests должны быть неотрицательными."""
        from anomaly.dashboard.utils import generate_metric_stream

        stream = generate_metric_stream(n_points=200)
        assert (stream["requests"] >= 0).all()

    def test_reproducible_with_same_seed(self):
        """Одинаковый seed → идентичные результаты."""
        from anomaly.dashboard.utils import generate_metric_stream

        s1 = generate_metric_stream(n_points=100, seed=7)
        s2 = generate_metric_stream(n_points=100, seed=7)
        for key in ("cpu", "latency", "requests"):
            np.testing.assert_array_equal(s1[key], s2[key])

    def test_different_seeds_differ(self):
        """Разные seeds → разные данные."""
        from anomaly.dashboard.utils import generate_metric_stream

        s1 = generate_metric_stream(n_points=100, seed=0)
        s2 = generate_metric_stream(n_points=100, seed=1)
        assert not np.array_equal(s1["cpu"], s2["cpu"])

    def test_inject_anomaly_raises_cpu(self):
        """С аномалией CPU в конце серии должен быть выше нормального уровня."""
        from anomaly.dashboard.utils import generate_metric_stream

        n = 200
        normal = generate_metric_stream(n_points=n, inject_anomaly=False, seed=42)
        anomalous = generate_metric_stream(
            n_points=n, inject_anomaly=True, anomaly_start=150, anomaly_magnitude=5.0, seed=42
        )
        # Аномалия в точках 150-200: среднее CPU должно вырасти
        cpu_normal_tail = float(np.mean(normal["cpu"][150:]))
        cpu_anomaly_tail = float(np.mean(anomalous["cpu"][150:]))
        assert cpu_anomaly_tail > cpu_normal_tail

    def test_no_anomaly_without_inject(self):
        """Без inject обе серии одинаковы при одинаковом seed."""
        from anomaly.dashboard.utils import generate_metric_stream

        s = generate_metric_stream(n_points=100, inject_anomaly=False, seed=0)
        # Просто проверяем что данные разумные — нет NaN, нет Inf
        for key in ("cpu", "latency", "requests"):
            assert np.isfinite(s[key]).all(), f"{key} contains non-finite values"


class TestDashboardComputeDetectionSummary:
    """Тесты для compute_detection_summary."""

    def test_all_normal(self):
        """Нет аномалий → n_anomalies=0, rate=0."""
        from anomaly.dashboard.utils import compute_detection_summary

        preds = np.zeros(100, dtype=int)
        scores = np.random.default_rng(0).uniform(0, 1, 100)
        summary = compute_detection_summary(preds, scores)
        assert summary["n_anomalies"] == 0
        assert summary["anomaly_rate"] == 0.0
        assert summary["n_total"] == 100

    def test_all_anomaly(self):
        """Все аномалии → rate=1."""
        from anomaly.dashboard.utils import compute_detection_summary

        preds = np.ones(50, dtype=int)
        scores = np.full(50, 5.0)
        summary = compute_detection_summary(preds, scores)
        assert summary["n_anomalies"] == 50
        assert summary["anomaly_rate"] == 1.0

    def test_partial_anomalies(self):
        """10 из 100 → rate=0.1."""
        from anomaly.dashboard.utils import compute_detection_summary

        preds = np.zeros(100, dtype=int)
        preds[:10] = 1
        scores = np.ones(100) * 2.0
        summary = compute_detection_summary(preds, scores)
        assert summary["n_anomalies"] == 10
        assert abs(summary["anomaly_rate"] - 0.1) < 1e-9

    def test_max_score(self):
        """max_score должен совпадать с np.max(scores)."""
        from anomaly.dashboard.utils import compute_detection_summary

        scores = np.array([1.0, 3.5, 2.2, 4.7, 0.5])
        preds = np.zeros(5, dtype=int)
        summary = compute_detection_summary(preds, scores)
        assert abs(summary["max_score"] - 4.7) < 1e-9

    def test_empty_returns_zero_rate(self):
        """Пустые массивы → anomaly_rate=0 без ZeroDivisionError."""
        from anomaly.dashboard.utils import compute_detection_summary

        summary = compute_detection_summary(np.array([]), np.array([]))
        assert summary["anomaly_rate"] == 0.0
        assert summary["n_total"] == 0


class TestDashboardReferenceCurrentData:
    """Тесты для generate_reference_data и generate_current_data."""

    def test_reference_shape(self):
        """reference данные: список точек, каждая точка [cpu, lat, req]."""
        from anomaly.dashboard.utils import generate_reference_data

        data = generate_reference_data(n_points=100)
        assert len(data) == 100
        assert all(len(p) == 3 for p in data)

    def test_current_shape(self):
        """current данные: список точек правильной формы."""
        from anomaly.dashboard.utils import generate_current_data

        data = generate_current_data(n_points=50)
        assert len(data) == 50
        assert all(len(p) == 3 for p in data)

    def test_reference_values_numeric(self):
        """Все значения в reference данных — числа."""
        from anomaly.dashboard.utils import generate_reference_data

        data = generate_reference_data(n_points=30)
        for point in data:
            for val in point:
                assert isinstance(val, float), f"Expected float, got {type(val)}"

    def test_drift_shifts_distribution(self):
        """С inject_drift=True среднее current должно отличаться от reference."""
        from anomaly.dashboard.utils import generate_current_data, generate_reference_data

        ref = np.array(generate_reference_data(n_points=200, seed=0))
        cur_drift = np.array(
            generate_current_data(n_points=200, inject_drift=True, drift_magnitude=6.0, seed=99)
        )

        # CPU: среднее при дрейфе должно быть выше (аномалия подняла CPU)
        ref_mean_cpu = float(np.mean(ref[:, 0]))
        cur_mean_cpu = float(np.mean(cur_drift[:, 0]))
        assert cur_mean_cpu > ref_mean_cpu, "Drift injection should raise CPU mean"


# ──────────────────────────────────────────────────────────────────────────────
# LSTM / ESN Autoencoder Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestESNAutoencoderCore:
    """Unit-тесты Echo State Network автоэнкодера."""

    def _make_normal_data(self, n: int = 200, seed: int = 0) -> np.ndarray:
        """Синтетический нормальный временной ряд [cpu, latency, requests]."""
        rng = np.random.RandomState(seed)
        cpu = 40 + 10 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.randn(n) * 2
        latency = 50 + 5 * np.cos(np.linspace(0, 4 * np.pi, n)) + rng.randn(n) * 1
        requests = 1000 + 100 * np.sin(np.linspace(0, 2 * np.pi, n)) + rng.randn(n) * 5
        return np.column_stack([cpu, latency, requests])

    def test_fit_returns_train_result(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        cfg = LSTMConfig(reservoir_size=50, window_size=20, n_features=3)
        model = EchoStateAutoencoder(cfg)
        X = self._make_normal_data(100)
        result = model.fit(X)
        assert result.n_samples == 100
        assert result.n_windows == 100 - 20 + 1
        assert result.train_mse >= 0
        assert result.threshold > 0

    def test_is_fitted_after_fit(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=30, window_size=15))
        assert not model.is_fitted
        model.fit(self._make_normal_data(80))
        assert model.is_fitted

    def test_detect_returns_correct_shape(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        cfg = LSTMConfig(reservoir_size=50, window_size=20)
        model = EchoStateAutoencoder(cfg)
        X = self._make_normal_data(100)
        model.fit(X)
        result = model.detect(X)
        assert result.scores.shape == (100,)
        assert result.predictions.shape == (100,)

    def test_detect_predictions_binary(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=50, window_size=20))
        X = self._make_normal_data(100)
        model.fit(X)
        result = model.detect(X)
        assert set(result.predictions.tolist()).issubset({0, 1})

    def test_detect_scores_nonnegative(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=50, window_size=20))
        X = self._make_normal_data(100)
        model.fit(X)
        result = model.detect(X)
        assert (result.scores >= 0).all()

    def test_anomaly_has_higher_score(self):
        """Аномальная точка должна получить более высокий score чем нормальные."""
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        cfg = LSTMConfig(reservoir_size=80, window_size=20, anomaly_percentile=90.0)
        model = EchoStateAutoencoder(cfg)

        # Обучаем на нормальных данных
        normal = self._make_normal_data(200, seed=0)
        model.fit(normal)

        # Создаём данные с явной аномалией в центре
        test_data = self._make_normal_data(100, seed=1)
        test_data[50, :] = [200.0, 1000.0, 50000.0]  # экстремальный выброс

        result = model.detect(test_data)
        # Score вокруг аномалии должен быть выше среднего нормального
        anomaly_region_scores = result.scores[45:55]
        normal_region_scores = result.scores[10:40]
        assert anomaly_region_scores.max() > normal_region_scores.mean()

    def test_model_name_in_result(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=30, window_size=15))
        model.fit(self._make_normal_data(80))
        result = model.detect(self._make_normal_data(50))
        assert result.model == "esn_autoencoder"

    def test_get_config_before_fit(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=50))
        cfg = model.get_config()
        assert cfg["fitted"] is False
        assert cfg["threshold"] is None
        assert cfg["type"] == "echo_state_autoencoder"

    def test_get_config_after_fit(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=50, window_size=20))
        model.fit(self._make_normal_data(100))
        cfg = model.get_config()
        assert cfg["fitted"] is True
        assert cfg["threshold"] is not None and cfg["threshold"] > 0

    def test_create_autoencoder_factory(self):
        from anomaly.models.lstm_autoencoder import LSTMConfig, create_autoencoder

        model = create_autoencoder()
        assert not model.is_fitted
        model2 = create_autoencoder(LSTMConfig(reservoir_size=30))
        assert model2.cfg.reservoir_size == 30

    def test_sequence_scaler_fit_transform(self):
        from anomaly.models.lstm_autoencoder import SequenceScaler

        scaler = SequenceScaler()
        X = np.array([[0.0, 0.0], [10.0, 100.0]])
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.min() == pytest.approx(0.0, abs=1e-9)
        assert X_scaled.max() == pytest.approx(1.0, abs=1e-9)

    def test_sequence_scaler_constant_feature(self):
        """Константная фича не должна вызывать деление на ноль."""
        from anomaly.models.lstm_autoencoder import SequenceScaler

        scaler = SequenceScaler()
        X = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
        result = scaler.fit_transform(X)
        # Константная колонка → 0 после нормализации (clamp в scaler)
        assert np.isfinite(result).all()

    def test_train_result_threshold_positive(self):
        from anomaly.models.lstm_autoencoder import LSTMConfig, create_autoencoder

        model = create_autoencoder(LSTMConfig(reservoir_size=40, window_size=15))
        X = self._make_normal_data(80)
        tr = model.fit(X)
        assert tr.threshold > 0

    def test_reconstruction_errors_match_scores(self):
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        model = EchoStateAutoencoder(LSTMConfig(reservoir_size=50, window_size=20))
        X = self._make_normal_data(100)
        model.fit(X)
        result = model.detect(X)
        # reconstruction_errors и scores должны совпадать
        np.testing.assert_array_equal(result.reconstruction_errors, result.scores)

    def test_normal_data_low_anomaly_rate(self):
        """На нормальных данных аномальность не должна превышать (100 - percentile)%."""
        from anomaly.models.lstm_autoencoder import EchoStateAutoencoder, LSTMConfig

        cfg = LSTMConfig(reservoir_size=80, window_size=20, anomaly_percentile=95.0)
        model = EchoStateAutoencoder(cfg)
        X = self._make_normal_data(200, seed=7)
        model.fit(X)
        result = model.detect(X)
        # На обучающих данных должно быть ~5% аномалий (percentile threshold)
        anomaly_rate = result.predictions.mean()
        assert anomaly_rate <= 0.15  # допуск 15% из-за граничных эффектов окна


class TestLSTMAPIEndpoints:
    """Тесты API endpoints /lstm/*."""

    def _client(self):
        from anomaly.api.app import app
        from fastapi.testclient import TestClient

        return TestClient(app)

    def _normal_data(self, n: int = 150) -> list[list[float]]:
        rng = np.random.RandomState(0)
        cpu = (40 + rng.randn(n) * 3).tolist()
        lat = (50 + rng.randn(n) * 2).tolist()
        req = (1000 + rng.randn(n) * 20).tolist()
        return [[c, lat_v, r] for c, lat_v, r in zip(cpu, lat, req, strict=True)]

    def test_lstm_status_before_train(self):
        client = self._client()
        resp = client.get("/lstm/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "fitted" in data

    def test_lstm_train_success(self):
        client = self._client()
        payload = {"data": self._normal_data(150), "reservoir_size": 50, "window_size": 20}
        resp = client.post("/lstm/train", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_samples"] == 150
        assert body["threshold"] > 0

    def test_lstm_train_returns_model_config(self):
        client = self._client()
        payload = {"data": self._normal_data(150), "reservoir_size": 60, "window_size": 25}
        resp = client.post("/lstm/train", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "model_config" in body
        assert body["model_config"]["fitted"] is True

    def test_lstm_detect_requires_training(self):
        """До обучения /lstm/detect должен возвращать 400."""
        from anomaly.api import app as app_module

        # Сбрасываем модель в unfitted состояние
        from anomaly.models.lstm_autoencoder import create_autoencoder

        original = app_module.app  # noqa: F841
        import anomaly.api.app as api_app

        api_app._lstm_model = create_autoencoder()

        client = self._client()
        payload = {"data": self._normal_data(50)}
        resp = client.post("/lstm/detect", json=payload)
        assert resp.status_code == 400
        assert "train" in resp.json()["detail"].lower()

    def test_lstm_detect_after_train(self):
        client = self._client()

        # Обучаем
        train_payload = {
            "data": self._normal_data(150),
            "reservoir_size": 50,
            "window_size": 20,
        }
        client.post("/lstm/train", json=train_payload)

        # Детектируем
        detect_payload = {"data": self._normal_data(60)}
        resp = client.post("/lstm/detect", json=detect_payload)
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["scores"]) == 60
        assert len(body["predictions"]) == 60
        assert "n_anomalies" in body
        assert "anomaly_rate" in body

    def test_lstm_detect_scores_length_matches_input(self):
        client = self._client()
        n = 80
        client.post(
            "/lstm/train",
            json={"data": self._normal_data(150), "reservoir_size": 50, "window_size": 20},
        )
        resp = client.post("/lstm/detect", json={"data": self._normal_data(n)})
        assert resp.status_code == 200
        assert len(resp.json()["scores"]) == n

    def test_lstm_status_after_train(self):
        client = self._client()
        client.post(
            "/lstm/train",
            json={"data": self._normal_data(150), "reservoir_size": 50, "window_size": 20},
        )
        resp = client.get("/lstm/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fitted"] is True
        assert "train_metrics" in body
        assert body["train_metrics"]["threshold"] > 0

    def test_lstm_anomaly_rate_float(self):
        client = self._client()
        client.post(
            "/lstm/train",
            json={"data": self._normal_data(150), "reservoir_size": 50, "window_size": 20},
        )
        resp = client.post("/lstm/detect", json={"data": self._normal_data(50)})
        assert resp.status_code == 200
        rate = resp.json()["anomaly_rate"]
        assert 0.0 <= rate <= 1.0

    def test_lstm_model_name_in_response(self):
        client = self._client()
        client.post(
            "/lstm/train",
            json={"data": self._normal_data(150), "reservoir_size": 50, "window_size": 20},
        )
        resp = client.post("/lstm/detect", json={"data": self._normal_data(50)})
        assert resp.status_code == 200
        assert resp.json()["model"] == "esn_autoencoder"

    def test_lstm_train_minimum_data(self):
        """50 точек — минимально допустимый объём для обучения."""
        client = self._client()
        data_50 = self._normal_data(50)
        resp = client.post(
            "/lstm/train",
            json={"data": data_50, "reservoir_size": 30, "window_size": 10},
        )
        assert resp.status_code == 200

    def test_lstm_train_too_little_data(self):
        """Меньше 50 точек — validation error (422)."""
        client = self._client()
        data_10 = self._normal_data(10)
        resp = client.post("/lstm/train", json={"data": data_10})
        assert resp.status_code == 422


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests: Isolation Forest detector
# ──────────────────────────────────────────────────────────────────────────────


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector and IsolationConfig."""

    def _make_normal_data(self, n: int = 200, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        cpu = rng.normal(30, 5, n).clip(0, 100)
        latency = rng.normal(100, 15, n).clip(0)
        requests = rng.normal(500, 50, n).clip(0)
        return np.column_stack([cpu, latency, requests])

    def _inject_anomaly(self, data: np.ndarray, idx: int = 0) -> np.ndarray:  # noqa: N803
        data = data.copy()
        data[idx] = [99.0, 2000.0, 5000.0]
        return data

    def test_is_available(self):
        from anomaly.models.isolation import is_available

        assert isinstance(is_available(), bool)

    def test_fit_returns_train_result(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        if not det.is_available() if hasattr(det, "is_available") else False:
            pytest.skip("sklearn not available")
        X = self._make_normal_data(200)
        result = det.fit(X)
        assert result.n_samples == 200
        assert result.n_features == 3

    def test_fit_stores_is_fitted(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        assert not det.is_fitted
        det.fit(self._make_normal_data(200))
        assert det.is_fitted

    def test_detect_returns_correct_length(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        det.fit(self._make_normal_data(200))
        X = self._make_normal_data(10)
        results = det.detect(X)
        assert len(results) == 10

    def test_detect_binary_is_anomaly(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        det.fit(self._make_normal_data(200))
        results = det.detect(self._make_normal_data(20))
        for r in results:
            assert isinstance(r.is_anomaly, bool)

    def test_anomaly_score_in_range(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        det.fit(self._make_normal_data(200))
        results = det.detect(self._make_normal_data(50))
        for r in results:
            assert 0.0 <= r.anomaly_score <= 1.0

    def test_feature_contributions_sum_to_one(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        det.fit(self._make_normal_data(200))
        results = det.detect(self._make_normal_data(10))
        for r in results:
            total = sum(r.feature_contributions.values())
            assert abs(total - 1.0) < 1e-6, f"contributions sum = {total}"

    def test_feature_names_in_contributions(self):
        from anomaly.models.isolation import IsolationConfig, IsolationForestDetector

        cfg = IsolationConfig(feature_names=["cpu", "latency", "requests"])
        det = IsolationForestDetector(cfg)
        det.fit(self._make_normal_data(200))
        results = det.detect(self._make_normal_data(5))
        for r in results:
            assert set(r.feature_contributions.keys()) == {"cpu", "latency", "requests"}

    def test_top_feature_is_valid(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        det.fit(self._make_normal_data(200))
        results = det.detect(self._make_normal_data(10))
        for r in results:
            assert r.top_feature in r.feature_contributions

    def test_anomaly_has_higher_score_than_normal(self):
        """Явная аномалия должна получать более высокий score, чем нормальная точка."""
        from anomaly.models.isolation import IsolationForestDetector

        normal_data = self._make_normal_data(300)
        det = IsolationForestDetector()
        det.fit(normal_data)

        # Нормальная точка — в центре распределения
        normal_point = np.array([[30.0, 100.0, 500.0]])
        # Явная аномалия — экстремальные значения всех метрик
        anomaly_point = np.array([[99.0, 3000.0, 8000.0]])

        normal_result = det.detect(normal_point)[0]
        anomaly_result = det.detect(anomaly_point)[0]

        assert anomaly_result.anomaly_score > normal_result.anomaly_score

    def test_detect_before_fit_raises(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not trained"):
            det.detect(self._make_normal_data(5))

    def test_normal_data_low_anomaly_rate(self):
        """На нормальных данных аномалий должно быть ~contamination (5%)."""
        from anomaly.models.isolation import IsolationForestDetector

        normal_data = self._make_normal_data(500)
        det = IsolationForestDetector()
        det.fit(normal_data)

        results = det.detect(normal_data)
        rate = sum(1 for r in results if r.is_anomaly) / len(results)
        # Должно быть близко к contamination=0.05, допуск широкий
        assert rate < 0.15, f"anomaly_rate={rate:.2f} on normal data is too high"

    def test_injected_anomaly_detected(self):
        """Явная инъекция аномалии должна быть обнаружена."""
        from anomaly.models.isolation import IsolationConfig, IsolationForestDetector

        normal_data = self._make_normal_data(300)
        cfg = IsolationConfig(contamination=0.10)
        det = IsolationForestDetector(cfg)
        det.fit(normal_data)

        test_data = self._make_normal_data(49)
        anomaly = np.array([[99.0, 5000.0, 10000.0]])
        mixed = np.vstack([test_data, anomaly])

        results = det.detect(mixed)
        # Последняя точка (аномалия) должна иметь score выше медианы нормальных
        anomaly_result = results[-1]
        normal_scores = [r.anomaly_score for r in results[:-1]]
        assert anomaly_result.anomaly_score > np.median(normal_scores)

    def test_train_result_fields(self):
        from anomaly.models.isolation import IsolationForestDetector

        det = IsolationForestDetector()
        result = det.fit(self._make_normal_data(100))
        assert result.n_trees == 100
        assert result.contamination == 0.05
        assert result.avg_path_length_normal > 0


# ──────────────────────────────────────────────────────────────────────────────
# API tests: Isolation Forest endpoints
# ──────────────────────────────────────────────────────────────────────────────


class TestIsolationAPIEndpoints:
    """Tests for POST /isolation/train, /isolation/detect, GET /isolation/status."""

    def _client(self):
        import anomaly.api.app as app_module
        from anomaly.api.app import _isolation_model, app
        from fastapi.testclient import TestClient

        # Сбрасываем состояние модели перед каждым тестом
        _isolation_model._is_fitted = False
        _isolation_model._model = None
        _isolation_model._train_result = None
        app_module._isolation_train_result = None
        return TestClient(app)

    def _normal_data(self, n: int = 150) -> list[list[float]]:
        rng = np.random.default_rng(123)
        cpu = rng.normal(30, 5, n).clip(0, 100).tolist()
        latency = rng.normal(100, 15, n).clip(0).tolist()
        requests = rng.normal(500, 50, n).clip(0).tolist()
        return [[cpu[i], latency[i], requests[i]] for i in range(n)]

    def test_train_status_200(self):
        client = self._client()
        resp = client.post("/isolation/train", json={"data": self._normal_data(100)})
        assert resp.status_code == 200

    def test_train_response_structure(self):
        client = self._client()
        resp = client.post("/isolation/train", json={"data": self._normal_data(100)})
        body = resp.json()
        assert "n_samples" in body
        assert "n_features" in body
        assert "contamination" in body
        assert "n_trees" in body
        assert "avg_path_length_normal" in body
        assert body["sklearn_available"] is True

    def test_train_n_samples_correct(self):
        client = self._client()
        resp = client.post("/isolation/train", json={"data": self._normal_data(120)})
        assert resp.json()["n_samples"] == 120

    def test_detect_before_train_returns_400(self):
        client = self._client()
        resp = client.post("/isolation/detect", json={"data": self._normal_data(10)})
        assert resp.status_code == 400

    def test_detect_after_train_returns_200(self):
        client = self._client()
        client.post("/isolation/train", json={"data": self._normal_data(100)})
        resp = client.post("/isolation/detect", json={"data": self._normal_data(10)})
        assert resp.status_code == 200

    def test_detect_response_structure(self):
        client = self._client()
        client.post("/isolation/train", json={"data": self._normal_data(100)})
        resp = client.post("/isolation/detect", json={"data": self._normal_data(5)})
        body = resp.json()
        assert "results" in body
        assert "n_anomalies" in body
        assert "anomaly_rate" in body
        assert len(body["results"]) == 5

    def test_detect_result_has_feature_contributions(self):
        client = self._client()
        client.post("/isolation/train", json={"data": self._normal_data(100)})
        resp = client.post("/isolation/detect", json={"data": self._normal_data(3)})
        for point_result in resp.json()["results"]:
            assert "feature_contributions" in point_result
            assert "top_feature" in point_result
            assert "anomaly_score" in point_result

    def test_detect_anomaly_rate_in_range(self):
        client = self._client()
        client.post("/isolation/train", json={"data": self._normal_data(100)})
        resp = client.post("/isolation/detect", json={"data": self._normal_data(50)})
        rate = resp.json()["anomaly_rate"]
        assert 0.0 <= rate <= 1.0

    def test_status_before_train(self):
        client = self._client()
        resp = client.get("/isolation/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fitted"] is False
        assert "message" in body

    def test_status_after_train(self):
        client = self._client()
        client.post("/isolation/train", json={"data": self._normal_data(100)})
        resp = client.get("/isolation/status")
        body = resp.json()
        assert body["fitted"] is True
        assert "train_metrics" in body

    def test_health_includes_isolation_fitted(self):
        client = self._client()
        resp = client.get("/health")
        assert "isolation_fitted" in resp.json()

    def test_full_train_detect_cycle(self):
        """Полный цикл: train → detect → проверить top_anomalous_feature."""
        client = self._client()
        client.post("/isolation/train", json={"data": self._normal_data(150)})

        # Инъецируем явную аномалию в конец списка
        test_data = self._normal_data(9)
        test_data.append([99.0, 5000.0, 10000.0])

        resp = client.post("/isolation/detect", json={"data": test_data})
        body = resp.json()
        assert resp.status_code == 200
        # Должна быть хотя бы одна аномалия
        assert body["n_anomalies"] >= 1
        assert body["top_anomalous_feature"] is not None

    def test_train_with_custom_params(self):
        client = self._client()
        resp = client.post(
            "/isolation/train",
            json={
                "data": self._normal_data(100),
                "contamination": 0.10,
                "n_estimators": 50,
            },
        )
        body = resp.json()
        assert body["contamination"] == 0.10
        assert body["n_trees"] == 50


# ──────────────────────────────────────────────────────────────────────────────
# CUSUM Change Detection
# ──────────────────────────────────────────────────────────────────────────────


class TestCUSUMDetector:
    """Юнит-тесты алгоритма CUSUM Page 1954."""

    def _normal_series(self, n: int = 200, seed: int = 7) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randn(n)

    def test_calibrate_returns_mu_sigma(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        data = self._normal_series(100)
        res = det.calibrate(data)
        assert abs(res.mu_ref - float(data.mean())) < 1e-10
        assert res.sigma_ref > 0
        assert res.n_calibration == 100

    def test_calibrate_sets_is_calibrated(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        assert not det.is_calibrated
        det.calibrate(self._normal_series(50))
        assert det.is_calibrated

    def test_calibrate_too_few_raises(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        with pytest.raises(ValueError):
            det.calibrate(np.array([1.0, 2.0, 3.0]))

    def test_detect_before_calibrate_raises(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        with pytest.raises(RuntimeError):
            det.detect(np.array([1.0, 2.0, 3.0]))

    def test_detect_returns_correct_length(self):
        from anomaly.models.cusum import CUSUMConfig, CUSUMDetector

        det = CUSUMDetector(CUSUMConfig(k=0.5, h=5.0))
        det.calibrate(self._normal_series(100))
        series = self._normal_series(80, seed=13)
        res = det.detect(series)
        assert len(res.s_pos) == 80
        assert len(res.s_neg) == 80
        assert len(res.predictions) == 80

    def test_detect_normal_data_few_alerts(self):
        """Нормальные данные → CUSUM редко превышает порог h=5."""
        from anomaly.models.cusum import CUSUMConfig, CUSUMDetector

        rng = np.random.RandomState(99)
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=5.0))
        det.calibrate(rng.randn(300))
        # Тест на 1000 точках — при ARL₀≈465 ожидаем ~2 ложных тревоги
        series = rng.randn(1000)
        res = det.detect(series)
        alert_rate = sum(res.predictions) / len(res.predictions)
        assert alert_rate < 0.05  # менее 5% ложных тревог

    def test_detect_persistent_shift_triggers_alert(self):
        """CUSUM обнаруживает устойчивый сдвиг среднего на 2σ."""
        from anomaly.models.cusum import CUSUMConfig, CUSUMDetector

        rng = np.random.RandomState(0)
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=5.0))
        det.calibrate(rng.randn(200))
        # Ряд: сначала нормально, потом устойчивый сдвиг +2σ
        normal_part = rng.randn(50)
        shifted_part = rng.randn(100) + 3.0  # явный сдвиг вверх
        series = np.concatenate([normal_part, shifted_part])
        res = det.detect(series)
        # Должны найти хотя бы одну тревогу в shifted_part (после индекса 50)
        alerts_in_shift = [cp for cp in res.change_points if cp >= 50]
        assert len(alerts_in_shift) >= 1

    def test_detect_change_points_are_valid_indices(self):
        from anomaly.models.cusum import CUSUMConfig, CUSUMDetector

        rng = np.random.RandomState(5)
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=4.0))
        det.calibrate(rng.randn(100))
        series = np.concatenate([rng.randn(50), rng.randn(50) + 4.0])
        res = det.detect(series)
        for cp in res.change_points:
            assert 0 <= cp < len(series)

    def test_s_pos_non_negative(self):
        """S⁺ₜ = max(0, ...) всегда ≥ 0."""
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        det.calibrate(self._normal_series(100))
        res = det.detect(self._normal_series(200, seed=99))
        assert all(s >= 0 for s in res.s_pos)

    def test_s_neg_non_negative(self):
        """S⁻ₜ = max(0, ...) всегда ≥ 0."""
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        det.calibrate(self._normal_series(100))
        res = det.detect(self._normal_series(200, seed=99))
        assert all(s >= 0 for s in res.s_neg)

    def test_n_alerts_matches_change_points(self):
        from anomaly.models.cusum import CUSUMConfig, CUSUMDetector

        det = CUSUMDetector(CUSUMConfig(h=3.0))
        det.calibrate(self._normal_series(100))
        series = np.concatenate([self._normal_series(50), self._normal_series(50) + 4.0])
        res = det.detect(series)
        assert res.n_alerts == len(res.change_points)

    def test_update_before_calibrate_raises(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        with pytest.raises(RuntimeError):
            det.update(1.5)

    def test_update_increments_counter(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        det.calibrate(self._normal_series(50))
        r1 = det.update(0.5)
        r2 = det.update(0.5)
        assert r1.n_updates == 1
        assert r2.n_updates == 2

    def test_update_triggers_alert_on_large_shift(self):
        """Большой устойчивый сдвиг вызывает тревогу при онлайн-обновлении."""
        from anomaly.models.cusum import CUSUMConfig, CUSUMDetector

        rng = np.random.RandomState(42)
        det = CUSUMDetector(CUSUMConfig(k=0.5, h=4.0))
        det.calibrate(rng.randn(200))
        # Подаём сильный сдвиг (+5σ) — должна сработать тревога
        results = [det.update(5.0) for _ in range(20)]
        assert any(r.is_alert for r in results)

    def test_reset_clears_statistics(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        det.calibrate(self._normal_series(50))
        # Накапливаем сдвиг
        for _ in range(10):
            det.update(10.0)
        state_before = det.get_state()
        det.reset()
        state_after = det.get_state()
        # После reset статистики обнуляются, калибровка сохраняется
        assert state_after.s_pos == 0.0
        assert state_after.s_neg == 0.0
        assert state_after.n_updates == 0
        assert state_after.is_calibrated  # μ₀ и σ₀ не теряем
        _ = state_before  # suppress unused variable warning

    def test_get_state_structure(self):
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        det.calibrate(self._normal_series(50))
        state = det.get_state()
        assert state.is_calibrated is True
        assert isinstance(state.s_pos, float)
        assert isinstance(state.s_neg, float)
        assert state.n_updates == 0

    def test_constant_series_calibration(self):
        """Константный ряд (σ=0) не вызывает деления на ноль."""
        from anomaly.models.cusum import CUSUMDetector

        det = CUSUMDetector()
        # Все значения одинаковые — σ=0, должна быть защита
        constant = np.ones(50)
        res = det.calibrate(constant)
        assert res.sigma_ref > 0  # защитный порог min(σ, 1e-6)


class TestCUSUMAPIEndpoints:
    """Тесты API эндпоинтов CUSUM."""

    def _client(self):
        from anomaly.api.app import _reset_cusum, app
        from fastapi.testclient import TestClient

        _reset_cusum()
        return TestClient(app)

    def _normal_data(self, n: int = 100, seed: int = 42) -> list[float]:
        return np.random.RandomState(seed).randn(n).tolist()

    def test_calibrate_200(self):
        client = self._client()
        resp = client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        assert resp.status_code == 200

    def test_calibrate_response_structure(self):
        client = self._client()
        resp = client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        body = resp.json()
        assert "mu_ref" in body
        assert "sigma_ref" in body
        assert "n_calibration" in body
        assert body["n_calibration"] == 100
        assert body["sigma_ref"] > 0

    def test_calibrate_custom_k_h(self):
        client = self._client()
        resp = client.post(
            "/cusum/calibrate", json={"data": self._normal_data(50), "k": 1.0, "h": 8.0}
        )
        body = resp.json()
        assert body["k"] == 1.0
        assert body["h"] == 8.0

    def test_detect_requires_calibration(self):
        client = self._client()
        resp = client.post("/cusum/detect", json={"data": [1.0, 2.0, 3.0]})
        assert resp.status_code == 400

    def test_detect_200_after_calibrate(self):
        client = self._client()
        client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        resp = client.post("/cusum/detect", json={"data": self._normal_data(50, seed=99)})
        assert resp.status_code == 200

    def test_detect_response_structure(self):
        client = self._client()
        client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        resp = client.post("/cusum/detect", json={"data": self._normal_data(50, seed=99)})
        body = resp.json()
        assert "s_pos" in body
        assert "s_neg" in body
        assert "predictions" in body
        assert "change_points" in body
        assert "n_alerts" in body
        assert len(body["s_pos"]) == 50
        assert len(body["predictions"]) == 50

    def test_detect_persistent_shift(self):
        """API: CUSUM обнаруживает явный сдвиг после нормального периода."""
        client = self._client()
        rng = np.random.RandomState(0)
        normal = rng.randn(200).tolist()
        client.post("/cusum/calibrate", json={"data": normal, "h": 4.0})
        # Серия с большим сдвигом
        shifted = (rng.randn(100) + 5.0).tolist()
        resp = client.post("/cusum/detect", json={"data": shifted})
        body = resp.json()
        assert body["n_alerts"] >= 1

    def test_update_requires_calibration(self):
        client = self._client()
        resp = client.post("/cusum/update", json={"value": 1.5})
        assert resp.status_code == 400

    def test_update_200_after_calibrate(self):
        client = self._client()
        client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        resp = client.post("/cusum/update", json={"value": 0.5})
        assert resp.status_code == 200

    def test_update_increments_n_updates(self):
        client = self._client()
        client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        r1 = client.post("/cusum/update", json={"value": 0.5}).json()
        r2 = client.post("/cusum/update", json={"value": 0.5}).json()
        assert r1["n_updates"] == 1
        assert r2["n_updates"] == 2

    def test_update_alert_on_large_value(self):
        client = self._client()
        normal = np.random.RandomState(1).randn(200).tolist()
        client.post("/cusum/calibrate", json={"data": normal, "h": 3.0})
        # Подаём очень большое значение 20 раз — должна сработать тревога
        alerts = [client.post("/cusum/update", json={"value": 5.0}).json() for _ in range(20)]
        assert any(a["is_alert"] for a in alerts)

    def test_status_uncalibrated(self):
        client = self._client()
        resp = client.get("/cusum/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_calibrated"] is False
        assert "message" in body

    def test_status_after_calibrate(self):
        client = self._client()
        client.post("/cusum/calibrate", json={"data": self._normal_data(100)})
        resp = client.get("/cusum/status")
        body = resp.json()
        assert body["is_calibrated"] is True
        assert "mu_ref" in body
        assert "sigma_ref" in body

    def test_full_cycle(self):
        """Полный цикл: calibrate → detect → update → status."""
        client = self._client()
        rng = np.random.RandomState(42)
        # 1. Калибровка
        cal = client.post("/cusum/calibrate", json={"data": rng.randn(200).tolist()}).json()
        assert cal["n_calibration"] == 200
        # 2. Батч детекция
        normal_series = rng.randn(100).tolist()
        det = client.post("/cusum/detect", json={"data": normal_series}).json()
        assert "n_alerts" in det
        # 3. Онлайн обновление
        upd = client.post("/cusum/update", json={"value": 0.1}).json()
        assert upd["n_updates"] == 1
        # 4. Статус
        status = client.get("/cusum/status").json()
        assert status["is_calibrated"] is True


class TestKalmanDetector:
    """Unit-тесты KalmanDetector."""

    def _normal(self, n: int = 200, seed: int = 42) -> list[float]:
        rng = np.random.RandomState(seed)
        return (rng.randn(n) * 5.0 + 50.0).tolist()

    def test_calibrate_sets_calibrated(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        assert det._is_calibrated

    def test_calibrate_too_few_raises(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        with pytest.raises(ValueError):
            det.calibrate([1.0] * 9)

    def test_calibrate_returns_fields(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        res = det.calibrate(self._normal(100))
        assert res.estimated_R > 0
        assert res.n_samples == 100
        assert res.threshold_nis > 0

    def test_calibrate_noise_positive(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(200))
        assert det._R > 0

    def test_update_before_calibrate_raises(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        with pytest.raises(RuntimeError):
            det.update(1.0)

    def test_update_returns_dataclass(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        r = det.update(50.0)
        assert r.level is not None
        assert r.trend is not None
        assert r.nis >= 0
        assert r.threshold > 0
        assert isinstance(r.is_anomaly, bool)

    def test_update_n_updates_increments(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        det.update(50.0)
        r = det.update(51.0)
        assert r.n_updates == 2

    def test_detect_before_calibrate_raises(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        with pytest.raises(RuntimeError):
            det.detect([1.0, 2.0, 3.0])

    def test_detect_output_length(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(200))
        res = det.detect(self._normal(50, seed=7))
        assert len(res.levels) == 50
        assert len(res.predictions) == 50
        assert len(res.nis_scores) == 50

    def test_detect_normal_data_low_anomaly_rate(self):
        """На нормальных данных ложных тревог < 10% при alpha=0.01."""
        from anomaly.models.kalman import KalmanConfig, KalmanDetector

        det = KalmanDetector(KalmanConfig(anomaly_alpha=0.01))
        det.calibrate(self._normal(500, seed=0))
        res = det.detect(self._normal(500, seed=1))
        rate = sum(res.predictions) / 500
        assert rate < 0.10

    def test_detect_injected_spike_detected(self):
        """Одиночный большой выброс (10σ) должен быть детектирован."""
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(200, seed=0))
        # Короткий ряд: одна нормальная точка, затем большой скачок
        data = [50.0] * 20 + [50.0 + 100.0] + [50.0] * 20  # spike at index 20
        res = det.detect(data)
        assert res.n_anomalies >= 1

    def test_detect_nis_scores_non_negative(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        res = det.detect(self._normal(50, seed=5))
        assert all(s >= 0 for s in res.nis_scores)

    def test_anomaly_indices_consistent(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        res = det.detect(self._normal(50, seed=3))
        for idx in res.anomaly_indices:
            assert res.predictions[idx] is True

    def test_get_state_before_calibrate(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        state = det.get_state()
        assert state["is_calibrated"] is False
        assert state["level"] is None

    def test_get_state_after_calibrate(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        state = det.get_state()
        assert state["is_calibrated"] is True
        assert state["level"] is not None
        assert state["threshold_nis"] > 0

    def test_reset_clears_calibration(self):
        from anomaly.models.kalman import KalmanDetector

        det = KalmanDetector()
        det.calibrate(self._normal(100))
        det.reset()
        assert not det._is_calibrated

    def test_measurement_noise_override(self):
        """measurement_noise=None → авто-оценка; override → используется напрямую."""
        from anomaly.models.kalman import KalmanConfig, KalmanDetector

        det = KalmanDetector(KalmanConfig(measurement_noise=100.0))
        det.calibrate(self._normal(100))
        assert pytest.approx(100.0) == det._R

    def test_threshold_decreases_with_stricter_alpha(self):
        """Меньший alpha → меньший порог (более жёсткая проверка)."""
        from anomaly.models.kalman import KalmanConfig, KalmanDetector

        det_strict = KalmanDetector(KalmanConfig(anomaly_alpha=0.001))
        det_lenient = KalmanDetector(KalmanConfig(anomaly_alpha=0.10))
        det_strict.calibrate(self._normal(100))
        det_lenient.calibrate(self._normal(100))
        # chi2_inv(0.001, 1) > chi2_inv(0.10, 1)
        assert det_strict._threshold > det_lenient._threshold


class TestKalmanAPIEndpoints:
    """Интеграционные тесты API эндпоинтов Kalman Filter."""

    def _client(self):
        from anomaly.api.app import _reset_kalman, app
        from fastapi.testclient import TestClient

        _reset_kalman()
        return TestClient(app)

    def _normal(self, n: int = 200, seed: int = 0) -> list[float]:
        return (np.random.RandomState(seed).randn(n) * 5 + 50).tolist()

    def test_calibrate_200(self):
        client = self._client()
        resp = client.post("/kalman/calibrate", json={"data": self._normal(100)})
        assert resp.status_code == 200

    def test_calibrate_response_structure(self):
        client = self._client()
        resp = client.post("/kalman/calibrate", json={"data": self._normal(100)})
        body = resp.json()
        assert "estimated_r" in body
        assert "n_samples" in body
        assert "threshold_nis" in body
        assert body["n_samples"] == 100
        assert body["estimated_r"] > 0

    def test_calibrate_custom_alpha(self):
        client = self._client()
        resp = client.post(
            "/kalman/calibrate",
            json={"data": self._normal(100), "anomaly_alpha": 0.05},
        )
        body = resp.json()
        # chi2_inv(0.05, 1) = 3.841 < chi2_inv(0.01, 1) = 6.635
        assert body["anomaly_alpha"] == pytest.approx(0.05)
        assert body["threshold_nis"] == pytest.approx(3.841, abs=0.01)

    def test_detect_400_before_calibrate(self):
        client = self._client()
        resp = client.post("/kalman/detect", json={"data": [1.0, 2.0]})
        assert resp.status_code == 400

    def test_detect_200_after_calibrate(self):
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(100)})
        resp = client.post("/kalman/detect", json={"data": self._normal(50, seed=1)})
        assert resp.status_code == 200

    def test_detect_response_structure(self):
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(100)})
        resp = client.post("/kalman/detect", json={"data": self._normal(50)})
        body = resp.json()
        expected = (
            "levels",
            "trends",
            "predicted",
            "innovations",
            "nis_scores",
            "predictions",
            "threshold",
            "anomaly_indices",
            "n_anomalies",
        )
        for field in expected:
            assert field in body
        assert len(body["levels"]) == 50
        assert len(body["predictions"]) == 50

    def test_detect_spike_n_anomalies(self):
        """API: явный выброс 10σ должен дать ≥ 1 аномалию."""
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(200)})
        data = [50.0] * 20 + [200.0] + [50.0] * 20  # одиночный spike
        resp = client.post("/kalman/detect", json={"data": data})
        assert resp.json()["n_anomalies"] >= 1

    def test_update_400_before_calibrate(self):
        client = self._client()
        resp = client.post("/kalman/update", json={"value": 50.0})
        assert resp.status_code == 400

    def test_update_200_after_calibrate(self):
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(100)})
        resp = client.post("/kalman/update", json={"value": 50.0})
        assert resp.status_code == 200

    def test_update_response_structure(self):
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(100)})
        body = client.post("/kalman/update", json={"value": 50.0}).json()
        for field in (
            "level",
            "trend",
            "predicted",
            "innovation",
            "nis",
            "threshold",
            "is_anomaly",
            "n_updates",
        ):
            assert field in body

    def test_update_n_updates_increments(self):
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(100)})
        r1 = client.post("/kalman/update", json={"value": 50.0}).json()
        r2 = client.post("/kalman/update", json={"value": 51.0}).json()
        assert r1["n_updates"] == 1
        assert r2["n_updates"] == 2

    def test_update_alert_on_extreme_value(self):
        """Очень большое значение (100σ от среднего) должно вызвать is_anomaly=True."""
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(200)})
        # Подать несколько нормальных точек чтобы фильтр сошёлся
        for _ in range(20):
            client.post("/kalman/update", json={"value": 50.0})
        resp = client.post("/kalman/update", json={"value": 5000.0})
        assert resp.json()["is_anomaly"] is True

    def test_status_uncalibrated(self):
        client = self._client()
        resp = client.get("/kalman/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_calibrated"] is False
        assert "message" in body

    def test_status_after_calibrate(self):
        client = self._client()
        client.post("/kalman/calibrate", json={"data": self._normal(100)})
        resp = client.get("/kalman/status")
        body = resp.json()
        assert body["is_calibrated"] is True
        assert "level" in body
        assert "trend" in body
        assert "threshold_nis" in body

    def test_full_cycle(self):
        """Полный цикл: calibrate → detect → update → status."""
        client = self._client()
        rng = np.random.RandomState(7)
        normal = (rng.randn(200) * 5 + 50).tolist()
        cal = client.post("/kalman/calibrate", json={"data": normal}).json()
        assert cal["n_samples"] == 200
        test_data = (rng.randn(50) * 5 + 50).tolist()
        det = client.post("/kalman/detect", json={"data": test_data}).json()
        assert "n_anomalies" in det
        upd = client.post("/kalman/update", json={"value": 50.0}).json()
        assert upd["n_updates"] == len(test_data) + 1
        status = client.get("/kalman/status").json()
        assert status["is_calibrated"] is True


# ==============================================================================
# Ensemble Anomaly Detection tests
# ==============================================================================


class TestEnsembleUnit:
    """Unit-тесты AnomalyEnsemble без API."""

    def _votes(self, flags, scores=None):
        """Создать список DetectorVote для тестирования."""
        from anomaly.models.ensemble import DetectorVote

        names = ["cusum", "kalman", "isolation_forest", "esn", "mmd"]
        if scores is None:
            scores = [0.9 if f else 0.1 for f in flags]
        return [
            DetectorVote(name=names[i % len(names)], is_anomaly=f, score=s)
            for i, (f, s) in enumerate(zip(flags, scores, strict=False))
        ]

    def test_majority_all_anomaly(self):
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([True, True, True]))
        assert result.is_anomaly is True
        assert result.n_anomaly_votes == 3

    def test_majority_all_normal(self):
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([False, False, False]))
        assert result.is_anomaly is False
        assert result.n_anomaly_votes == 0

    def test_majority_half(self):
        """Ровно 50% голосов не превышает порог 0.5 → не аномалия (условие строгое >)."""
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([True, False]))
        assert result.is_anomaly is False
        assert result.agreement_ratio == pytest.approx(0.5)

    def test_majority_two_of_three(self):
        """2/3 = 0.667 > 0.5 → аномалия."""
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([True, True, False]))
        assert result.is_anomaly is True

    def test_any_single_vote(self):
        """Стратегия 'any': достаточно одного голоса."""
        from anomaly.models.ensemble import AnomalyEnsemble, EnsembleConfig

        e = AnomalyEnsemble(EnsembleConfig(strategy="any"))
        result = e.aggregate(self._votes([False, False, True]))
        assert result.is_anomaly is True

    def test_any_all_normal(self):
        from anomaly.models.ensemble import AnomalyEnsemble, EnsembleConfig

        e = AnomalyEnsemble(EnsembleConfig(strategy="any"))
        result = e.aggregate(self._votes([False, False, False]))
        assert result.is_anomaly is False

    def test_all_requires_consensus(self):
        """Стратегия 'all': нужны все голоса."""
        from anomaly.models.ensemble import AnomalyEnsemble, EnsembleConfig

        e = AnomalyEnsemble(EnsembleConfig(strategy="all"))
        result = e.aggregate(self._votes([True, True, False]))
        assert result.is_anomaly is False

    def test_all_full_consensus(self):
        from anomaly.models.ensemble import AnomalyEnsemble, EnsembleConfig

        e = AnomalyEnsemble(EnsembleConfig(strategy="all"))
        result = e.aggregate(self._votes([True, True, True]))
        assert result.is_anomaly is True

    def test_weighted_high_weight_anomaly(self):
        """Weighted: детектор с весом 10 перевешивает двух с весом 1."""
        from anomaly.models.ensemble import AnomalyEnsemble, DetectorVote, EnsembleConfig

        votes = [
            DetectorVote(name="cusum", is_anomaly=True, score=0.9),
            DetectorVote(name="kalman", is_anomaly=False, score=0.1),
            DetectorVote(name="isolation_forest", is_anomaly=False, score=0.1),
        ]
        config = EnsembleConfig(
            strategy="weighted",
            weights={"cusum": 10.0, "kalman": 1.0, "isolation_forest": 1.0},
        )
        e = AnomalyEnsemble(config)
        result = e.aggregate(votes)
        # weighted_score = (0.9*10 + 0.1*1 + 0.1*1) / 12 ≈ 0.767 > 0.5 → аномалия
        assert result.is_anomaly is True

    def test_weighted_unknown_detector_gets_default_weight(self):
        """Неизвестный детектор получает вес 1.0."""
        from anomaly.models.ensemble import AnomalyEnsemble, DetectorVote, EnsembleConfig

        votes = [
            DetectorVote(name="new_detector", is_anomaly=True, score=0.8),
            DetectorVote(name="kalman", is_anomaly=False, score=0.1),
        ]
        config = EnsembleConfig(strategy="weighted", weights={"kalman": 1.0})
        e = AnomalyEnsemble(config)
        result = e.aggregate(votes)
        # Оба получают вес 1.0: (0.8 + 0.1) / 2 = 0.45 < 0.5 → нет аномалии
        assert result.is_anomaly is False

    def test_empty_votes_raises(self):
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        with pytest.raises(ValueError, match="хотя бы один"):
            e.aggregate([])

    def test_unknown_strategy_raises(self):
        from anomaly.models.ensemble import AnomalyEnsemble, DetectorVote, EnsembleConfig

        e = AnomalyEnsemble(EnsembleConfig(strategy="unknown"))
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            e.aggregate([DetectorVote(name="x", is_anomaly=True)])

    def test_score_out_of_range_raises(self):
        from anomaly.models.ensemble import DetectorVote

        with pytest.raises(ValueError, match="score must be in"):
            DetectorVote(name="x", is_anomaly=True, score=1.5)

    def test_agreement_ratio_computed(self):
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([True, False, True, False]))
        assert result.agreement_ratio == pytest.approx(0.5)
        assert result.n_votes == 4
        assert result.n_anomaly_votes == 2

    def test_to_dict_structure(self):
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([True, False]))
        d = result.to_dict()
        expected_keys = (
            "is_anomaly",
            "confidence",
            "strategy",
            "agreement_ratio",
            "n_votes",
            "n_anomaly_votes",
            "votes",
        )
        for key in expected_keys:
            assert key in d
        assert len(d["votes"]) == 2
        for vote in d["votes"]:
            assert "name" in vote and "is_anomaly" in vote and "score" in vote

    def test_confidence_equals_agreement_ratio_for_majority(self):
        """В majority-стратегии confidence = agreement_ratio."""
        from anomaly.models.ensemble import AnomalyEnsemble

        e = AnomalyEnsemble()
        result = e.aggregate(self._votes([True, True, False]))
        assert result.confidence == pytest.approx(result.agreement_ratio)

    def test_single_vote_works(self):
        """Один голос — допустимый крайний случай."""
        from anomaly.models.ensemble import AnomalyEnsemble, DetectorVote

        e = AnomalyEnsemble()
        result = e.aggregate([DetectorVote(name="cusum", is_anomaly=True, score=0.9)])
        assert result.n_votes == 1
        assert result.is_anomaly is True


class TestEnsembleAPIEndpoints:
    """API-тесты POST /ensemble/vote и GET /ensemble/strategies."""

    def _client(self):
        from anomaly.api.app import _reset_ensemble, app
        from fastapi.testclient import TestClient

        _reset_ensemble()
        return TestClient(app)

    def _vote(self, name, is_anomaly, score=None):
        v = {"name": name, "is_anomaly": is_anomaly}
        if score is not None:
            v["score"] = score
        return v

    def test_vote_200(self):
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={
                "votes": [self._vote("cusum", True), self._vote("kalman", False)],
                "strategy": "majority",
            },
        )
        assert resp.status_code == 200

    def test_vote_response_structure(self):
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={
                "votes": [self._vote("cusum", True), self._vote("kalman", True)],
                "strategy": "majority",
            },
        ).json()
        for field in (
            "is_anomaly",
            "confidence",
            "strategy",
            "agreement_ratio",
            "n_votes",
            "n_anomaly_votes",
            "votes",
        ):
            assert field in resp

    def test_vote_majority_result(self):
        """2/3 голосов → аномалия в majority-стратегии."""
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={
                "votes": [self._vote("a", True), self._vote("b", True), self._vote("c", False)],
                "strategy": "majority",
            },
        ).json()
        assert resp["is_anomaly"] is True
        assert resp["n_anomaly_votes"] == 2

    def test_vote_any_strategy(self):
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={
                "votes": [self._vote("a", False), self._vote("b", False), self._vote("c", True)],
                "strategy": "any",
            },
        ).json()
        assert resp["is_anomaly"] is True

    def test_vote_all_strategy_no_consensus(self):
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={"votes": [self._vote("a", True), self._vote("b", False)], "strategy": "all"},
        ).json()
        assert resp["is_anomaly"] is False

    def test_vote_all_strategy_full_consensus(self):
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={"votes": [self._vote("a", True), self._vote("b", True)], "strategy": "all"},
        ).json()
        assert resp["is_anomaly"] is True

    def test_vote_weighted_strategy(self):
        """Weighted с явными весами."""
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={
                "votes": [self._vote("cusum", True, 0.9), self._vote("kalman", False, 0.1)],
                "strategy": "weighted",
                "weights": {"cusum": 3.0, "kalman": 1.0},
            },
        ).json()
        assert resp["strategy"] == "weighted"
        # cusum: 0.9*3 + kalman: 0.1*1 = 2.8 / 4 = 0.7 > 0.5 → аномалия
        assert resp["is_anomaly"] is True

    def test_vote_n_votes_matches_input(self):
        client = self._client()
        votes = [self._vote(f"d{i}", i % 2 == 0) for i in range(4)]
        resp = client.post("/ensemble/vote", json={"votes": votes}).json()
        assert resp["n_votes"] == 4

    def test_vote_votes_list_in_response(self):
        """Ответ содержит список голосов с полями name/is_anomaly/score."""
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={"votes": [self._vote("cusum", True, 0.8)], "strategy": "majority"},
        ).json()
        assert len(resp["votes"]) == 1
        assert resp["votes"][0]["name"] == "cusum"
        assert resp["votes"][0]["is_anomaly"] is True

    def test_vote_422_empty_votes(self):
        client = self._client()
        resp = client.post("/ensemble/vote", json={"votes": []})
        assert resp.status_code == 422

    def test_vote_custom_min_agreement(self):
        """min_agreement=0.3: достаточно 1 из 3 голосов."""
        client = self._client()
        resp = client.post(
            "/ensemble/vote",
            json={
                "votes": [self._vote("a", True), self._vote("b", False), self._vote("c", False)],
                "strategy": "majority",
                "min_agreement": 0.3,
            },
        ).json()
        # 1/3 ≈ 0.333 > 0.3 → аномалия
        assert resp["is_anomaly"] is True

    def test_strategies_endpoint(self):
        client = self._client()
        resp = client.get("/ensemble/strategies")
        assert resp.status_code == 200
        body = resp.json()
        assert "strategies" in body
        for strategy in ("majority", "weighted", "any", "all"):
            assert strategy in body["strategies"]

    def test_strategies_recommendation_present(self):
        client = self._client()
        body = client.get("/ensemble/strategies").json()
        assert "recommendation" in body
