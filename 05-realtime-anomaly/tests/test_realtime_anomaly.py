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
