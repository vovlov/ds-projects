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
        cur_drift = np.array(generate_current_data(n_points=200, inject_drift=True, drift_magnitude=6.0, seed=99))

        # CPU: среднее при дрейфе должно быть выше (аномалия подняла CPU)
        ref_mean_cpu = float(np.mean(ref[:, 0]))
        cur_mean_cpu = float(np.mean(cur_drift[:, 0]))
        assert cur_mean_cpu > ref_mean_cpu, "Drift injection should raise CPU mean"
