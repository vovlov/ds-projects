"""
Prometheus metrics exporter для сервиса детекции аномалий.

Зачем Prometheus? SRE-команды используют Prometheus+Grafana как стандарт
мониторинга. Экспортируя метрики в формате /metrics, сервис становится
наблюдаемым (observable): можно настроить алёрты на anomaly_rate > 5%
или API latency > 200ms прямо в Alertmanager.

Принцип: метрики — это о сервисе, а не о данных. Отслеживаем:
- throughput (сколько точек обработали)
- anomaly rate (доля аномалий, SLO-критичная метрика)
- detection latency (производительность детектора)
- model health (текущий порог, окно детектора)
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager

# Graceful import: если prometheus_client не установлен в среде,
# используем заглушки чтобы не ломать CI без зависимости
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def is_available() -> bool:
    """Check if prometheus_client is installed."""
    return PROMETHEUS_AVAILABLE


class AnomalyMetrics:
    """
    Prometheus-метрики для сервиса детекции аномалий.

    Использует отдельный registry (не глобальный default) чтобы
    несколько экземпляров в тестах не конфликтовали между собой.

    Metrics exported:
        anomaly_detector_requests_total     — кол-во запросов к /detect
        anomaly_detector_points_total       — кол-во обработанных точек
        anomaly_detector_anomalies_total    — кол-во обнаруженных аномалий
        anomaly_detector_anomaly_score      — гистограмма Z-score аномалий
        anomaly_detector_detection_seconds  — задержка детекции (latency)
        anomaly_detector_threshold          — текущий порог (σ)
        anomaly_detector_window_size        — размер скользящего окна
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if not PROMETHEUS_AVAILABLE:
            raise RuntimeError(
                "prometheus_client is not installed. Install it with: pip install prometheus-client"
            )

        # Отдельный registry предотвращает дублирование метрик в тестах
        self.registry = registry or CollectorRegistry()

        self.requests_total = Counter(
            "anomaly_detector_requests_total",
            "Total number of detection requests to /detect endpoint",
            registry=self.registry,
        )

        self.points_total = Counter(
            "anomaly_detector_points_total",
            "Total number of metric points processed",
            registry=self.registry,
        )

        self.anomalies_total = Counter(
            "anomaly_detector_anomalies_total",
            "Total number of anomalies detected",
            registry=self.registry,
        )

        # Гистограмма Z-score: позволяет увидеть распределение "серьёзности"
        # аномалий — не просто count, но и severity
        self.anomaly_score = Histogram(
            "anomaly_detector_anomaly_score",
            "Distribution of anomaly Z-scores for detected anomalies",
            buckets=[1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, float("inf")],
            registry=self.registry,
        )

        # Latency детектора: критично для SRE SLO (должно быть < 10ms)
        self.detection_seconds = Histogram(
            "anomaly_detector_detection_seconds",
            "Time spent in anomaly detection (seconds)",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry,
        )

        # Gauge-метрики: текущее состояние модели
        self.threshold = Gauge(
            "anomaly_detector_threshold",
            "Current anomaly detection threshold (sigma)",
            registry=self.registry,
        )

        self.window_size = Gauge(
            "anomaly_detector_window_size",
            "Current sliding window size for Z-score calculation",
            registry=self.registry,
        )

    def record_request(self, n_points: int, n_anomalies: int, scores: list[float]) -> None:
        """
        Записать метрики одного запроса к /detect.

        Args:
            n_points: кол-во точек в запросе
            n_anomalies: кол-во обнаруженных аномалий
            scores: Z-score для каждой аномальной точки
        """
        self.requests_total.inc()
        self.points_total.inc(n_points)
        self.anomalies_total.inc(n_anomalies)

        # Записываем score только для аномалий — нормальные точки не интересны
        for score in scores:
            self.anomaly_score.observe(score)

    def set_model_config(self, threshold_sigma: float, window_size: int) -> None:
        """
        Обновить gauge-метрики конфигурации модели.

        Вызывается при инициализации и при изменении параметров детектора.
        Позволяет в Grafana видеть, с каким порогом работает модель.
        """
        self.threshold.set(threshold_sigma)
        self.window_size.set(window_size)

    @contextmanager
    def track_detection(self) -> Generator[None, None, None]:
        """
        Context manager для измерения latency детекции.

        Usage:
            with metrics.track_detection():
                result = detector.detect(data)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.detection_seconds.observe(elapsed)

    def get_metrics_text(self) -> str:
        """
        Вернуть метрики в формате Prometheus text exposition.

        Используется в /metrics endpoint FastAPI.
        Content-Type: text/plain; version=0.0.4
        """
        return generate_latest(self.registry).decode("utf-8")

    def get_summary(self) -> dict[str, float]:
        """
        Вернуть сводку метрик как dict (для /health endpoint).

        Позволяет видеть ключевые показатели без парсинга Prometheus-формата.
        """
        # Считаем anomaly_rate из накопленных счётчиков
        total_points = self.points_total._value.get()
        total_anomalies = self.anomalies_total._value.get()

        anomaly_rate = (total_anomalies / total_points) if total_points > 0 else 0.0

        return {
            "requests_total": self.requests_total._value.get(),
            "points_total": total_points,
            "anomalies_total": total_anomalies,
            "anomaly_rate": round(anomaly_rate, 4),
            "threshold_sigma": self.threshold._value.get(),
            "window_size": self.window_size._value.get(),
        }
