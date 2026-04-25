"""
Anomaly predictor adapter (Project 05 bridge).

Реализует rolling Z-score детекцию из anomaly/models/detector.py
без зависимости от numpy для CI-совместимости (numpy-only, без torch).
В production: заменить на HTTP-вызов POST /detect anomaly-сервиса.

Подход: rolling Z-score с окном window_size.
score(t) = |x(t) - mean(window)| / std(window)
Аномалия если score > threshold_sigma (3σ по умолчанию).
"""

from __future__ import annotations

from ..models import AnomalyResult, MetricSnapshot

# Минимальное количество точек для надёжной статистики
_MIN_POINTS = 10


class AnomalyPredictor:
    """Rolling Z-score anomaly detector for multi-metric time series.

    Совместим с логикой StatisticalDetector из Project 05.
    Работает без numpy — чистый Python для максимальной переносимости.
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold_sigma: float = 3.0,
    ) -> None:
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma

    def _zscore_last(self, series: list[float]) -> float:
        """Compute z-score of the last point against the reference window."""
        if len(series) < _MIN_POINTS:
            return 0.0

        # Окно: все точки кроме последних 5 (текущие), но не более window_size
        ref_end = max(0, len(series) - 5)
        ref_start = max(0, ref_end - self.window_size)
        ref = series[ref_start:ref_end]

        if len(ref) < 2:
            return 0.0

        mean = sum(ref) / len(ref)
        variance = sum((x - mean) ** 2 for x in ref) / len(ref)
        std = variance**0.5

        # Максимальный z-score по последним 5 точкам
        current = series[ref_end:]

        if std == 0:
            # Нулевая дисперсия в окне: любое отклонение — однозначная аномалия.
            # Возвращаем фиксированный высокий скор выше любого порога.
            max_dev = max((abs(x - mean) for x in current), default=0.0)
            return 10.0 if max_dev > 0 else 0.0

        scores = [abs(x - mean) / std for x in current]
        return max(scores) if scores else 0.0

    def predict(self, metrics: MetricSnapshot) -> AnomalyResult:
        """Detect anomalies across cpu, latency, and requests metrics.

        Args:
            metrics: MetricSnapshot with time series for each metric.

        Returns:
            AnomalyResult with anomaly flag, max z-score, and affected metrics.
        """
        metric_series = {
            "cpu": metrics.cpu,
            "latency": metrics.latency,
            "requests": metrics.requests,
        }

        scores: dict[str, float] = {}
        for name, series in metric_series.items():
            scores[name] = self._zscore_last(series)

        affected = [name for name, score in scores.items() if score > self.threshold_sigma]
        max_score = max(scores.values()) if scores else 0.0

        return AnomalyResult(
            is_anomaly=len(affected) > 0,
            max_score=round(max_score, 4),
            affected_metrics=affected,
        )
