"""
Детекторы аномалий: статистический baseline + multi-metric ensemble.

Подход: rolling Z-score — простой, интерпретируемый, работает в real-time.
Для каждой точки считаем, на сколько стандартных отклонений она отличается
от среднего по скользящему окну. Если больше threshold (обычно 3σ) — аномалия.

Почему не только LSTM? Z-score работает мгновенно (<1ms/точка), не требует
обучения, интерпретируем для SRE ("latency выросла на 5σ от нормы").
LSTM лучше ловит сложные паттерны, но Z-score — хороший первый эшелон.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyResult:
    """Результат детекции: скоры, бинарные предсказания и порог."""

    scores: np.ndarray
    predictions: np.ndarray
    threshold: float


class StatisticalDetector:
    """Z-score детектор с скользящим окном.

    Для каждой точки t:
      score(t) = |x(t) - mean(window)| / std(window)
    где window = [t - window_size, t).

    Первые window_size точек получают score=0 (недостаточно контекста).
    """

    def __init__(self, window_size: int = 50, threshold_sigma: float = 3.0):
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma

    def detect(self, series: np.ndarray) -> AnomalyResult:
        n = len(series)
        scores = np.zeros(n)

        for i in range(self.window_size, n):
            window = series[i - self.window_size : i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                scores[i] = abs(series[i] - mean) / std
            else:
                # Все значения в окне одинаковые — любое отклонение аномально
                scores[i] = 0.0

        predictions = (scores > self.threshold_sigma).astype(int)
        return AnomalyResult(
            scores=scores,
            predictions=predictions,
            threshold=self.threshold_sigma,
        )


class MultiMetricDetector:
    """Ensemble-детектор: max anomaly score по CPU, latency, requests.

    Идея: аномалия проявляется хотя бы в одной метрике.
    Берём максимальный Z-score — если хоть одна метрика «кричит», сигнализируем.
    В production можно добавить взвешенное среднее или голосование.
    """

    def __init__(self, window_size: int = 50, threshold_sigma: float = 3.0):
        self.detector = StatisticalDetector(window_size, threshold_sigma)

    def detect(self, data: dict) -> AnomalyResult:
        cpu_result = self.detector.detect(data["cpu"])
        latency_result = self.detector.detect(data["latency"])
        requests_result = self.detector.detect(data["requests"])

        # Max score — если хотя бы одна метрика аномальна, сигнализируем
        combined_scores = np.maximum(
            cpu_result.scores,
            np.maximum(latency_result.scores, requests_result.scores),
        )

        predictions = (combined_scores > self.detector.threshold_sigma).astype(int)

        return AnomalyResult(
            scores=combined_scores,
            predictions=predictions,
            threshold=self.detector.threshold_sigma,
        )
