"""Kalman Filter for online anomaly detection.

Модель состояния: x = [level, trend]^T (constant-velocity / random walk with drift).
  x_t = F·x_{t-1} + w,  w ~ N(0, Q)
  y_t = H·x_t + v,      v ~ N(0, R)

F = [[1, 1], [0, 1]],  H = [1, 0]

Anomaly score — Normalized Innovation Squared (NIS):
  NIS_t = ν_t² / S_t,  ν_t = y_t - H·x̂_t|t-1,  S_t = H·P_t|t-1·H^T + R

Под H₀ (нет аномалии) NIS ~ χ²(1), поэтому NIS > χ²_{1-α}(1) = тревога.
Порог не требует эмпирической калибровки — вытекает из статистической теории.

Преимущество перед CUSUM: CUSUM реагирует на персистентный сдвиг уровня,
Kalman + NIS — на любой выброс (impulse) или смену дисперсии, автоматически
адаптируя предсказание к текущему тренду.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Chi-squared inverse CDF table for df=1 (scalar observations).
# Позволяет работать без scipy в CI.
_CHI2_1: dict[float, float] = {
    0.10: 2.706,
    0.05: 3.841,
    0.025: 5.024,
    0.01: 6.635,
    0.005: 7.879,
    0.001: 10.828,
}


@dataclass
class KalmanConfig:
    """Параметры Kalman Filter детектора.

    process_noise_level: дисперсия шума уровня (Q[0,0]).
    process_noise_trend: дисперсия шума тренда (Q[1,1]); мало — тренд меняется медленно.
    measurement_noise:   дисперсия шума наблюдения (R). None → оценивается из данных.
    anomaly_alpha:       уровень значимости для chi2 порога (0.01 = 1% ложных тревог).
    """

    process_noise_level: float = 1e-3
    process_noise_trend: float = 1e-5
    measurement_noise: float | None = None
    anomaly_alpha: float = 0.01


@dataclass
class KalmanCalibrationResult:
    """Результат калибровки на нормальных данных."""

    estimated_R: float
    n_samples: int
    initial_level: float
    initial_trend: float
    threshold_nis: float


@dataclass
class KalmanUpdateResult:
    """Результат обработки одной точки (online update)."""

    level: float
    trend: float
    predicted: float
    innovation: float
    nis: float
    threshold: float
    is_anomaly: bool
    n_updates: int


@dataclass
class KalmanBatchResult:
    """Результат батч-детекции."""

    levels: list[float]
    trends: list[float]
    predicted: list[float]
    innovations: list[float]
    nis_scores: list[float]
    predictions: list[bool]
    threshold: float
    anomaly_indices: list[int]
    n_anomalies: int


class KalmanDetector:
    """Online univariate anomaly detector на основе Kalman Filter + NIS score.

    Идеален для гладких временных рядов с трендом (CPU, latency, throughput):
    адаптивно отслеживает уровень и тренд, бьёт тревогу только при статистически
    значимых отклонениях от предсказания.

    Usage::

        det = KalmanDetector()
        det.calibrate(normal_cpu_series)           # fit R, init state
        result = det.detect(new_cpu_series)        # batch
        online = det.update(single_value)          # streaming
    """

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self._reset_state()

    def _reset_state(self) -> None:
        self._is_calibrated = False
        self._R: float = 1.0
        self._threshold: float = _CHI2_1[0.01]
        self._x = np.zeros(2)
        self._P = np.eye(2) * 1e6  # diffuse prior
        self._n_updates: int = 0
        # Constant model matrices
        self._F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self._H = np.array([[1.0, 0.0]])

    def _Q(self) -> np.ndarray:
        return np.diag([self.config.process_noise_level, self.config.process_noise_trend])

    def calibrate(
        self, normal_data: list[float] | np.ndarray
    ) -> KalmanCalibrationResult:
        """Оценить R из нормальных данных и инициализировать состояние.

        R оценивается как дисперсия детрендированного ряда — proxy для
        «шума наблюдения» при условии, что данные не содержат аномалий.
        """
        y = np.asarray(normal_data, dtype=float)
        n = len(y)
        if n < 10:
            raise ValueError(f"Need ≥ 10 samples for calibration, got {n}")

        if self.config.measurement_noise is not None:
            R = self.config.measurement_noise
        else:
            # OLS-детрендирование + оценка дисперсии остатков
            t = np.arange(n, dtype=float)
            slope, intercept = np.polyfit(t, y, 1)
            residuals = y - (slope * t + intercept)
            R = float(np.var(residuals, ddof=1))
            R = max(R, 1e-8)  # защита от константного ряда

        # Начальное состояние: уровень = первое значение, тренд = средняя скорость
        initial_level = float(y[0])
        window = min(10, n)
        initial_trend = float(np.mean(np.diff(y[:window])))

        # Инициализируем Kalman state
        self._R = R
        self._x = np.array([initial_level, initial_trend])
        self._P = np.eye(2) * R  # умеренно диффузный prior
        self._n_updates = 0

        # Порог из chi2 таблицы
        alpha = self.config.anomaly_alpha
        # Берём ближайшее значение из таблицы
        available = sorted(_CHI2_1.keys())
        closest = min(available, key=lambda a: abs(a - alpha))
        self._threshold = _CHI2_1[closest]

        self._is_calibrated = True

        return KalmanCalibrationResult(
            estimated_R=R,
            n_samples=n,
            initial_level=initial_level,
            initial_trend=initial_trend,
            threshold_nis=self._threshold,
        )

    def update(self, observation: float) -> KalmanUpdateResult:
        """Обработать одну точку: predict → update → NIS score.

        Обновляет внутреннее состояние фильтра — вызывать последовательно
        для онлайн-потока. Можно чередовать с detect().
        """
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before update()")

        F, H = self._F, self._H
        Q = self._Q()
        R_mat = np.array([[self._R]])

        # --- Predict ---
        x_pred = F @ self._x            # shape (2,)
        P_pred = F @ self._P @ F.T + Q  # shape (2,2)

        # --- Innovation ---
        y_scalar = float(observation)
        # H is (1,2), x_pred is (2,) → product is (1,); take element 0
        y_pred = float((H @ x_pred)[0])
        innovation = y_scalar - y_pred
        # S is innovation variance: scalar extracted from (1,1) matrix
        S = float((H @ P_pred @ H.T + R_mat)[0, 0])

        # NIS: квадрат нормализованного нововведения
        nis = (innovation ** 2) / S

        # --- Kalman Gain & Update ---
        K = (P_pred @ H.T) / S   # shape (2,1) / scalar = (2,1)
        self._x = x_pred + K[:, 0] * innovation   # (2,) + (2,) * scalar
        # Joseph form для числовой стабильности
        I_KH = np.eye(2) - np.outer(K[:, 0], H[0])  # (2,2)
        self._P = I_KH @ P_pred

        self._n_updates += 1

        return KalmanUpdateResult(
            level=float(self._x[0]),
            trend=float(self._x[1]),
            predicted=y_pred,
            innovation=innovation,
            nis=float(nis),
            threshold=self._threshold,
            is_anomaly=bool(nis > self._threshold),
            n_updates=self._n_updates,
        )

    def detect(self, data: list[float] | np.ndarray) -> KalmanBatchResult:
        """Батч-детекция: процессирует точки последовательно, обновляет стейт."""
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before detect()")

        y = np.asarray(data, dtype=float)
        levels, trends, predicted, innovations, nis_scores, preds = [], [], [], [], [], []

        for val in y:
            r = self.update(float(val))
            levels.append(r.level)
            trends.append(r.trend)
            predicted.append(r.predicted)
            innovations.append(r.innovation)
            nis_scores.append(r.nis)
            preds.append(r.is_anomaly)

        anomaly_indices = [i for i, p in enumerate(preds) if p]

        return KalmanBatchResult(
            levels=levels,
            trends=trends,
            predicted=predicted,
            innovations=innovations,
            nis_scores=nis_scores,
            predictions=preds,
            threshold=self._threshold,
            anomaly_indices=anomaly_indices,
            n_anomalies=len(anomaly_indices),
        )

    def get_state(self) -> dict:
        """Текущее состояние фильтра для мониторинга."""
        return {
            "is_calibrated": self._is_calibrated,
            "level": float(self._x[0]) if self._is_calibrated else None,
            "trend": float(self._x[1]) if self._is_calibrated else None,
            "measurement_noise_R": self._R if self._is_calibrated else None,
            "threshold_nis": self._threshold if self._is_calibrated else None,
            "n_updates": self._n_updates,
        }

    def reset(self) -> None:
        """Сброс к начальному состоянию (без калибровки)."""
        self._reset_state()
