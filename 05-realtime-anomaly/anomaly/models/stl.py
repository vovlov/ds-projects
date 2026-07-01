"""
STL-inspired Classical Seasonal Decomposition for time series anomaly detection.

Decomposes a signal into Trend + Seasonal + Residual via Centered Moving Average (CMA).
Anomalies are detected in the residual component using a robust Z-score (MAD-based),
making detection immune to the trend and seasonal variation that would dominate a
naive threshold approach for metrics like CPU usage (daily cycle) or request rates.

Algorithm: Cleveland et al. 1990 simplified CMA variant (no LOESS, numpy-only).
Robust Z: Rousseeuw & Croux 1993 MAD = median(|x - median(x)|) * 1.4826.
OneShotSTL reference: He et al. 2023 (arxiv:2304.01506) — O(1) online update idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class STLConfig:
    """Параметры STL-декомпозиции.

    period: длина сезонного цикла в точках. 24 = часовые данные с суточной сезонностью.
    threshold_z: порог аномалии в единицах робастного sigma (MAD-based).
    robust: использовать MAD вместо std (устойчивость к выбросам в train-данных).
    min_periods: минимум наблюдений для калибровки (≥ 2 полных периода).
    """

    period: int = 24
    threshold_z: float = 3.0
    robust: bool = True
    min_periods: int = 2


@dataclass
class STLCalibrationResult:
    """Результат калибровки: сезонный паттерн + статистики остатка."""

    period: int
    n_samples: int
    seasonal_pattern: list[float]
    mu_residual: float
    sigma_residual: float
    n_complete_cycles: int


@dataclass
class STLDecomposition:
    """Результат батч-декомпозиции временного ряда."""

    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    anomaly_score: list[float]
    predictions: list[int]
    anomaly_indices: list[int]
    n_anomalies: int
    period: int
    threshold_z: float


@dataclass
class STLUpdateResult:
    """Результат онлайн-обновления: одна точка."""

    value: float
    trend_estimate: float
    seasonal_estimate: float
    residual: float
    anomaly_score: float
    is_anomaly: bool
    n_updates: int


@dataclass
class STLState:
    """Текущее состояние детектора для /stl/status endpoint."""

    is_calibrated: bool
    period: int
    threshold_z: float
    n_calibration: int
    n_updates: int
    mu_residual: float
    sigma_residual: float
    seasonal_pattern: list[float] = field(default_factory=list)


class STLDetector:
    """Seasonal-Trend decomposition via Centered Moving Average + robust residual scoring.

    Три режима:
    1. calibrate(data) — оценить сезонный паттерн и статистики остатков на нормальных данных
    2. detect(data) — батч-декомпозиция с аномальными метками
    3. update(value) — онлайн-обновление O(period) памяти

    Без внешних зависимостей — работает в CI и на macOS x86_64 без PyTorch/statsmodels.
    """

    def __init__(self, config: STLConfig | None = None) -> None:
        self.config = config or STLConfig()
        self._is_calibrated = False
        self._seasonal_pattern: np.ndarray = np.zeros(self.config.period)
        self._mu_residual: float = 0.0
        self._sigma_residual: float = 1.0
        self._n_calibration: int = 0
        self._n_updates: int = 0
        # Скользящий буфер для online-режима (2 периода достаточно для локального CMA)
        self._buffer: list[float] = []
        self._buffer_maxlen = 2 * self.config.period + 1

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _centered_moving_average(self, data: np.ndarray) -> np.ndarray:
        """Центрированное скользящее среднее с окном = period.

        Для чётного period использует двойное сглаживание 2×CMA(period)
        чтобы выровнять временну́ю метку (стандарт классической декомпозиции).
        """
        p = self.config.period
        n = len(data)
        trend = np.full(n, np.nan)

        if p % 2 == 1:
            half = p // 2
            for i in range(half, n - half):
                trend[i] = np.mean(data[i - half : i + half + 1])
        else:
            # Двойной CMA: сначала p, потом 2 (даёт центрированный p-MA)
            half = p // 2
            ma1 = np.full(n, np.nan)
            for i in range(half, n - half):
                ma1[i] = np.mean(data[i - half + 1 : i + half + 1])
            # Второй MA длиной 2 выравнивает метку
            for i in range(half, n - half):
                if not (np.isnan(ma1[i]) or np.isnan(ma1[i - 1])):
                    trend[i] = (ma1[i] + ma1[i - 1]) / 2.0

        return trend

    def _estimate_seasonal(self, data: np.ndarray, trend: np.ndarray) -> np.ndarray:
        """Оценить сезонный паттерн из детрендированного ряда.

        seasonal[j] = среднее деtrended_t для всех t: t mod period == j.
        Нормируем чтобы сумма seasonal_pattern == 0 (additive decomposition).
        """
        p = self.config.period
        detrended = data - trend
        pattern = np.zeros(p)
        counts = np.zeros(p, dtype=int)

        for i, val in enumerate(detrended):
            if not np.isnan(val):
                j = i % p
                pattern[j] += val
                counts[j] += 1

        # Защита от пустых бинов (данных меньше одного периода)
        mask = counts > 0
        pattern[mask] /= counts[mask]

        # Нормировка: сумма паттерна = 0 (additive invariant)
        pattern -= pattern.mean()
        return pattern

    def _robust_sigma(self, residuals: np.ndarray) -> tuple[float, float]:
        """Вернуть (μ, σ) через медиану и MAD (устойчиво к аномалиям в train-данных)."""
        clean = residuals[~np.isnan(residuals)]
        if len(clean) == 0:
            return 0.0, 1.0
        mu = float(np.median(clean))
        if self.config.robust:
            mad = float(np.median(np.abs(clean - mu)))
            sigma = max(mad * 1.4826, 1e-8)  # 1.4826 = consistency factor для Normal
        else:
            sigma = max(float(np.std(clean)), 1e-8)
        return mu, sigma

    def _anomaly_score(self, residual: float) -> float:
        """Робастная Z-оценка: |residual - μ| / σ, затем нормализуем в [0, 1].

        Нормировка: score = |z| / (|z| + threshold_z) → sigmoid-образная форма
        где score=0.5 точно на пороге (|z|=threshold_z).
        """
        z = abs(residual - self._mu_residual) / self._sigma_residual
        return float(z / (z + self.config.threshold_z))

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def calibrate(self, data: list[float] | np.ndarray) -> STLCalibrationResult:
        """Откалибровать детектор на нормальных данных.

        Оценивает сезонный паттерн (CMA-based) и статистики остатков (μ, σ).
        Требует ≥ min_periods × period точек.

        Raises:
            ValueError: если данных недостаточно для оценки паттерна.
        """
        p = self.config.period
        arr = np.asarray(data, dtype=float)
        min_len = self.config.min_periods * p
        if len(arr) < min_len:
            raise ValueError(
                f"Need ≥ {min_len} points for period={p} × min_periods={self.config.min_periods}, "
                f"got {len(arr)}"
            )

        trend = self._centered_moving_average(arr)
        self._seasonal_pattern = self._estimate_seasonal(arr, trend)

        # Вычислить остатки там где есть тренд
        seasonal_component = np.array([self._seasonal_pattern[i % p] for i in range(len(arr))])
        residuals = arr - trend - seasonal_component

        self._mu_residual, self._sigma_residual = self._robust_sigma(residuals)
        self._n_calibration = len(arr)
        self._n_updates = 0
        self._buffer = list(arr[-self._buffer_maxlen :])  # инициализировать онлайн-буфер
        self._is_calibrated = True

        return STLCalibrationResult(
            period=p,
            n_samples=len(arr),
            seasonal_pattern=self._seasonal_pattern.tolist(),
            mu_residual=self._mu_residual,
            sigma_residual=self._sigma_residual,
            n_complete_cycles=len(arr) // p,
        )

    def detect(self, data: list[float] | np.ndarray) -> STLDecomposition:
        """Батч-декомпозиция и аномальные метки.

        Работает без калибровки (использует встроенный паттерн нулей), но
        рекомендуется вызвать calibrate() сначала для корректных порогов.
        """
        p = self.config.period
        arr = np.asarray(data, dtype=float)

        if len(arr) < p:
            raise ValueError(f"Need ≥ {p} points for decomposition, got {len(arr)}")

        trend = self._centered_moving_average(arr)

        # Если не откалиброваны — оценить паттерн из текущих данных
        if self._is_calibrated:
            seasonal_pattern = self._seasonal_pattern
        else:
            seasonal_pattern = self._estimate_seasonal(arr, trend)

        seasonal = np.array([seasonal_pattern[i % p] for i in range(len(arr))])
        residual = arr - trend - seasonal

        # Для NaN (края тренда) используем центральный residual (avg)
        mu, sigma = self._robust_sigma(residual)
        effective_mu = mu if self._is_calibrated else mu
        effective_sigma = sigma if self._is_calibrated else sigma

        scores: list[float] = []
        preds: list[int] = []
        for r in residual:
            if np.isnan(r):
                scores.append(0.0)
                preds.append(0)
            else:
                z = abs(r - effective_mu) / effective_sigma
                score = float(z / (z + self.config.threshold_z))
                scores.append(score)
                preds.append(1 if z > self.config.threshold_z else 0)

        anomaly_indices = [i for i, p_ in enumerate(preds) if p_ == 1]

        def _nan_to_zero(arr_: np.ndarray) -> list[float]:
            return [0.0 if np.isnan(v) else float(v) for v in arr_]

        return STLDecomposition(
            trend=_nan_to_zero(trend),
            seasonal=seasonal.tolist(),
            residual=_nan_to_zero(residual),
            anomaly_score=scores,
            predictions=preds,
            anomaly_indices=anomaly_indices,
            n_anomalies=len(anomaly_indices),
            period=p,
            threshold_z=self.config.threshold_z,
        )

    def update(self, value: float) -> STLUpdateResult:
        """Онлайн-обновление: одна точка → аномалия / норма.

        Использует скользящий буфер размером 2×period для локальной оценки
        тренда. Сезонный паттерн берётся из калибровки (не обновляется онлайн).

        Raises:
            RuntimeError: если calibrate() не был вызван.
        """
        if not self._is_calibrated:
            raise RuntimeError("Call calibrate() before update()")

        p = self.config.period
        self._buffer.append(value)
        if len(self._buffer) > self._buffer_maxlen:
            self._buffer.pop(0)

        self._n_updates += 1
        phase = (self._n_calibration + self._n_updates - 1) % p

        # Локальный тренд из буфера
        buf = np.array(self._buffer, dtype=float)
        if len(buf) >= p:
            trend_estimate = float(np.mean(buf[-p:]))
        else:
            trend_estimate = float(np.mean(buf))

        seasonal_estimate = float(self._seasonal_pattern[phase])
        residual = value - trend_estimate - seasonal_estimate
        score = self._anomaly_score(residual)
        z = abs(residual - self._mu_residual) / self._sigma_residual
        is_anomaly = bool(z > self.config.threshold_z)

        return STLUpdateResult(
            value=value,
            trend_estimate=trend_estimate,
            seasonal_estimate=seasonal_estimate,
            residual=residual,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            n_updates=self._n_updates,
        )

    def get_state(self) -> STLState:
        """Текущее состояние детектора для мониторинга и /stl/status endpoint."""
        return STLState(
            is_calibrated=self._is_calibrated,
            period=self.config.period,
            threshold_z=self.config.threshold_z,
            n_calibration=self._n_calibration,
            n_updates=self._n_updates,
            mu_residual=self._mu_residual,
            sigma_residual=self._sigma_residual,
            seasonal_pattern=self._seasonal_pattern.tolist(),
        )

    def reset(self) -> None:
        """Сброс состояния детектора (не трогает config)."""
        self._is_calibrated = False
        self._seasonal_pattern = np.zeros(self.config.period)
        self._mu_residual = 0.0
        self._sigma_residual = 1.0
        self._n_calibration = 0
        self._n_updates = 0
        self._buffer = []
