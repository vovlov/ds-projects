"""
CUSUM (CUmulative SUM) control chart for persistent shift detection.

Unlike Z-score (reactive to instantaneous spikes), CUSUM accumulates evidence
of systematic drift — ideal for gradual CPU exhaustion, memory leaks, or
slowly increasing latency invisible to threshold-based alerts.

Reference: Page 1954, "Continuous Inspection Schemes", Biometrika 41(1-2):100-115.
ARL (Average Run Length) analysis: Hawkins & Olwell 1998, "CUSUM Charts" Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CUSUMConfig:
    """Параметры CUSUM контрольной карты Шухарта-Пейджа.

    k: допустимое отклонение (slack). Стандарт: 0.5 для обнаружения сдвига 1σ.
       При k=0.5 ARL₀ ≈ 465 (Hawkins 1998) — примерно одна ложная тревога
       на ~500 нормальных точек при h=5.
    h: порог решения. Стандарт: 4–5 для χ=0.0027 (3σ-аналог).
       h=4: ARL₀≈168, h=5: ARL₀≈465. Больший h = меньше ложных тревог,
       но медленнее обнаружение.
    """

    k: float = 0.5
    h: float = 5.0


@dataclass
class CUSUMState:
    """Текущее состояние онлайн-детектора (для API status endpoint)."""

    s_pos: float
    s_neg: float
    mu_ref: float
    sigma_ref: float
    n_updates: int
    is_calibrated: bool
    n_alerts: int


@dataclass
class CUSUMBatchResult:
    """Результат батч-детекции CUSUM по временному ряду."""

    s_pos: list[float]
    s_neg: list[float]
    predictions: list[int]
    change_points: list[int]
    mu_ref: float
    sigma_ref: float
    threshold_k: float
    threshold_h: float
    n_alerts: int


@dataclass
class CUSUMUpdateResult:
    """Результат онлайн-обновления: одна точка → новое состояние."""

    s_pos: float
    s_neg: float
    is_alert: bool
    n_updates: int
    n_alerts: int


@dataclass
class CUSUMCalibrationResult:
    """Результат калибровки: оценки μ₀ и σ₀ на нормальных данных."""

    mu_ref: float
    sigma_ref: float
    n_calibration: int
    k: float
    h: float


class CUSUMDetector:
    """CUSUM детектор для одномерного временного ряда.

    Алгоритм Page 1954:
      zₜ = (xₜ - μ₀) / σ₀            — нормализованное отклонение
      S⁺ₜ = max(0, S⁺ₜ₋₁ + zₜ - k)   — верхняя CUSUM (рост выше нормы)
      S⁻ₜ = max(0, S⁻ₜ₋₁ - zₜ - k)   — нижняя CUSUM (падение ниже нормы)
      Тревога: S⁺ₜ > h ИЛИ S⁻ₜ > h

    После тревоги S⁺ или S⁻ сбрасывается в 0 (self-resetting variant),
    позволяя детектировать множественные смены за один вызов detect().

    Поддерживает два режима:
    - Батч: detect(series) — весь ряд сразу
    - Онлайн: update(value) — точка за точкой, для streaming
    """

    def __init__(self, config: CUSUMConfig | None = None):
        self._cfg = config or CUSUMConfig()
        self._mu: float = 0.0
        self._sigma: float = 1.0
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._n_updates: int = 0
        self._n_alerts: int = 0
        self._calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, normal_data: np.ndarray) -> CUSUMCalibrationResult:
        """Оценить μ₀ и σ₀ на нормальных данных; сбросить S⁺ и S⁻.

        Вызывать на «чистых» данных (без аномалий) перед мониторингом.
        Минимум 10 точек — стабильная оценка среднеквадратического отклонения.
        """
        if len(normal_data) < 10:
            raise ValueError("Need at least 10 calibration points to estimate μ₀ and σ₀")
        self._mu = float(np.mean(normal_data))
        # Защита от константного ряда: σ=0 → CUSUM на z=0 всегда, никогда не сработает.
        # Устанавливаем минимальный σ=1e-6 чтобы не делить на ноль.
        sigma = float(np.std(normal_data, ddof=1))
        self._sigma = max(sigma, 1e-6)
        # Сброс состояния после калибровки
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n_updates = 0
        self._n_alerts = 0
        self._calibrated = True
        return CUSUMCalibrationResult(
            mu_ref=self._mu,
            sigma_ref=self._sigma,
            n_calibration=len(normal_data),
            k=self._cfg.k,
            h=self._cfg.h,
        )

    def detect(self, series: np.ndarray) -> CUSUMBatchResult:
        """Батч-детекция CUSUM: ряд чисел → статистики + точки смены.

        Self-resetting: после каждой тревоги CUSUM сбрасывается в 0.
        Это позволяет ловить несколько смен режима в одном вызове.

        Требует предварительной калибровки.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() with normal data before detect()")

        n = len(series)
        s_pos_arr = np.zeros(n)
        s_neg_arr = np.zeros(n)
        preds = np.zeros(n, dtype=int)
        change_points: list[int] = []

        s_p = 0.0
        s_n = 0.0
        k = self._cfg.k
        h = self._cfg.h

        for i, x in enumerate(series):
            z = (x - self._mu) / self._sigma
            s_p = max(0.0, s_p + z - k)
            s_n = max(0.0, s_n - z - k)

            alert = (s_p > h) or (s_n > h)
            if alert:
                preds[i] = 1
                change_points.append(i)
                # Self-reset после тревоги — начинаем накапливать с нуля
                if s_p > h:
                    s_p = 0.0
                if s_n > h:
                    s_n = 0.0

            s_pos_arr[i] = s_p
            s_neg_arr[i] = s_n

        return CUSUMBatchResult(
            s_pos=s_pos_arr.tolist(),
            s_neg=s_neg_arr.tolist(),
            predictions=preds.tolist(),
            change_points=change_points,
            mu_ref=self._mu,
            sigma_ref=self._sigma,
            threshold_k=k,
            threshold_h=h,
            n_alerts=len(change_points),
        )

    def update(self, value: float) -> CUSUMUpdateResult:
        """Онлайн-обновление: принять одну точку, обновить S⁺/S⁻.

        Предназначен для streaming: каждую секунду новое значение → решение.
        Не требует хранения истории — весь контекст в S⁺ и S⁻.
        Self-resetting при тревоге (s_pos или s_neg → 0).
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() with normal data before update()")

        z = (value - self._mu) / self._sigma
        self._s_pos = max(0.0, self._s_pos + z - self._cfg.k)
        self._s_neg = max(0.0, self._s_neg - z - self._cfg.k)

        is_alert = (self._s_pos > self._cfg.h) or (self._s_neg > self._cfg.h)
        if is_alert:
            self._n_alerts += 1
            if self._s_pos > self._cfg.h:
                self._s_pos = 0.0
            if self._s_neg > self._cfg.h:
                self._s_neg = 0.0

        self._n_updates += 1
        return CUSUMUpdateResult(
            s_pos=self._s_pos,
            s_neg=self._s_neg,
            is_alert=is_alert,
            n_updates=self._n_updates,
            n_alerts=self._n_alerts,
        )

    def reset(self) -> None:
        """Сброс CUSUM статистик в 0 (без потери калибровки).

        Вызывать при известной смене режима (например, после retraining модели)
        чтобы не получать тревоги на переходный период.
        """
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n_updates = 0
        self._n_alerts = 0

    def get_state(self) -> CUSUMState:
        """Вернуть текущее состояние детектора (для API /cusum/status)."""
        return CUSUMState(
            s_pos=self._s_pos,
            s_neg=self._s_neg,
            mu_ref=self._mu,
            sigma_ref=self._sigma,
            n_updates=self._n_updates,
            is_calibrated=self._calibrated,
            n_alerts=self._n_alerts,
        )
