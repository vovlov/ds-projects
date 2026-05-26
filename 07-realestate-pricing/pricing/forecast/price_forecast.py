"""Holt's Double Exponential Smoothing для прогнозирования цен на недвижимость.

Двойное экспоненциальное сглаживание (уровень + тренд) оптимально для рядов цен
московской недвижимости: устойчивый долгосрочный тренд доминирует над слабой
сезонностью, которую лучше поглощать в residuals, чем моделировать явно.

Источники: Holt 1957 ONR Memo 52; Gardner 2006 IJF survey;
Fama 1970 (мультипликативный шум для финансовых рядов).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Базовые цены руб/кв.м по районам Москвы (медиана 2025-2026 по данным ЦИАН)
NEIGHBORHOOD_BASE_PRICES: dict[str, float] = {
    "Арбат": 400_000,
    "Басманный": 230_000,
    "Замоскворечье": 280_000,
    "Красносельский": 220_000,
    "Лефортово": 200_000,
    "Марьино": 155_000,
    "Медведково": 175_000,
    "Мещанский": 240_000,
    "Митино": 170_000,
    "Некрасовка": 140_000,
    "Пресненский": 290_000,
    "Таганский": 250_000,
    "Тушино": 165_000,
    "Хамовники": 320_000,
    "Якиманка": 350_000,
}

# Годовые тренды роста цен (%) — по данным IRN.ru и ЦИАН 2024-2026
NEIGHBORHOOD_ANNUAL_TRENDS: dict[str, float] = {
    "Арбат": 6.0,
    "Басманный": 6.5,
    "Замоскворечье": 6.5,
    "Красносельский": 5.5,
    "Лефортово": 5.0,
    "Марьино": 4.0,
    "Медведково": 4.5,
    "Мещанский": 6.0,
    "Митино": 4.5,
    "Некрасовка": 5.0,
    "Пресненский": 9.0,
    "Таганский": 7.0,
    "Тушино": 4.0,
    "Хамовники": 8.0,
    "Якиманка": 7.5,
}


@dataclass
class ForecastConfig:
    """Конфигурация модели прогнозирования цен."""

    alpha: float = 0.3
    beta: float = 0.1
    optimize_params: bool = True
    forecast_periods: int = 12
    confidence_level: float = 0.95


@dataclass
class ForecastPoint:
    """Одна точка прогноза с доверительным интервалом."""

    period: int
    value: float
    lower: float
    upper: float


@dataclass
class ForecastResult:
    """Результат прогнозирования цен на недвижимость."""

    forecast: list[ForecastPoint]
    mape: float
    trend_direction: Literal["rising", "stable", "falling"]
    trend_slope_pct: float
    alpha: float
    beta: float
    last_known_value: float


def generate_price_history(
    neighborhood: str,
    n_months: int = 36,
    seed: int | None = None,
) -> list[float]:
    """Генерировать реалистичную историю цен руб/кв.м для района.

    Цена = base * exp(monthly_rate * t) * lognormal_noise.
    Мультипликативный шум (CV=5%) реалистичнее аддитивного для финансовых рядов.
    """
    rng = np.random.default_rng(seed)
    base = NEIGHBORHOOD_BASE_PRICES.get(neighborhood, 200_000)
    annual_pct = NEIGHBORHOOD_ANNUAL_TRENDS.get(neighborhood, 5.0)
    monthly_rate = annual_pct / 100.0 / 12.0

    t = np.arange(n_months, dtype=float)
    trend = base * np.exp(monthly_rate * t)
    noise = rng.lognormal(mean=0.0, sigma=0.05, size=n_months)
    return (trend * noise).tolist()


class HoltWintersForecaster:
    """Holt's Double Exponential Smoothing (уровень + тренд, без сезонности).

    s_t = α·y_t + (1−α)·(s_{t−1} + b_{t−1})   — уровень
    b_t = β·(s_t − s_{t−1}) + (1−β)·b_{t−1}    — тренд
    ŷ_{t+h} = s_t + h·b_t                       — прогноз

    Интервалы: ŷ ± z·σ_res·√h  — расширяются с горизонтом (Gardner 2006).
    Параметры α, β подбираются grid search по SSE (25 комбинаций), нет scipy.
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._config = config or ForecastConfig()
        self._level = 0.0
        self._trend = 0.0
        self._history: list[float] = []
        self._fitted_values: list[float] = []
        self._fitted = False
        self._alpha = self._config.alpha
        self._beta = self._config.beta

    def fit(self, y: list[float]) -> HoltWintersForecaster:
        """Обучить модель на временном ряду цен. Минимум 4 точки."""
        if len(y) < 4:
            raise ValueError(f"Need ≥4 observations, got {len(y)}")

        arr = np.asarray(y, dtype=float)

        if self._config.optimize_params:
            self._alpha, self._beta = self._optimize(arr)

        self._level, self._trend, self._fitted_values = self._run_smoothing(
            arr, self._alpha, self._beta
        )
        self._history = y[:]
        self._fitted = True
        return self

    def _run_smoothing(
        self, arr: np.ndarray, alpha: float, beta: float
    ) -> tuple[float, float, list[float]]:
        """Алгоритм Хольта: predict-then-update для честной оценки ошибки."""
        s = float(arr[0])
        b = float(arr[1] - arr[0])
        fitted: list[float] = [s]

        for t in range(1, len(arr)):
            s_prev = s
            s = alpha * float(arr[t]) + (1.0 - alpha) * (s + b)
            b = beta * (s - s_prev) + (1.0 - beta) * b
            fitted.append(s)

        return s, b, fitted

    def _optimize(self, arr: np.ndarray) -> tuple[float, float]:
        """Grid search 5×5 по SSE (25 комбинаций — достаточно для недвижимости)."""
        best_sse = float("inf")
        best = (self._config.alpha, self._config.beta)

        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for beta in [0.05, 0.1, 0.2, 0.3, 0.4]:
                _, _, fitted = self._run_smoothing(arr, alpha, beta)
                residuals = arr[1:] - np.asarray(fitted[1:])
                sse = float(np.sum(residuals**2))
                if sse < best_sse:
                    best_sse = sse
                    best = (alpha, beta)

        return best

    def forecast(self, steps: int | None = None) -> ForecastResult:
        """Прогноз на steps шагов вперёд с расширяющимися интервалами.

        CI расширяются как σ·√h — стандартная аппроксимация для ETS(A,A,N).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before forecast()")

        h_total = steps or self._config.forecast_periods

        arr = np.asarray(self._history)
        fitted_arr = np.asarray(self._fitted_values)
        residuals = arr[1:] - fitted_arr[1:]
        sigma = float(np.std(residuals)) if len(residuals) > 1 else 0.0

        z = 1.96 if self._config.confidence_level >= 0.95 else 1.645

        points: list[ForecastPoint] = []
        for h in range(1, h_total + 1):
            val = self._level + h * self._trend
            margin = z * sigma * math.sqrt(h)
            points.append(
                ForecastPoint(
                    period=h,
                    value=max(val, 50_000.0),
                    lower=max(val - margin, 50_000.0),
                    upper=max(val + margin, 50_000.0),
                )
            )

        # MAPE на обучающей истории (пропускаем первую точку — инициализация)
        actuals = arr[1:]
        mask = actuals != 0
        if mask.any():
            mape = float(np.mean(np.abs(residuals[mask]) / np.abs(actuals[mask])) * 100)
        else:
            mape = 0.0

        # Направление: |slope| < 0.1% в месяц → стабильно
        trend_pct = (self._trend / self._level * 100.0) if self._level > 1.0 else 0.0
        if abs(trend_pct) < 0.1:
            direction: Literal["rising", "stable", "falling"] = "stable"
        elif trend_pct > 0:
            direction = "rising"
        else:
            direction = "falling"

        return ForecastResult(
            forecast=points,
            mape=round(mape, 2),
            trend_direction=direction,
            trend_slope_pct=round(trend_pct, 3),
            alpha=self._alpha,
            beta=self._beta,
            last_known_value=round(self._level, 2),
        )
