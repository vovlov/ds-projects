"""
Мониторинг распределения предсказаний модели / Prediction distribution monitoring.

Отслеживает дрейф выходного распределения (concept drift) отдельно от
дрейфа входных признаков (covariate shift). Обнаруживает три паттерна:
  - Mean shift:    изменение средней уверенности модели
  - Variance shift: рост/снижение энтропии предсказаний
  - Rate shift:    изменение доли положительных предсказаний (для классификации)

Tracks output distribution drift (concept drift) independently from
input feature drift (covariate shift). Detects three patterns:
  - Mean shift:     change in model's average confidence
  - Variance shift: entropy growth/collapse
  - Rate shift:     positive prediction rate change (classification)

Algorithm:
  1. First N observations → reference window (auto-established on first flush)
  2. Each new flush → PSI on prediction histogram + Welch z-test on mean
  3. Severity: PSI ≥ 0.2 → CRITICAL, PSI ≥ 0.1 → WARNING, else OK

References:
  - Bifet & Gavalda 2007 (ADWIN, adaptive windowing concept)
  - Gama et al. 2014 "Survey on concept drift adaptation" ACM CSUR 46(4)
  - BCBS 2011 Basel II PSI thresholds (adapted for prediction distributions)
  - Sculley et al. 2015 NeurIPS "Hidden Technical Debt in ML Systems"
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PredictionStats:
    """Сводные статистики окна предсказаний / Summary statistics for a prediction window."""

    n: int
    mean: float
    std: float
    min: float
    max: float
    positive_rate: float  # доля предсказаний > 0.5 (для вероятностных моделей)
    hist: list[float]  # нормализованная гистограмма (10 бинов, 0–1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "positive_rate": round(self.positive_rate, 4),
            "histogram_bins": 10,
            "histogram": [round(v, 4) for v in self.hist],
        }


@dataclass
class PredictionDriftResult:
    """Результат сравнения текущего окна с референсным / Drift comparison result."""

    drift_id: str
    timestamp: str
    has_drift: bool
    severity: str  # "ok" | "warning" | "critical"
    psi: float
    z_score_mean: float  # Welch z-test на среднем
    rate_delta: float  # |positive_rate_cur - positive_rate_ref|
    reference_stats: PredictionStats
    current_stats: PredictionStats
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "timestamp": self.timestamp,
            "has_drift": self.has_drift,
            "severity": self.severity,
            "psi": round(self.psi, 4),
            "z_score_mean": round(self.z_score_mean, 4),
            "rate_delta": round(self.rate_delta, 4),
            "reason": self.reason,
            "reference_stats": self.reference_stats.to_dict(),
            "current_stats": self.current_stats.to_dict(),
        }


@dataclass
class MonitorStatus:
    """Текущее состояние монитора / Current monitor state."""

    is_ready: bool  # True когда reference window установлен
    reference_size: int
    current_window_size: int
    window_capacity: int
    total_observed: int
    last_drift_check: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "reference_size": self.reference_size,
            "current_window_size": self.current_window_size,
            "window_capacity": self.window_capacity,
            "total_observed": self.total_observed,
            "last_drift_check": self.last_drift_check,
        }


# ---------------------------------------------------------------------------
# Core monitor
# ---------------------------------------------------------------------------

_N_BINS = 10  # бинов гистограммы для PSI


def _compute_hist(values: np.ndarray, bins: int = _N_BINS) -> np.ndarray:
    """Нормализованная гистограмма на фиксированном диапазоне [0, 1].

    Фиксированный диапазон гарантирует сопоставимость бинов между
    reference и current окнами.
    Fixed range [0, 1] ensures histogram bins align across windows.
    """
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return np.ones(bins) / bins
    return counts / total


def _psi(reference_hist: np.ndarray, current_hist: np.ndarray) -> float:
    """PSI по BCBS Basel II (Population Stability Index).

    PSI = Σ (A_i - E_i) * ln(A_i / E_i)
    где A_i — текущая доля, E_i — референсная доля.

    Laplace сглаживание предотвращает деление на ноль.
    Laplace smoothing prevents division-by-zero.
    """
    eps = 1e-6
    ref = reference_hist + eps
    cur = current_hist + eps
    ref /= ref.sum()
    cur /= cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _welch_z(mu1: float, mu2: float, std1: float, std2: float, n1: int, n2: int) -> float:
    """Welch z-статистика для проверки равенства средних.

    |z| > 1.96 → p < 0.05 (двустороннее, нормальное приближение).
    |z| > 1.96 → p < 0.05 (two-sided normal approximation).
    """
    var = std1**2 / max(n1, 1) + std2**2 / max(n2, 1)
    if var < 1e-12:
        return 0.0
    return float((mu1 - mu2) / var**0.5)


def _compute_stats(values: np.ndarray) -> PredictionStats:
    hist = _compute_hist(values)
    return PredictionStats(
        n=len(values),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        positive_rate=float(np.mean(values > 0.5)),
        hist=hist.tolist(),
    )


class PredictionMonitor:
    """
    Монитор распределения предсказаний модели.
    Monitors model prediction distribution for concept drift.

    Workflow:
      1. observe(predictions) — добавляет предсказания в текущее окно
      2. Первый flush (после достижения min_reference_size) → устанавливает reference
      3. Каждый последующий flush → detect_drift() сравнивает с reference

    Window capacity — скользящее окно: старые предсказания вытесняются
    новыми (FIFO deque). Это позволяет отслеживать актуальное состояние,
    а не усреднять по всей истории.
    """

    def __init__(
        self,
        window_size: int = 1000,
        min_reference_size: int = 200,
        warning_psi: float = 0.1,
        critical_psi: float = 0.2,
    ) -> None:
        self.window_size = window_size
        self.min_reference_size = min_reference_size
        self.warning_psi = warning_psi
        self.critical_psi = critical_psi

        self._current: deque[float] = deque(maxlen=window_size)
        self._reference: np.ndarray | None = None
        self._total_observed: int = 0
        self._last_drift_check: str | None = None

    def observe(self, predictions: list[float] | np.ndarray) -> int:
        """Добавляет предсказания в текущее окно. / Add predictions to window.

        Принимает вероятности [0, 1] или бинарные метки {0, 1}.
        Accepts probabilities [0, 1] or binary labels {0, 1}.
        Returns number of observations added.
        """
        preds = np.asarray(predictions, dtype=float)
        if preds.size == 0:
            return 0
        for v in preds:
            self._current.append(float(v))
        self._total_observed += len(preds)

        # Автоматически устанавливаем reference при первом наполнении
        if self._reference is None and len(self._current) >= self.min_reference_size:
            self._reference = np.array(self._current)

        return len(preds)

    def detect_drift(self) -> PredictionDriftResult:
        """Сравнивает текущее окно с референсным / Compare current vs reference.

        Raises ValueError если reference ещё не установлен.
        Raises ValueError if reference window is not yet established.
        """
        if self._reference is None:
            raise ValueError(
                f"Reference window not established yet. "
                f"Need {self.min_reference_size} observations, "
                f"got {len(self._current)}."
            )
        if len(self._current) < 10:
            raise ValueError(f"Too few current observations ({len(self._current)}). Need ≥ 10.")

        ref_arr = self._reference
        cur_arr = np.array(self._current)

        ref_stats = _compute_stats(ref_arr)
        cur_stats = _compute_stats(cur_arr)

        ref_hist = np.array(ref_stats.hist)
        cur_hist = np.array(cur_stats.hist)
        psi_val = _psi(ref_hist, cur_hist)

        z = _welch_z(
            cur_stats.mean,
            ref_stats.mean,
            cur_stats.std,
            ref_stats.std,
            cur_stats.n,
            ref_stats.n,
        )
        rate_delta = abs(cur_stats.positive_rate - ref_stats.positive_rate)

        # Severity по PSI (BCBS Basel II критерии, адаптированные для предсказаний)
        if psi_val >= self.critical_psi:
            severity = "critical"
            has_drift = True
            reason = f"PSI={psi_val:.3f} ≥ {self.critical_psi} (critical threshold)"
        elif psi_val >= self.warning_psi:
            severity = "warning"
            has_drift = True
            reason = f"PSI={psi_val:.3f} ≥ {self.warning_psi} (warning threshold)"
        else:
            severity = "ok"
            has_drift = False
            reason = f"PSI={psi_val:.3f} < {self.warning_psi} (no drift)"

        # Дополнительный сигнал: сильный z-тест усиливает severity
        if not has_drift and abs(z) > 3.0:
            has_drift = True
            severity = "warning"
            reason += f"; |z|={abs(z):.2f} > 3.0 (mean shift)"

        ts = datetime.now(UTC).isoformat()
        self._last_drift_check = ts

        return PredictionDriftResult(
            drift_id=str(uuid.uuid4()),
            timestamp=ts,
            has_drift=has_drift,
            severity=severity,
            psi=psi_val,
            z_score_mean=z,
            rate_delta=rate_delta,
            reference_stats=ref_stats,
            current_stats=cur_stats,
            reason=reason,
        )

    def get_status(self) -> MonitorStatus:
        """Статус монитора для health-check / Monitor status for health-check."""
        return MonitorStatus(
            is_ready=self._reference is not None,
            reference_size=len(self._reference) if self._reference is not None else 0,
            current_window_size=len(self._current),
            window_capacity=self.window_size,
            total_observed=self._total_observed,
            last_drift_check=self._last_drift_check,
        )

    def reset(self) -> None:
        """Сбросить состояние монитора / Reset monitor state."""
        self._current.clear()
        self._reference = None
        self._total_observed = 0
        self._last_drift_check = None

    def set_reference(self, predictions: list[float] | np.ndarray) -> PredictionStats:
        """Явно установить референсное окно / Explicitly set reference window.

        Полезно при смене версии модели (model champion rotation).
        Useful when rotating model versions (champion/challenger swap).
        """
        ref_arr = np.asarray(predictions, dtype=float)
        if ref_arr.size < 2:
            raise ValueError("Reference window requires ≥ 2 observations.")
        self._reference = ref_arr
        self._current.clear()
        return _compute_stats(ref_arr)


# ---------------------------------------------------------------------------
# Module-level singleton (для API / for API use)
# ---------------------------------------------------------------------------

_monitor: PredictionMonitor | None = None


def get_monitor(
    window_size: int = 1000,
    min_reference_size: int = 200,
) -> PredictionMonitor:
    """Ленивая инициализация singleton-монитора / Lazy singleton initializer."""
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor(
            window_size=window_size,
            min_reference_size=min_reference_size,
        )
    return _monitor


def reset_prediction_monitor() -> None:
    """Сбросить глобальный монитор (для тестовой изоляции) / Reset global monitor."""
    global _monitor
    _monitor = None
