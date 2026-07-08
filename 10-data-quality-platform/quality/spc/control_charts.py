"""
Контрольные карты Шухарта (SPC) для мониторинга метрик качества данных.
Shewhart control charts (SPC) for data quality metric monitoring.

Стандарт: Shewhart 1931 "Economic Control of Quality of Manufactured Product".
Правила: Western Electric Handbook (WECO) 1956 — 4 правила детекции паттернов.

Ключевое различие с CUSUM (Project 05):
- CUSUM: детекция персистентного сдвига в стриминговых временных рядах метрик.
- SPC (этот модуль): мониторинг метрик качества данных по batch-запускам пайплайна,
  где каждая точка = один запуск ETL/ML пайплайна (null rate, mean, std и т.д.).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# ---------------------------------------------------------------------------
# Типы нарушений / Violation types
# ---------------------------------------------------------------------------


class ViolationType(StrEnum):
    """Western Electric Rules (WECO 1956) — 4 типа нарушений."""

    NONE = "none"
    RULE_1_BEYOND_3SIGMA = "rule_1_beyond_3sigma"
    RULE_2_TWO_OF_THREE_2SIGMA = "rule_2_two_of_three_2sigma"
    RULE_3_FOUR_OF_FIVE_1SIGMA = "rule_3_four_of_five_1sigma"
    RULE_4_EIGHT_CONSECUTIVE = "rule_4_eight_consecutive"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SPCConfig:
    """Конфигурация контрольной карты / Control chart configuration."""

    n_sigma: float = 3.0
    window_size: int = 50
    min_calibration_samples: int = 10


@dataclass
class SPCCalibrationResult:
    """Результат калибровки карты / Chart calibration result."""

    center_line: float
    sigma: float
    ucl: float
    lcl: float
    n_samples: int
    is_calibrated: bool = True

    def to_dict(self) -> dict:
        return {
            "center_line": self.center_line,
            "sigma": self.sigma,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "n_samples": self.n_samples,
            "is_calibrated": self.is_calibrated,
        }


@dataclass
class ControlChartResult:
    """Результат добавления одной точки на карту / Single-point chart update result."""

    value: float
    center_line: float
    ucl: float
    lcl: float
    ucl_2sigma: float
    lcl_2sigma: float
    ucl_1sigma: float
    lcl_1sigma: float
    z_score: float
    violation: ViolationType
    is_out_of_control: bool
    n_points: int

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "center_line": self.center_line,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "ucl_2sigma": self.ucl_2sigma,
            "lcl_2sigma": self.lcl_2sigma,
            "ucl_1sigma": self.ucl_1sigma,
            "lcl_1sigma": self.lcl_1sigma,
            "z_score": self.z_score,
            "violation": str(self.violation),
            "is_out_of_control": self.is_out_of_control,
            "n_points": self.n_points,
        }


@dataclass
class SPCState:
    """Текущее состояние контрольной карты / Current chart state."""

    is_calibrated: bool
    center_line: float | None
    sigma: float | None
    ucl: float | None
    lcl: float | None
    n_calibration: int
    n_updates: int
    n_violations: int

    def to_dict(self) -> dict:
        return {
            "is_calibrated": self.is_calibrated,
            "center_line": self.center_line,
            "sigma": self.sigma,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "n_calibration": self.n_calibration,
            "n_updates": self.n_updates,
            "n_violations": self.n_violations,
        }


# ---------------------------------------------------------------------------
# Основной класс / Main class
# ---------------------------------------------------------------------------


class ShewhartChart:
    """
    Контрольная карта Шухарта (X-chart) для непрерывных метрик качества данных.
    Shewhart X-chart for continuous data quality metrics.

    Workflow:
        chart = ShewhartChart()
        chart.calibrate(normal_run_values)  # fit center ± limits
        result = chart.update(new_run_value)  # check each new pipeline run
        result = chart.detect_batch(values)   # check a batch of runs

    Western Electric Rules (WECO 1956):
        Rule 1: 1 point beyond ±3σ         → immediate out-of-control
        Rule 2: 2 of 3 consecutive > ±2σ   → trend toward limit
        Rule 3: 4 of 5 consecutive > ±1σ   → systematic bias
        Rule 4: 8 consecutive on same side  → sustained shift

    Правила применяются к скользящему буферу последних window_size точек.
    Rules apply to a rolling buffer of the last window_size points.
    """

    def __init__(self, config: SPCConfig | None = None) -> None:
        self._config = config or SPCConfig()
        self._center_line: float | None = None
        self._sigma: float | None = None
        self._is_calibrated: bool = False
        self._n_calibration: int = 0
        self._n_updates: int = 0
        self._n_violations: int = 0
        self._buffer: deque = deque(maxlen=self._config.window_size)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self, values: list) -> SPCCalibrationResult:
        """
        Оценить центральную линию μ и σ из нормальных данных.
        Estimate center line (mean) and sigma from in-control calibration data.

        Контрольные пределы: UCL = μ + n_sigma·σ, LCL = μ − n_sigma·σ.
        Control limits: UCL = μ + n_sigma·σ, LCL = μ − n_sigma·σ.

        Args:
            values: список значений метрики при нормальной работе пайплайна.
                    Quality metric values from normal pipeline runs.
        """
        arr = np.asarray(values, dtype=float)
        if len(arr) < self._config.min_calibration_samples:
            raise ValueError(
                f"Нужно минимум {self._config.min_calibration_samples} точек для "
                f"калибровки / Need at least {self._config.min_calibration_samples} "
                f"samples for calibration, got {len(arr)}"
            )

        self._center_line = float(np.mean(arr))
        self._sigma = float(np.std(arr, ddof=1))

        # Защита от нулевой дисперсии (константная метрика) /
        # Guard against zero variance for constant-value metrics
        if self._sigma < 1e-10:
            self._sigma = 1e-8

        self._is_calibrated = True
        self._n_calibration = len(arr)
        self._n_updates = 0
        self._n_violations = 0
        self._buffer.clear()

        # Засеять буфер хвостом калибровочных данных для корректной инициализации WER /
        # Seed buffer with calibration tail so WER has context from first update
        for v in arr[-self._config.window_size :]:
            self._buffer.append(float(v))

        ns = self._config.n_sigma
        return SPCCalibrationResult(
            center_line=self._center_line,
            sigma=self._sigma,
            ucl=self._center_line + ns * self._sigma,
            lcl=self._center_line - ns * self._sigma,
            n_samples=len(arr),
        )

    def update(self, value: float) -> ControlChartResult:
        """
        Добавить одну новую точку (запуск пайплайна) и проверить WER.
        Add one new observation (pipeline run) and check WER violations.

        Returns:
            ControlChartResult с позицией точки и типом нарушения WER.
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Карта не откалибрована / Chart not calibrated. Call calibrate() first."
            )

        self._buffer.append(float(value))
        self._n_updates += 1

        z = (value - self._center_line) / self._sigma
        violation = self._check_wer()
        is_out = violation != ViolationType.NONE

        if is_out:
            self._n_violations += 1

        cl = self._center_line
        sigma = self._sigma
        ns = self._config.n_sigma

        return ControlChartResult(
            value=float(value),
            center_line=cl,
            ucl=cl + ns * sigma,
            lcl=cl - ns * sigma,
            ucl_2sigma=cl + 2.0 * sigma,
            lcl_2sigma=cl - 2.0 * sigma,
            ucl_1sigma=cl + 1.0 * sigma,
            lcl_1sigma=cl - 1.0 * sigma,
            z_score=float(z),
            violation=violation,
            is_out_of_control=is_out,
            n_points=self._n_updates,
        )

    def detect_batch(self, values: list) -> list:
        """
        Проверить батч значений последовательно с учётом WER-контекста.
        Check a batch of values sequentially, carrying WER context forward.

        Каждое значение обновляет скользящий буфер → правила WER применяются
        к скользящему окну предыдущих + текущего значения.
        """
        if not self._is_calibrated:
            raise RuntimeError(
                "Карта не откалибрована / Chart not calibrated. Call calibrate() first."
            )

        results = []
        for v in values:
            results.append(self.update(float(v)))
        return results

    def get_state(self) -> SPCState:
        """Текущее состояние карты / Current chart state for monitoring."""
        cl = self._center_line
        sigma = self._sigma
        ns = self._config.n_sigma

        return SPCState(
            is_calibrated=self._is_calibrated,
            center_line=cl,
            sigma=sigma,
            ucl=(cl + ns * sigma) if self._is_calibrated else None,
            lcl=(cl - ns * sigma) if self._is_calibrated else None,
            n_calibration=self._n_calibration,
            n_updates=self._n_updates,
            n_violations=self._n_violations,
        )

    def reset(self) -> None:
        """Сбросить карту / Reset chart to uncalibrated state."""
        self._center_line = None
        self._sigma = None
        self._is_calibrated = False
        self._n_calibration = 0
        self._n_updates = 0
        self._n_violations = 0
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Внутренние методы / Private methods
    # ------------------------------------------------------------------

    def _check_wer(self) -> ViolationType:
        """
        Применить Western Electric Rules к текущему скользящему буферу.
        Apply Western Electric Rules to current rolling buffer.

        Правила проверяются в порядке убывания серьёзности /
        Rules checked in descending severity order (Rule 1 → Rule 4).
        """
        buf = list(self._buffer)
        if not buf or self._center_line is None:
            return ViolationType.NONE

        z_scores = [(v - self._center_line) / self._sigma for v in buf]
        last = z_scores[-1]

        # Rule 1: последняя точка за пределами ±3σ /
        # Rule 1: last point beyond ±n_sigma
        if abs(last) > self._config.n_sigma:
            return ViolationType.RULE_1_BEYOND_3SIGMA

        # Rule 2: 2 из 3 последних за пределами ±2σ с одной стороны /
        # Rule 2: 2 of 3 consecutive beyond ±2σ on same side
        if len(z_scores) >= 3:
            last3 = z_scores[-3:]
            if sum(1 for z in last3 if z > 2.0) >= 2:
                return ViolationType.RULE_2_TWO_OF_THREE_2SIGMA
            if sum(1 for z in last3 if z < -2.0) >= 2:
                return ViolationType.RULE_2_TWO_OF_THREE_2SIGMA

        # Rule 3: 4 из 5 последних за пределами ±1σ с одной стороны /
        # Rule 3: 4 of 5 consecutive beyond ±1σ on same side
        if len(z_scores) >= 5:
            last5 = z_scores[-5:]
            if sum(1 for z in last5 if z > 1.0) >= 4:
                return ViolationType.RULE_3_FOUR_OF_FIVE_1SIGMA
            if sum(1 for z in last5 if z < -1.0) >= 4:
                return ViolationType.RULE_3_FOUR_OF_FIVE_1SIGMA

        # Rule 4: 8 последних точек с одной стороны от центральной линии /
        # Rule 4: 8 consecutive points on same side of center line
        if len(z_scores) >= 8:
            last8 = z_scores[-8:]
            if all(z > 0 for z in last8) or all(z < 0 for z in last8):
                return ViolationType.RULE_4_EIGHT_CONSECUTIVE

        return ViolationType.NONE
