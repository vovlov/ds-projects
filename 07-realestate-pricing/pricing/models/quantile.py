"""
Quantile Regression с Conformalized Quantile Regression (CQR) для интервалов предсказаний.

Вместо наивного MAPE-интервала (+/- X% от точки) используем два подхода:
1. LightGBM quantile loss (pinball loss): прямое предсказание квантилей — [q_0.05, q_0.95]
2. CQR (Romano et al. 2019, NeurIPS): split-conformal калибровка для гарантированного покрытия

Гарантия CQR: P(y ∈ C(x)) ≥ 1-α при любом распределении данных (распределение-free).
Это важно для оценки недвижимости: покупатель/банк могут доверять интервалу статистически.

Схема CQR:
  score(x, y) = max(q_lo(x) - y, y - q_hi(x))  ← ошибка выхода за границы
  q_hat = квантиль scores на calibration set (конечно-выборочная поправка)
  C(x) = [q_lo(x) - q_hat, q_hi(x) + q_hat]

Источники:
- Romano et al. 2019 "Conformalized Quantile Regression" (NeurIPS)
- Angelopoulos & Bates 2022 "Gentle Introduction to Conformal Prediction" (arxiv 2107.07511)
- LightGBM quantile docs: lightgbm.readthedocs.io
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import polars as pl

try:
    from lightgbm import LGBMRegressor

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


def is_available() -> bool:
    """Проверить доступность LightGBM для quantile regression."""
    return _LGBM_AVAILABLE


@dataclass
class QuantileConfig:
    """Конфигурация quantile regression модели."""

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31
    random_state: int = 42


@dataclass
class PredictionInterval:
    """Интервал предсказания для одного объекта недвижимости."""

    point_estimate: float
    lower_90: float
    upper_90: float
    lower_95: float
    upper_95: float
    width_90: float
    width_95: float
    is_cqr_calibrated: bool = False


@dataclass
class CalibrationResult:
    """Coverage-метрики quantile модели на тестовой выборке.

    Основная метрика качества интервалов — эмпирическое покрытие:
    coverage_90 ≈ 0.90 означает, что 90% реальных цен попадают в интервал.
    Интервал слишком широкий → модель консервативна (хорошо для банка).
    Интервал слишком узкий → модель самоуверенна (плохо для покупателя).
    """

    coverage_90: float
    coverage_95: float
    mean_width_90: float
    mean_width_95: float
    n_samples: int
    cqr_adjustment_90: float = 0.0
    cqr_adjustment_95: float = 0.0

    def is_well_calibrated(self, tolerance: float = 0.05) -> bool:
        """Проверить, что покрытие соответствует номинальному уровню ±tolerance."""
        ok_90 = abs(self.coverage_90 - 0.90) <= tolerance
        ok_95 = abs(self.coverage_95 - 0.95) <= tolerance
        return ok_90 and ok_95

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_90": round(self.coverage_90, 4),
            "coverage_95": round(self.coverage_95, 4),
            "mean_width_90": round(self.mean_width_90),
            "mean_width_95": round(self.mean_width_95),
            "n_samples": self.n_samples,
            "cqr_adjustment_90": round(self.cqr_adjustment_90),
            "cqr_adjustment_95": round(self.cqr_adjustment_95),
            "is_well_calibrated": self.is_well_calibrated(),
        }


class QuantileRegressionModel:
    """LightGBM quantile regression с CQR калибровкой.

    Обучает 5 независимых моделей: q=0.025, 0.05, 0.5, 0.95, 0.975.
    После CQR калибровки интервалы гарантированно покрывают ≥ 1-α.

    Использование:
        model = QuantileRegressionModel()
        model.fit(X_train, y_train)
        model.calibrate(X_calib, y_calib)  # CQR шаг
        intervals = model.predict_interval(X_test)
        coverage = model.compute_coverage(X_test, y_test)
    """

    def __init__(self, config: QuantileConfig | None = None) -> None:
        self.config = config or QuantileConfig()
        self._models: dict[float, Any] = {}
        self._cqr_q_hat_90: float = 0.0
        self._cqr_q_hat_95: float = 0.0
        self._is_fitted: bool = False
        self._is_calibrated: bool = False

    def _make_model(self, quantile: float) -> Any:
        cfg = self.config
        return LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            num_leaves=cfg.num_leaves,
            random_state=cfg.random_state,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRegressionModel":
        """Обучить 5 quantile моделей (q=0.025, 0.05, 0.5, 0.95, 0.975)."""
        if not _LGBM_AVAILABLE:
            raise RuntimeError("lightgbm not installed; pip install lightgbm")
        for q in [0.025, 0.05, 0.5, 0.95, 0.975]:
            self._models[q] = self._make_model(q)
            self._models[q].fit(X, y)
        self._is_fitted = True
        return self

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "QuantileRegressionModel":
        """CQR калибровка на holdout-выборке.

        Nonconformity score = max(q_lo(x) - y, y - q_hi(x)) — насколько y
        вышла за предсказанные границы. Порог q_hat = квантиль этих scores.
        Конечно-выборочная поправка Venn-Abers: level = ⌈(n+1)(1-α)⌉/n.
        """
        if not self._is_fitted:
            raise RuntimeError("fit() перед calibrate()")

        n = len(y_calib)

        lo_90 = self._models[0.05].predict(X_calib)
        hi_90 = self._models[0.95].predict(X_calib)
        scores_90 = np.maximum(lo_90 - y_calib, y_calib - hi_90)
        level_90 = min(np.ceil((n + 1) * 0.90) / n, 1.0)
        self._cqr_q_hat_90 = float(np.quantile(scores_90, level_90))

        lo_95 = self._models[0.025].predict(X_calib)
        hi_95 = self._models[0.975].predict(X_calib)
        scores_95 = np.maximum(lo_95 - y_calib, y_calib - hi_95)
        level_95 = min(np.ceil((n + 1) * 0.95) / n, 1.0)
        self._cqr_q_hat_95 = float(np.quantile(scores_95, level_95))

        self._is_calibrated = True
        return self

    def predict_interval(self, X: np.ndarray) -> list[PredictionInterval]:
        """Предсказать интервалы для каждого объекта в X."""
        if not self._is_fitted:
            raise RuntimeError("fit() перед predict_interval()")

        point = self._models[0.5].predict(X)
        lo_90 = self._models[0.05].predict(X) - self._cqr_q_hat_90
        hi_90 = self._models[0.95].predict(X) + self._cqr_q_hat_90
        lo_95 = self._models[0.025].predict(X) - self._cqr_q_hat_95
        hi_95 = self._models[0.975].predict(X) + self._cqr_q_hat_95

        return [
            PredictionInterval(
                point_estimate=float(point[i]),
                lower_90=float(lo_90[i]),
                upper_90=float(hi_90[i]),
                lower_95=float(lo_95[i]),
                upper_95=float(hi_95[i]),
                width_90=float(hi_90[i] - lo_90[i]),
                width_95=float(hi_95[i] - lo_95[i]),
                is_cqr_calibrated=self._is_calibrated,
            )
            for i in range(len(X))
        ]

    def compute_coverage(self, X_test: np.ndarray, y_test: np.ndarray) -> CalibrationResult:
        """Вычислить эмпирическое покрытие на тестовой выборке (calibration plot)."""
        intervals = self.predict_interval(X_test)
        n = len(y_test)

        in_90 = sum(1 for i, iv in enumerate(intervals) if iv.lower_90 <= y_test[i] <= iv.upper_90)
        in_95 = sum(1 for i, iv in enumerate(intervals) if iv.lower_95 <= y_test[i] <= iv.upper_95)

        return CalibrationResult(
            coverage_90=in_90 / n,
            coverage_95=in_95 / n,
            mean_width_90=float(np.mean([iv.width_90 for iv in intervals])),
            mean_width_95=float(np.mean([iv.width_95 for iv in intervals])),
            n_samples=n,
            cqr_adjustment_90=self._cqr_q_hat_90,
            cqr_adjustment_95=self._cqr_q_hat_95,
        )


def train_quantile_model(
    df_train: "pl.DataFrame",
    df_calib: "pl.DataFrame",
    df_test: "pl.DataFrame",
    config: QuantileConfig | None = None,
) -> dict[str, Any]:
    """Полный pipeline: обучение + CQR калибровка + coverage report.

    Использует label encoding (как в train_lightgbm) — та же подготовка данных,
    чтобы предсказания квантилей были совместимы с точечными оценками.

    Returns:
        dict с ключами: model, feature_names, calibration (CalibrationResult).
    """
    from .train import _prepare_xy

    X_train, y_train, feature_names = _prepare_xy(df_train, encode_cats=True)
    X_calib, y_calib, _ = _prepare_xy(df_calib, encode_cats=True)
    X_test, y_test, _ = _prepare_xy(df_test, encode_cats=True)

    model = QuantileRegressionModel(config=config)
    model.fit(X_train, y_train)
    model.calibrate(X_calib, y_calib)

    calibration = model.compute_coverage(X_test, y_test)

    return {
        "model": model,
        "feature_names": feature_names,
        "calibration": calibration,
    }
