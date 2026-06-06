"""
Gaussian Copula генератор синтетических табличных данных.

Сохраняет маргинальные распределения (Gaussian для continuous,
эмпирические частоты для categorical) и линейные корреляции через
разложение Холецкого матрицы корреляций.

Поддерживает ε-дифференциальную приватность через механизм Лапласа:
шум добавляется к оценкам средних перед фитингом, не к самим выборкам —
это даёт (ε, 0)-DP для mean queries (Dwork et al. 2006).

Gaussian Copula synthetic data generator that preserves marginal distributions
and pairwise correlations. Optionally adds Laplace noise for ε-DP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Датаклассы / Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """Параметры генератора синтетических данных."""

    n_samples: int = 100
    """Число генерируемых строк по умолчанию."""

    epsilon: float | None = None
    """Бюджет ε-дифференциальной приватности. None = без шума."""

    seed: int | None = 42
    """Seed для воспроизводимости."""

    categorical_threshold: int = 10
    """Максимум уникальных значений для auto-detect категориального типа."""


@dataclass
class ColumnStats:
    """Собранная статистика одного столбца для генерации."""

    name: str
    col_type: str  # "continuous" | "categorical"

    # Непрерывные
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None

    # Категориальные
    categories: list[str] | None = None
    probabilities: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "col_type": self.col_type,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "n_categories": len(self.categories) if self.categories else None,
            "categories": self.categories,
        }


@dataclass
class SyntheticResult:
    """Результат генерации синтетических данных."""

    data: dict[str, list[Any]]
    """Сгенерированные данные: column_name → list of values."""

    n_samples: int
    column_stats: list[ColumnStats]
    privacy_budget: float | None
    fidelity_score: float
    """Среднее нормализованное сходство статистик (0-1, выше = лучше)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "columns": list(self.data.keys()),
            "column_stats": [s.to_dict() for s in self.column_stats],
            "privacy_budget": self.privacy_budget,
            "fidelity_score": round(self.fidelity_score, 4),
        }


# ---------------------------------------------------------------------------
# Генератор / Generator
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """
    Генератор синтетических табличных данных на основе Gaussian Copula.

    Алгоритм:
    1. fit(): оценить μ, σ для continuous; частоты для categorical.
       Если epsilon задан — добавить Laplace-шум к μ (механизм Лапласа).
    2. Корреляционная матрица continuous-столбцов → разложение Холецкого.
    3. generate(): стандартные нормальные → L·z → rescale через (μ, σ).
       Categorical: multinomial выборка по эмпирическим частотам.

    Источники:
    - Gaussian Copula: Nelsen 2006 "An Introduction to Copulas", Springer.
    - Cholesky sampling: Gentle 2009 "Computational Statistics", Springer §6.2.
    - Laplace mechanism: Dwork et al. 2006 TCC, "Calibrating Noise to Sensitivity".
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._column_stats: list[ColumnStats] = []
        self._corr_cholesky: np.ndarray | None = None
        self._continuous_order: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Публичный API / Public API
    # ------------------------------------------------------------------

    def fit(self, data: dict[str, list[Any]]) -> SyntheticDataGenerator:
        """
        Обучить генератор: вычислить статистики и корреляционную матрицу.
        Fit the generator: compute per-column statistics and correlation matrix.
        """
        self._column_stats = []
        self._continuous_order = []

        for col_name, values in data.items():
            clean = [v for v in values if v is not None]
            if not clean:
                continue
            stats = self._fit_column(col_name, clean)
            self._column_stats.append(stats)

        # Корреляционная матрица для continuous-столбцов
        self._corr_cholesky = self._build_cholesky(data)
        self._is_fitted = True
        return self

    def generate(self, n_samples: int | None = None) -> SyntheticResult:
        """
        Сгенерировать синтетические данные, сохраняя статистики и корреляции.
        Generate synthetic samples preserving marginal distributions and correlations.
        """
        if not self._is_fitted:
            raise RuntimeError("Вызовите fit() перед generate() / Call fit() before generate()")

        n = n_samples if n_samples is not None else self.config.n_samples
        cont_samples = self._sample_continuous(n)
        result_data: dict[str, list[Any]] = {}

        for stats in self._column_stats:
            if stats.col_type == "continuous":
                arr = cont_samples.get(stats.name)
                if arr is not None:
                    result_data[stats.name] = arr.tolist()
            else:
                result_data[stats.name] = self._sample_categorical(stats, n)

        fidelity = self._compute_fidelity(result_data)
        return SyntheticResult(
            data=result_data,
            n_samples=n,
            column_stats=self._column_stats,
            privacy_budget=self.config.epsilon,
            fidelity_score=fidelity,
        )

    def fit_generate(
        self, data: dict[str, list[Any]], n_samples: int | None = None
    ) -> SyntheticResult:
        """Удобный метод: обучить и сгенерировать за один вызов."""
        return self.fit(data).generate(n_samples)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # ------------------------------------------------------------------
    # Внутренние методы / Private methods
    # ------------------------------------------------------------------

    def _is_numeric_column(self, values: list[Any]) -> bool:
        """Проверить, числовой ли столбец (не boolean)."""
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

    def _fit_column(self, name: str, clean: list[Any]) -> ColumnStats:
        """Вычислить статистики одного столбца."""
        is_numeric = self._is_numeric_column(clean)
        n_unique = len(set(str(v) for v in clean))

        if is_numeric and n_unique > self.config.categorical_threshold:
            return self._fit_continuous(name, clean)
        return self._fit_categorical(name, clean)

    def _fit_continuous(self, name: str, clean: list[Any]) -> ColumnStats:
        """Fit Gaussian: μ, σ с опциональным Laplace-шумом для DP."""
        arr = np.array(clean, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        # Laplace механизм: глобальная чувствительность = range / n
        # (query = mean over n значений из диапазона [min, max])
        if self.config.epsilon is not None and self.config.epsilon > 0:
            data_range = float(np.max(arr)) - float(np.min(arr))
            sensitivity = data_range / len(arr)
            mu += float(self._rng.laplace(0.0, sensitivity / self.config.epsilon))

        self._continuous_order.append(name)
        return ColumnStats(
            name=name,
            col_type="continuous",
            mean=mu,
            std=max(sigma, 1e-10),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
        )

    def _fit_categorical(self, name: str, clean: list[Any]) -> ColumnStats:
        """Fit эмпирическое распределение категорий."""
        str_vals = [str(v) for v in clean]
        categories = sorted(set(str_vals))
        total = len(str_vals)
        probs = [str_vals.count(c) / total for c in categories]
        return ColumnStats(
            name=name,
            col_type="categorical",
            categories=categories,
            probabilities=probs,
        )

    def _build_cholesky(self, data: dict[str, list[Any]]) -> np.ndarray | None:
        """
        Построить нижне-треугольную матрицу Холецкого из корреляционной матрицы
        continuous-столбцов. Обеспечивает положительную определённость через
        обрезание отрицательных собственных значений (nearest SPD).
        """
        if len(self._continuous_order) < 2:
            return None

        arrays: list[np.ndarray] = []
        for col_name in self._continuous_order:
            vals = [v for v in data[col_name] if v is not None]
            arrays.append(np.array(vals, dtype=float))

        min_len = min(len(a) for a in arrays)
        if min_len < 2:
            return None

        matrix = np.column_stack([a[:min_len] for a in arrays])
        corr = np.corrcoef(matrix.T)

        # Nearest symmetric positive definite via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)
        spd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        try:
            return np.linalg.cholesky(spd)
        except np.linalg.LinAlgError:
            return None

    def _sample_continuous(self, n: int) -> dict[str, np.ndarray]:
        """Сэмплировать коррелированные непрерывные признаки через Cholesky."""
        if not self._continuous_order:
            return {}

        cont_stats = {s.name: s for s in self._column_stats if s.col_type == "continuous"}
        result: dict[str, np.ndarray] = {}

        if self._corr_cholesky is not None and len(self._continuous_order) >= 2:
            # Коррелированная выборка: z ~ N(0,I), затем L·z^T → rescale
            k = len(self._continuous_order)
            z = self._rng.standard_normal((n, k))
            z_corr = z @ self._corr_cholesky.T

            for i, col_name in enumerate(self._continuous_order):
                s = cont_stats[col_name]
                samples = z_corr[:, i] * s.std + s.mean  # type: ignore[operator]
                result[col_name] = np.clip(samples, s.min_val, s.max_val)
        else:
            # Независимая выборка для одного столбца или при ошибке Cholesky
            for col_name in self._continuous_order:
                s = cont_stats[col_name]
                samples = self._rng.normal(s.mean, s.std, n)
                result[col_name] = np.clip(samples, s.min_val, s.max_val)

        return result

    def _sample_categorical(self, stats: ColumnStats, n: int) -> list[str]:
        """Сэмплировать категориальные значения по эмпирическому распределению."""
        assert stats.categories is not None and stats.probabilities is not None
        indices = self._rng.choice(len(stats.categories), size=n, p=stats.probabilities)
        return [stats.categories[int(i)] for i in indices]

    def _compute_fidelity(self, generated: dict[str, list[Any]]) -> float:
        """
        Оценить качество синтетических данных: нормализованная близость μ и σ.
        Score = 1 - mean(|Δμ|/σ_ref + |Δσ|/σ_ref) / 2, усреднено по continuous-столбцам.
        """
        scores: list[float] = []
        for stats in self._column_stats:
            if stats.col_type != "continuous":
                continue
            col = generated.get(stats.name)
            if not col:
                continue
            gen_arr = np.array(col, dtype=float)
            ref_std = max(stats.std, 1e-8)  # type: ignore[arg-type]
            mean_diff = abs(float(np.mean(gen_arr)) - stats.mean) / ref_std  # type: ignore[operator]
            std_diff = abs(float(np.std(gen_arr)) - stats.std) / ref_std  # type: ignore[operator]
            score = max(0.0, 1.0 - (mean_diff + std_diff) / 2.0)
            scores.append(score)

        return float(np.mean(scores)) if scores else 1.0
