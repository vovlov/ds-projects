"""
MMD (Maximum Mean Discrepancy) детектор дрейфа данных.

Зачем MMD вместо PSI? PSI работает попеременно (1D), а MMD — многомерный
непараметрический тест. Для SRE-метрик (cpu, latency, requests) важна
именно совместная структура: дрейф может проявиться в корреляции,
а не в маргинальных распределениях. MMD это ловит.

## Математика

MMD с RBF-ядром (ядро Гаусса):

    k(u, v) = exp(-γ ‖u - v‖²)

    MMD²(P, Q) = E[k(X, X')] - 2·E[k(X, Y)] + E[k(Y, Y')]

    где X ~ P (reference), Y ~ Q (current)

При P = Q → MMD² ≈ 0. Чем сильнее сдвиг, тем больше MMD.

## Порог

Используем перестановочный bootstrap: перемешиваем X и Y, считаем
MMD на перестановках, берём (1-α)-квантиль → порог при уровне α=0.05.
Это эквивалентно тесту гипотезы H0: P = Q.

Источник: Gretton et al. 2012 "A Kernel Two-Sample Test", JMLR.
         Evidently AI v0.5+ реализация с numpy-fallback.
         Alibi-detect v0.12 MMDDrift (опциональная зависимость).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

logger = logging.getLogger(__name__)

# Медианная эвристика: gamma = 1 / (2 * sigma^2), где sigma — медианное
# расстояние между точками. Широко используется как дефолт в kernel methods.
_GAMMA_AUTO = "auto"


def _rbf_kernel_matrix(  # noqa: N803
    X: np.ndarray, Y: np.ndarray, gamma: float  # noqa: N803
) -> np.ndarray:
    """Вычислить матрицу ядра RBF K[i,j] = exp(-γ ‖X[i] - Y[j]‖²).

    Args:
        X: Матрица (n, d). Заглавная буква — математическое соглашение.
        Y: Матрица (m, d). Заглавная буква — математическое соглашение.
        gamma: Параметр ядра Гаусса.

    Returns:
        Матрица ядра (n, m).
    """
    # ||X[i] - Y[j]||^2 = ||X[i]||^2 + ||Y[j]||^2 - 2 X[i]·Y[j]
    # Это быстрее чем явный цикл по парам
    XX = np.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    YY = np.sum(Y**2, axis=1, keepdims=True)  # (m, 1)
    sq_dists = XX + YY.T - 2 * X @ Y.T  # (n, m)
    return np.exp(-gamma * sq_dists)


def _median_heuristic_gamma(  # noqa: N803
    X: np.ndarray, Y: np.ndarray, subsample: int = 500  # noqa: N803
) -> float:
    """Выбрать gamma по медианной эвристике Гретона.

    Берём случайную подвыборку (до subsample точек) для вычисления
    медианного попарного расстояния — это O(n²) операция, дорого на больших n.

    Args:
        X: Первая выборка (n, d). Заглавная буква — математическое соглашение.
        Y: Вторая выборка (m, d). Заглавная буква — математическое соглашение.
        subsample: Максимальный размер подвыборки для эффективности.

    Returns:
        gamma = 1 / (2 * median_dist²).
    """
    rng = np.random.RandomState(0)
    combined = np.vstack([X, Y])
    n = len(combined)

    if n > subsample:
        idx = rng.choice(n, subsample, replace=False)
        combined = combined[idx]

    # Попарные квадратичные расстояния (верхний треугольник)
    diffs = combined[:, None, :] - combined[None, :, :]  # (n, n, d)
    sq_dists = np.sum(diffs**2, axis=-1)  # (n, n)

    # Берём только верхний треугольник (без диагонали)
    upper_tri = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median_sq = np.median(upper_tri)

    # Защита от нулевого расстояния (все точки одинаковые)
    if median_sq < 1e-10:
        return 1.0

    return float(1.0 / (2.0 * median_sq))


def compute_mmd_rbf(  # noqa: N803
    X: np.ndarray,  # noqa: N803
    Y: np.ndarray,  # noqa: N803
    gamma: float | str = _GAMMA_AUTO,
) -> tuple[float, float]:
    """Вычислить MMD² между выборками X и Y с RBF-ядром.

    MMD² = (1/n²) Σ k(xᵢ, xⱼ) - (2/nm) Σ k(xᵢ, yⱼ) + (1/m²) Σ k(yᵢ, yⱼ)

    Unbiased estimator (V-statistic для скорости, достаточно для threshold-теста).

    Args:
        X: Reference samples (n, d) или (n,) для 1D.
        Y: Current samples (m, d) или (m,) для 1D.
        gamma: Параметр RBF-ядра. "auto" = медианная эвристика.

    Returns:
        (mmd_squared, gamma_used) — MMD² значение и использованный gamma.
    """
    X = np.atleast_2d(X).astype(float)
    Y = np.atleast_2d(Y).astype(float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Транспонируем если форма (1, n) вместо (n, 1)
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    if Y.shape[0] == 1 and Y.shape[1] > 1:
        Y = Y.T

    if isinstance(gamma, str) and gamma == _GAMMA_AUTO:
        gamma = _median_heuristic_gamma(X, Y)

    K_XX = _rbf_kernel_matrix(X, X, gamma)
    K_YY = _rbf_kernel_matrix(Y, Y, gamma)
    K_XY = _rbf_kernel_matrix(X, Y, gamma)

    # V-statistic: включает диагональ (k(x,x)=1), немного смещена, но быстра
    mmd2 = K_XX.mean() - 2 * K_XY.mean() + K_YY.mean()

    # Теоретически MMD² ≥ 0, но числовые ошибки могут дать малое отрицательное
    mmd2 = max(float(mmd2), 0.0)
    return mmd2, float(gamma)


def bootstrap_mmd_threshold(  # noqa: N803
    X: np.ndarray,  # noqa: N803
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    gamma: float | str = _GAMMA_AUTO,
) -> tuple[float, float]:
    """Оценить порог MMD² через перестановочный bootstrap.

    При H0 (P = Q) оба набора выборочно из одного распределения.
    Перемешиваем X+X (т.е. одну выборку) n_bootstrap раз и считаем
    MMD между двумя половинами → эмпирическое null-распределение.

    Порог = (1-α)-квантиль null-распределения.

    Args:
        X: Эталонные данные (n, d).
        n_bootstrap: Число перестановок для bootstrap.
        alpha: Уровень значимости (дефолт 0.05 = 95% confidence).
        gamma: Параметр ядра ("auto" для медианной эвристики).

    Returns:
        (threshold, gamma_used).
    """
    X = np.atleast_2d(X).astype(float)
    if X.shape[0] == 1:
        X = X.T

    n = len(X)
    half = n // 2

    # Предвычисляем gamma один раз
    if isinstance(gamma, str):
        gamma = _median_heuristic_gamma(X, X)

    rng = np.random.RandomState(42)
    null_mmds = []

    for _ in range(n_bootstrap):
        perm = rng.permutation(n)
        X_perm = X[perm]
        mmd2, _ = compute_mmd_rbf(X_perm[:half], X_perm[half:], gamma=gamma)
        null_mmds.append(mmd2)

    threshold = float(np.quantile(null_mmds, 1.0 - alpha))
    return threshold, float(gamma)


@dataclass
class DriftResult:
    """Результат MMD drift detection.

    Attributes:
        mmd_statistic: Значение MMD² между reference и current.
        threshold: Порог (выше = дрейф обнаружен).
        is_drift: True если mmd_statistic > threshold.
        gamma: Использованный параметр RBF-ядра.
        p_value: Приближённый p-value (доля bootstrap-MMD > mmd_statistic).
        reference_size: Размер эталонной выборки.
        current_size: Размер текущей выборки.
        features: Список мониторируемых признаков.
        audit_id: Уникальный UUID для аудит-трейла (EU AI Act compliance).
        timestamp: ISO 8601 временная метка детекции.
        reason: Человекочитаемое объяснение решения.
    """

    mmd_statistic: float
    threshold: float
    is_drift: bool
    gamma: float
    p_value: float
    reference_size: int
    current_size: int
    features: list[str] = field(default_factory=list)
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    reason: str = ""


class MMDDriftDetector:
    """Детектор дрейфа данных на основе Maximum Mean Discrepancy.

    Алгоритм:
    1. При инициализации: сохраняем reference data, считаем bootstrap-порог.
    2. При вызове detect(): считаем MMD(reference, current).
    3. Если MMD > threshold → drift detected.

    Преимущества перед PSI:
    - Многомерный: ловит дрейф в совместном распределении (cpu×latency×requests)
    - Непараметрический: не нужна биннинг
    - Теоретически обоснован (Gretton et al. 2012)

    Пример:
        detector = MMDDriftDetector(reference_data)
        result = detector.detect(current_data)
        if result.is_drift:
            trigger_retraining()
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        features: list[str] | None = None,
        gamma: float | str = _GAMMA_AUTO,
        alpha: float = 0.05,
        n_bootstrap: int = 200,
    ) -> None:
        """Инициализировать детектор с эталонными данными.

        Args:
            reference_data: Эталонная выборка (n, d) или (n,) для 1D.
                           Сохраняется как эталон для последующих сравнений.
            features: Имена признаков (для аудит-лога). Если None — f0, f1, ...
            gamma: Параметр RBF-ядра. "auto" = медианная эвристика Гретона.
            alpha: Уровень значимости для bootstrap-порога (дефолт 0.05).
            n_bootstrap: Число итераций bootstrap для порога.
        """
        self.reference = np.atleast_2d(reference_data).astype(float)
        if self.reference.shape[0] == 1:
            self.reference = self.reference.T

        n_features = self.reference.shape[1] if self.reference.ndim > 1 else 1
        self.features = features or [f"f{i}" for i in range(n_features)]
        self.alpha = alpha

        # Предвычисляем gamma и bootstrap-порог один раз при инициализации.
        # Это дорого (O(n_bootstrap * n²)), но делается один раз — при деплое.
        logger.info(
            "MMDDriftDetector: computing bootstrap threshold (n=%d, n_bootstrap=%d, alpha=%.2f)...",
            len(self.reference),
            n_bootstrap,
            alpha,
        )
        self.threshold, self.gamma = bootstrap_mmd_threshold(
            self.reference, n_bootstrap=n_bootstrap, alpha=alpha, gamma=gamma
        )
        logger.info(
            "MMDDriftDetector ready: gamma=%.6f, threshold=%.6f",
            self.gamma,
            self.threshold,
        )

        # Кэшируем null-распределение для p-value
        self._null_mmds = self._compute_null_distribution(n_bootstrap)

    def _compute_null_distribution(self, n_bootstrap: int) -> list[float]:
        """Вычислить null-распределение MMD для p-value."""
        n = len(self.reference)
        half = n // 2
        rng = np.random.RandomState(42)
        null = []
        for _ in range(n_bootstrap):
            perm = rng.permutation(n)
            X_perm = self.reference[perm]
            mmd2, _ = compute_mmd_rbf(X_perm[:half], X_perm[half:], gamma=self.gamma)
            null.append(mmd2)
        return null

    def detect(
        self,
        current_data: np.ndarray,
    ) -> DriftResult:
        """Обнаружить дрейф между reference и current.

        Args:
            current_data: Текущие данные (m, d) или (m,) для 1D.

        Returns:
            DriftResult с флагом is_drift, MMD-статистикой и аудит-метаданными.
        """
        current = np.atleast_2d(current_data).astype(float)
        if current.shape[0] == 1:
            current = current.T

        mmd2, _ = compute_mmd_rbf(self.reference, current, gamma=self.gamma)

        # P-value: доля null-MMD >= наблюдаемого (приближённая permutation p-value)
        if self._null_mmds:
            p_value = float(np.mean([v >= mmd2 for v in self._null_mmds]))
        else:
            p_value = 1.0 if mmd2 <= self.threshold else 0.0

        is_drift = mmd2 > self.threshold

        if is_drift:
            reason = (
                f"DRIFT DETECTED: MMD²={mmd2:.6f} > threshold={self.threshold:.6f} "
                f"(p-value={p_value:.4f}, alpha={self.alpha})"
            )
        else:
            reason = (
                f"No drift: MMD²={mmd2:.6f} <= threshold={self.threshold:.6f} "
                f"(p-value={p_value:.4f})"
            )

        result = DriftResult(
            mmd_statistic=mmd2,
            threshold=self.threshold,
            is_drift=is_drift,
            gamma=self.gamma,
            p_value=p_value,
            reference_size=len(self.reference),
            current_size=len(current),
            features=self.features,
            reason=reason,
        )

        logger.info("MMD drift check: %s", reason)
        return result
