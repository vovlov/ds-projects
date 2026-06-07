"""
Mahalanobis Distance anomaly detector for correlated multivariate metrics.

For SRE metrics (cpu/latency/requests) correlation is the key challenge:
a CPU spike normally causes latency to spike too. Euclidean outlier detection
treats each metric independently and flags the latency spike even when it is
proportional to CPU — a false positive. Mahalanobis distance accounts for the
covariance structure so correlated joint behaviour is treated as normal.

  d(x, μ) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))

Under multivariate normality d² ~ χ²(p), but we use a percentile threshold
computed from training data to stay robust to real-world deviations.

Feature contributions use the same marginal neutralisation as IsolationForest:
temporarily set feature i to its mean and measure how much the distance drops.
This keeps the explanability API consistent across detectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MahalanobisConfig:
    """Конфигурация Mahalanobis детектора."""

    feature_names: list[str] = field(
        default_factory=lambda: ["cpu", "latency", "requests"]
    )
    threshold_percentile: float = 97.5
    regularization: float = 1e-6


@dataclass
class MahalanobisResult:
    """Результат детекции одной точки.

    Attributes:
        is_anomaly: True если расстояние превышает threshold.
        mahalanobis_distance: Расстояние Махаланобиса (в единицах std c учётом корреляций).
        anomaly_score: Нормализованный score [0, 1]; 1 = максимально аномальная.
        threshold: Порог, при котором точка считается аномальной.
        feature_contributions: Вклад каждого признака в аномалию (сумма = 1).
        top_feature: Признак с наибольшим вкладом.
    """

    is_anomaly: bool
    mahalanobis_distance: float
    anomaly_score: float
    threshold: float
    feature_contributions: dict[str, float]
    top_feature: str


@dataclass
class MahalanobisTrainResult:
    """Метаданные обучения детектора.

    Attributes:
        n_samples: Количество обучающих наблюдений.
        n_features: Количество признаков.
        mean: Среднее по каждому признаку.
        condition_number: Число обусловленности ковариационной матрицы.
            Высокое (> 1e6) означает мультиколлинеарность — regularization важна.
        threshold: Вычисленный порог расстояния Махаланобиса.
    """

    n_samples: int
    n_features: int
    mean: list[float]
    condition_number: float
    threshold: float


class MahalanobisDetector:
    """
    Детектор аномалий на основе расстояния Махаланобиса.

    Преимущества над существующими детекторами в платформе:
    - Z-score: унивариатный, игнорирует корреляции между метриками.
    - Isolation Forest: не использует явную модель ковариационной структуры.
    - ESN Autoencoder: требует временной зависимости; плохо на i.i.d. батчах.
    Mahalanobis замечателен для точечных аномалий в коррелированном пространстве:
    CPU=90%, latency=200ms — нормально; CPU=10%, latency=200ms — аномально.

    Pure numpy — нет внешних зависимостей, CI-совместим.
    """

    def __init__(self, config: MahalanobisConfig | None = None) -> None:
        self.config = config or MahalanobisConfig()
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._threshold: float | None = None
        self._train_dist_max: float | None = None
        self._is_fitted: bool = False
        self._train_result: MahalanobisTrainResult | None = None

    def fit(self, X: np.ndarray) -> MahalanobisTrainResult:
        """Оценить μ и Σ по нормальным данным, вычислить порог.

        Args:
            X: Матрица наблюдений shape (n_samples, n_features). Минимум 10 строк.

        Returns:
            MahalanobisTrainResult с метаданными.

        Raises:
            ValueError: Если менее 10 наблюдений или n_features == 1.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        if n < 10:
            raise ValueError(f"Need at least 10 samples, got {n}")

        self._mean = X.mean(axis=0)

        cov = np.cov(X, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Регуляризация ε·I предотвращает сингулярность при мультиколлинеарности.
        # Малый ε не меняет геометрию заметно, но делает обращение численно устойчивым.
        cov_reg = cov + self.config.regularization * np.eye(p)
        condition_number = float(np.linalg.cond(cov_reg))

        self._cov_inv = np.linalg.inv(cov_reg)
        self._is_fitted = True

        train_distances = self._compute_distances(X)

        self._threshold = float(
            np.percentile(train_distances, self.config.threshold_percentile)
        )
        # Максимум train дистанций для нормализации score к [0, 1]
        self._train_dist_max = float(train_distances.max()) if len(train_distances) > 0 else 1.0

        self._train_result = MahalanobisTrainResult(
            n_samples=n,
            n_features=p,
            mean=self._mean.tolist(),
            condition_number=condition_number,
            threshold=self._threshold,
        )
        return self._train_result

    def detect(self, X: np.ndarray) -> list[MahalanobisResult]:
        """Обнаружить аномалии в батче наблюдений.

        Args:
            X: Матрица наблюдений shape (n_samples, n_features).

        Returns:
            Список MahalanobisResult — по одному на строку X.

        Raises:
            RuntimeError: Если модель не обучена.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        distances = self._compute_distances(X)
        results = []
        for i, dist in enumerate(distances):
            anomaly_score = self._normalize_distance(dist)
            contributions = self._marginal_contributions(X[i])
            top_feature = max(contributions, key=lambda k: contributions[k])

            results.append(
                MahalanobisResult(
                    is_anomaly=bool(dist > self._threshold),
                    mahalanobis_distance=float(dist),
                    anomaly_score=anomaly_score,
                    threshold=float(self._threshold),
                    feature_contributions=contributions,
                    top_feature=top_feature,
                )
            )
        return results

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Вычислить расстояния Махаланобиса для всех строк X."""
        assert self._mean is not None and self._cov_inv is not None
        delta = X - self._mean
        # d²_i = delta_i @ Σ⁻¹ @ delta_iᵀ (диагональ произведения)
        sq_distances = np.einsum("ij,jk,ik->i", delta, self._cov_inv, delta)
        # Clip на случай крошечных отрицательных значений из-за floating-point
        return np.sqrt(np.maximum(sq_distances, 0.0))

    def _normalize_distance(self, dist: float) -> float:
        """Нормализовать расстояние к [0, 1] через train-max.

        Точки за пределами train-диапазона clamp к 1.0 (гарантированная аномалия).
        """
        if self._train_dist_max is None or self._train_dist_max == 0:
            return float(np.clip(dist, 0.0, 1.0))
        return float(np.clip(dist / self._train_dist_max, 0.0, 1.0))

    def _marginal_contributions(self, x: np.ndarray) -> dict[str, float]:
        """Вклад признаков через маргинальную нейтрализацию.

        Признак i заменяется на его среднее, и измеряется уменьшение расстояния.
        Большее уменьшение = больший вклад этого признака в аномалию.
        Этот подход консистентен с IsolationForestDetector._compute_feature_contributions().
        """
        x = x.flatten()
        base_dist = self._compute_distances(x.reshape(1, -1))[0]

        contributions: dict[str, float] = {}
        n_features = min(len(self.config.feature_names), len(x))

        for i in range(n_features):
            name = self.config.feature_names[i]
            x_neutral = x.copy()
            x_neutral[i] = float(self._mean[i])  # type: ignore[index]
            neutral_dist = self._compute_distances(x_neutral.reshape(1, -1))[0]
            contributions[name] = max(0.0, base_dist - neutral_dist)

        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        else:
            uniform = 1.0 / n_features if n_features > 0 else 1.0
            contributions = {k: uniform for k in contributions}

        return contributions

    @property
    def is_fitted(self) -> bool:
        """Обучена ли модель."""
        return self._is_fitted

    @property
    def train_info(self) -> MahalanobisTrainResult | None:
        """Метаданные последнего обучения."""
        return self._train_result
