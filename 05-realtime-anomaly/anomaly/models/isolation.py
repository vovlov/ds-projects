"""
Isolation Forest anomaly detector with per-feature contribution scoring.

Isolation Forest isolates anomalies by random recursive partitioning —
anomalies require fewer splits (shorter average path length) than normal
observations. This is fundamentally different from density-based methods
and handles high-dimensional data without the curse of dimensionality.

Per-feature contributions via marginal neutralization: each feature is
temporarily replaced by its training-time mean, and the change in anomaly
score measures that feature's contribution to the current anomaly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IsolationConfig:
    """Конфигурация Isolation Forest детектора."""

    n_estimators: int = 100
    contamination: float = 0.05
    max_samples: str | int = "auto"
    random_state: int = 42
    feature_names: list[str] = field(
        default_factory=lambda: ["cpu", "latency", "requests"]
    )


@dataclass
class IsolationResult:
    """Результат анализа одной точки Isolation Forest.

    Attributes:
        is_anomaly: Флаг аномалии (True если точка в top-contamination).
        anomaly_score: Нормализованная оценка [0, 1]; 1 = максимально аномальная.
        path_length: Средняя длина пути изоляции (меньше = аномальнее).
        feature_contributions: Нормализованный вклад каждого признака [0, 1] (сумма = 1).
        top_feature: Признак с наибольшим вкладом в аномалию.
    """

    is_anomaly: bool
    anomaly_score: float
    path_length: float
    feature_contributions: dict[str, float]
    top_feature: str


@dataclass
class IsolationTrainResult:
    """Результат обучения Isolation Forest.

    Attributes:
        n_samples: Количество обучающих наблюдений.
        n_features: Количество признаков.
        contamination: Доля ожидаемых аномалий.
        n_trees: Количество деревьев.
        avg_path_length_normal: Средняя длина пути нормальных точек на train data.
    """

    n_samples: int
    n_features: int
    contamination: float
    n_trees: int
    avg_path_length_normal: float


class IsolationForestDetector:
    """
    Isolation Forest с объяснимостью на уровне признаков.

    Используется как дополнительный метод к Z-score и ESN autoencoder.
    Хорошо обнаруживает точечные аномалии в многомерных метрических данных
    (CPU spike + latency spike — паттерн, который Z-score упускает в unvariate
    режиме).

    Graceful degradation: без sklearn возбуждает RuntimeError с понятным сообщением.
    """

    def __init__(self, config: IsolationConfig | None = None) -> None:
        self.config = config or IsolationConfig()
        self._model = None
        self._train_means: np.ndarray | None = None
        self._score_range: tuple[float, float] | None = None
        self._is_fitted: bool = False
        self._train_result: IsolationTrainResult | None = None

    def fit(self, X: np.ndarray) -> IsolationTrainResult:
        """Обучить Isolation Forest на нормальных (без аномалий) данных.

        Args:
            X: Матрица наблюдений shape (n_samples, n_features).

        Returns:
            IsolationTrainResult с метаданными обучения.

        Raises:
            RuntimeError: Если sklearn не установлен.
        """
        if not is_available():
            raise RuntimeError(
                "scikit-learn not available; install it to use IsolationForestDetector"
            )

        from sklearn.ensemble import IsolationForest

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._train_means = X.mean(axis=0)

        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )
        self._model.fit(X)
        self._is_fitted = True

        # Собираем диапазон оценок на train данных для нормализации к [0, 1].
        # score_samples() возвращает отрицательную среднюю длину пути — более
        # отрицательное значение = короче путь = аномальнее точка.
        train_scores = self._model.score_samples(X)
        self._score_range = (float(train_scores.min()), float(train_scores.max()))

        # Средняя длина пути нормальных точек (тех, что не помечены аномалией)
        normal_mask = self._model.predict(X) == 1
        normal_scores = train_scores[normal_mask] if normal_mask.sum() > 0 else train_scores
        avg_path = float(-normal_scores.mean())

        self._train_result = IsolationTrainResult(
            n_samples=len(X),
            n_features=X.shape[1],
            contamination=self.config.contamination,
            n_trees=self.config.n_estimators,
            avg_path_length_normal=avg_path,
        )
        return self._train_result

    def detect(self, X: np.ndarray) -> list[IsolationResult]:
        """Обнаружить аномалии и объяснить вклад каждого признака.

        Args:
            X: Матрица наблюдений shape (n_samples, n_features).

        Returns:
            Список IsolationResult — по одному на каждую точку.

        Raises:
            RuntimeError: Если модель не обучена.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        raw_scores = self._model.score_samples(X)
        predictions = self._model.predict(X)

        results = []
        for i in range(len(X)):
            anomaly_score = self._normalize_score(float(raw_scores[i]))
            feature_contributions = self._compute_feature_contributions(X[i])
            top_feature = max(feature_contributions, key=lambda k: feature_contributions[k])

            results.append(
                IsolationResult(
                    is_anomaly=bool(predictions[i] == -1),
                    anomaly_score=anomaly_score,
                    path_length=float(-raw_scores[i]),
                    feature_contributions=feature_contributions,
                    top_feature=top_feature,
                )
            )

        return results

    def _normalize_score(self, raw_score: float) -> float:
        """Нормализовать score_samples() к [0, 1] где 1 = максимально аномальное.

        Используем min-max нормализацию по диапазону train данных.
        Точки за пределами train-диапазона clamp к [0, 1].
        """
        if self._score_range is None:
            return float(np.clip(-raw_score, 0.0, 1.0))

        lo, hi = self._score_range
        if hi == lo:
            return 0.5

        # Инвертируем: высокий raw_score = нормальная точка → низкий anomaly_score
        normalized = (hi - raw_score) / (hi - lo)
        return float(np.clip(normalized, 0.0, 1.0))

    def _compute_feature_contributions(self, x: np.ndarray) -> dict[str, float]:
        """Вклад признаков через маргинальную нейтрализацию.

        Каждый признак по очереди заменяется на среднее обучающей выборки.
        Разница в anomaly_score до и после — вклад этого признака.
        Признаки с большим вкладом делают точку более аномальной.
        """
        x = x.flatten()
        means = self._train_means if self._train_means is not None else np.zeros(len(x))
        base_score = float(self._model.score_samples(x.reshape(1, -1))[0])

        contributions: dict[str, float] = {}
        n_features = min(len(self.config.feature_names), len(x))

        for i in range(n_features):
            name = self.config.feature_names[i]
            x_neutralized = x.copy()
            x_neutralized[i] = float(means[i])
            neutral_score = float(self._model.score_samples(x_neutralized.reshape(1, -1))[0])
            # Нейтрализация улучшает score → признак вносит вклад в аномалию
            contributions[name] = max(0.0, neutral_score - base_score)

        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        else:
            # Все признаки равновероятны (равномерный шум или нейтральная точка)
            contributions = {k: 1.0 / n_features for k in contributions}

        return contributions

    @property
    def is_fitted(self) -> bool:
        """Обучена ли модель."""
        return self._is_fitted

    @property
    def train_info(self) -> IsolationTrainResult | None:
        """Метаданные последнего обучения."""
        return self._train_result


def is_available() -> bool:
    """Проверить доступность sklearn для IsolationForest."""
    try:
        from sklearn.ensemble import IsolationForest  # noqa: F401

        return True
    except ImportError:
        return False
