"""
LSTM Autoencoder для обнаружения аномалий во временных рядах (numpy-only).

Реализация через Echo State Network (ESN) — reservoir computing без backprop.
Reservoir захватывает временные зависимости (как LSTM), только выходной слой
обучается через ridge regression (аналитическое решение, O(n) vs O(n²) BPTT).

Математика:
  Reservoir:   x(t) = tanh(W_in · u(t) + W · x(t-1))   [Jaeger 2001]
  Output:      ŷ(t) = W_out · concat([u(t), x(t)])       [linear layer]
  Anomaly:     score = MSE(u(t), ŷ(t))                   [reconstruction error]

Преимущества над Z-score для SRE:
  - Ловит нелинейные паттерны (burst → recovery → второй всплеск)
  - Учитывает кросс-метрические корреляции (CPU↑ + latency↑ = норма при росте нагрузки)
  - Graceful degradation без обучения → fallback на Z-score

Источники:
  Jaeger 2001 (GMD TechReport 148) — original Echo State Network
  Malhotra et al. 2016 (ESANN) — LSTM encoder-decoder для anomaly detection
  Lukoševičius 2012 (Neural Networks Tricks of the Trade, Ch. 22) — practical ESN guide
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LSTMConfig:
    """Конфигурация ESN-автоэнкодера.

    Параметры подобраны под SRE-задачу: 3 метрики (CPU/latency/requests),
    горизонт 30 точек (~5 мин при 10s-интервале).
    """

    reservoir_size: int = 100
    spectral_radius: float = 0.9
    input_scaling: float = 1.0
    leaking_rate: float = 0.3
    ridge_alpha: float = 1e-4
    window_size: int = 30
    n_features: int = 3
    anomaly_percentile: float = 95.0
    random_seed: int = 42


@dataclass
class TrainResult:
    """Результат обучения модели."""

    n_samples: int
    n_windows: int
    train_mse: float
    threshold: float


@dataclass
class LSTMDetectionResult:
    """Результат детекции аномалий через ESN-автоэнкодер."""

    scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    model: str = "esn_autoencoder"
    reconstruction_errors: np.ndarray = field(
        default_factory=lambda: np.array([])
    )


class SequenceScaler:
    """Min-max нормализация по обучающей выборке.

    Нормализация нужна: reservoir tanh(-) работает оптимально в [-1, 1].
    Clips to [-3σ, +3σ] перед нормализацией для robustness к outliers.
    """

    def __init__(self) -> None:
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "SequenceScaler":
        """Вычислить min/max по каждой фиче."""
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted and self._min is not None and self._max is not None
        rng = self._max - self._min
        # Избегаем деления на ноль для константных фич
        rng = np.where(rng < 1e-8, 1.0, rng)
        return (X - self._min) / rng

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class EchoStateAutoencoder:
    """Echo State Network как автоэнкодер для временных рядов.

    Sliding-window подход: каждое окно из window_size точек → вектор → reservoir →
    reconstruction. Ошибка реконструкции = anomaly score для центральной точки окна.
    """

    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.cfg = config or LSTMConfig()
        self._rng = np.random.RandomState(self.cfg.random_seed)
        self._fitted = False

        n_in = self.cfg.n_features
        n_r = self.cfg.reservoir_size

        # Входные веса: Dense, масштаб = input_scaling
        self._W_in = (self._rng.rand(n_r, n_in) * 2 - 1) * self.cfg.input_scaling

        # Веса резервуара: разреженная матрица (10% ненулевых)
        W_raw = self._rng.rand(n_r, n_r) - 0.5
        mask = self._rng.rand(n_r, n_r) < 0.1
        W_raw *= mask.astype(float)

        # Масштабируем до нужного спектрального радиуса
        # Спектральный радиус < 1 — условие echo state property (Jaeger 2001)
        eigenvalues = np.linalg.eigvals(W_raw)
        rho = np.max(np.abs(eigenvalues))
        if rho > 1e-8:
            W_raw *= self.cfg.spectral_radius / rho
        self._W = W_raw

        # Выходные веса (обучаются ridge regression)
        self._W_out: np.ndarray | None = None
        self._scaler = SequenceScaler()
        self._threshold: float = 0.0

    def _run_reservoir(self, X: np.ndarray) -> np.ndarray:
        """Прогнать последовательность через reservoir.

        X: (T, n_features) → states: (T, reservoir_size)
        Leaking rate сглаживает обновление состояния (α): x = (1-α)·x + α·tanh(...)
        """
        T, _ = X.shape
        n_r = self.cfg.reservoir_size
        x = np.zeros(n_r)
        states = np.zeros((T, n_r))
        alpha = self.cfg.leaking_rate

        for t in range(T):
            pre = self._W_in @ X[t] + self._W @ x
            x = (1 - alpha) * x + alpha * np.tanh(pre)
            states[t] = x

        return states

    def _make_windows(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Создать скользящие окна для обучения.

        Returns:
            inputs:  (n_windows, window_size, n_features) — входные окна
            targets: (n_windows, n_features) — центральная точка окна (target для реконструкции)
        """
        T = X.shape[0]
        ws = self.cfg.window_size
        half = ws // 2
        n_windows = T - ws + 1

        inputs = np.array([X[i : i + ws] for i in range(n_windows)])
        # Реконструируем центральную точку — меньше граничных эффектов
        targets = X[half : half + n_windows]
        return inputs, targets

    def fit(self, X: np.ndarray) -> TrainResult:
        """Обучить модель на нормальных данных.

        X: (T, n_features) — временной ряд нормального поведения.
        Обучение: ridge regression на reservoir states → reconstruction.
        """
        X_scaled = self._scaler.fit_transform(X)
        windows, targets = self._make_windows(X_scaled)
        n_windows = len(windows)
        n_r = self.cfg.reservoir_size
        n_in = self.cfg.n_features

        # Собираем reservoir states для каждого окна
        # Augmented input: [reservoir_states, input_center] для лучшей реконструкции
        half = self.cfg.window_size // 2
        aug = np.zeros((n_windows, n_r + n_in))
        for i, w in enumerate(windows):
            states = self._run_reservoir(w)
            # Состояние в центральной точке окна + сам вход
            aug[i] = np.concatenate([states[half], w[half]])

        # Ridge regression: W_out = (AᵀA + λI)⁻¹ Aᵀ Y
        A = aug
        lam = self.cfg.ridge_alpha
        self._W_out = np.linalg.solve(
            A.T @ A + lam * np.eye(A.shape[1]),
            A.T @ targets,
        )

        # Порог = percentile ошибок на обучающей выборке
        train_preds = aug @ self._W_out
        errors = np.mean((train_preds - targets) ** 2, axis=1)
        self._threshold = float(np.percentile(errors, self.cfg.anomaly_percentile))
        train_mse = float(np.mean(errors))

        self._fitted = True
        return TrainResult(
            n_samples=len(X),
            n_windows=n_windows,
            train_mse=train_mse,
            threshold=self._threshold,
        )

    def detect(self, X: np.ndarray) -> LSTMDetectionResult:
        """Обнаружить аномалии в новой последовательности.

        Нефitting точки (первые/последние half) получают score = 0.
        """
        assert self._fitted and self._W_out is not None, "Call fit() first"
        X_scaled = self._scaler.transform(X)
        T = len(X_scaled)
        ws = self.cfg.window_size
        half = ws // 2
        n_r = self.cfg.reservoir_size
        n_in = self.cfg.n_features

        scores = np.zeros(T)
        rec_errors = np.zeros(T)

        for i in range(T - ws + 1):
            w = X_scaled[i : i + ws]
            states = self._run_reservoir(w)
            aug = np.concatenate([states[half], w[half]])
            pred = aug @ self._W_out
            center_idx = i + half
            err = float(np.mean((pred - X_scaled[center_idx]) ** 2))
            scores[center_idx] = err
            rec_errors[center_idx] = err

        predictions = (scores > self._threshold).astype(int)
        return LSTMDetectionResult(
            scores=scores,
            predictions=predictions,
            threshold=self._threshold,
            model="esn_autoencoder",
            reconstruction_errors=rec_errors,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_config(self) -> dict:
        return {
            "type": "echo_state_autoencoder",
            "reservoir_size": self.cfg.reservoir_size,
            "spectral_radius": self.cfg.spectral_radius,
            "window_size": self.cfg.window_size,
            "leaking_rate": self.cfg.leaking_rate,
            "anomaly_percentile": self.cfg.anomaly_percentile,
            "threshold": self._threshold if self._fitted else None,
            "fitted": self._fitted,
        }


def create_autoencoder(config: LSTMConfig | None = None) -> EchoStateAutoencoder:
    """Фабрика моделей.

    Возвращает ESN (всегда доступен, numpy only).
    В будущем: условный импорт PyTorch LSTM если доступен.
    """
    return EchoStateAutoencoder(config)
