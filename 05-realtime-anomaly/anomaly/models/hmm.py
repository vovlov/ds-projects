"""Hidden Markov Model for anomaly regime detection.

Двухсостояниевая Гауссовская HMM: NORMAL (низкие скоры) и ANOMALY (высокие скоры).

Математическая база:
  Состояния: q_t ∈ {0=NORMAL, 1=ANOMALY}
  Эмиссия:   y_t | q_t=i ~ N(μ_i, σ_i²)
  Переходы:  A[i,j] = P(q_t=j | q_{t-1}=i)
  Начало:    π[i] = P(q_0=i)

Оценка параметров: Baum-Welch EM (Welch 2003, IEEE Signal Processing Magazine).
Декодирование состояний: алгоритм Витерби (лог-домен для численной стабильности).
Порог аномалии: P(ANOMALY | y_{1:T}) > anomaly_threshold.

Преимущества перед point-wise детекторами:
  - Учитывает персистентность состояния: однажды войдя в ANOMALY-режим,
    система вероятно останется в нём (A[1,1] >> A[1,0]).
  - Устойчив к отдельным выбросам: ложный spike не переключает режим,
    если стационарная вероятность перехода низка.
  - Probabilistic output: P(anomaly) вместо binary boolean.

Complementary: используется как postprocessor над скорами Isolation Forest / Kalman /
CUSUM / ESN / STL — reduces false alarm rate в спокойных периодах.

Источники:
  Baum & Petrie 1966 Ann. Math. Stat. 37(6):1554-1563 (HMM foundations).
  Welch 2003 IEEE SP Mag. 20(6) (Baum-Welch tutorial).
  Rabiner 1989 Proc. IEEE 77(2):257-286 (классический tutorial).
  Bilmes 1998 ICSI TR-97-021 (численно стабильный scaled forward-backward).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Config & State dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class HMMConfig:
    """Конфигурация Gaussian HMM детектора.

    n_states: число скрытых состояний (2 = NORMAL + ANOMALY).
    n_iter: максимум итераций Baum-Welch.
    tol: порог сходимости log-likelihood.
    anomaly_threshold: P(ANOMALY | наблюдения) > этого → is_anomaly.
    min_sigma: нижняя граница σ (защита от вырожденных эмиссий).
    """

    n_states: int = 2
    n_iter: int = 100
    tol: float = 1e-4
    anomaly_threshold: float = 0.5
    min_sigma: float = 1e-3


@dataclass
class HMMCalibrationResult:
    """Результат Baum-Welch оценки параметров.

    n_iter_done: сколько итераций выполнено.
    converged: True если |ΔlogL| < tol.
    log_likelihood: финальный log P(Y | θ) на обучающей последовательности.
    transition_matrix: A[i,j] — вероятность перехода i→j.
    means: μ[i] — среднее эмиссии состояния i.
    stds: σ[i] — стандартное отклонение эмиссии состояния i.
    normal_state: индекс состояния с меньшим средним (NORMAL).
    anomaly_state: индекс состояния с большим средним (ANOMALY).
    """

    n_iter_done: int
    converged: bool
    log_likelihood: float
    transition_matrix: list[list[float]]
    means: list[float]
    stds: list[float]
    normal_state: int
    anomaly_state: int


@dataclass
class HMMDecodeResult:
    """Результат Viterbi-декодирования последовательности скоров.

    states: наиболее вероятная последовательность состояний (0/1).
    state_names: ["NORMAL"/"ANOMALY", ...].
    anomaly_probability: P(ANOMALY_t | y_{1:T}) для каждой точки (forward-backward γ).
    predictions: is_anomaly[t] = anomaly_probability[t] > threshold.
    anomaly_indices: индексы аномальных точек.
    n_anomalies: число аномальных точек.
    change_points: индексы смен режима (переход NORMAL↔ANOMALY).
    """

    states: list[int]
    state_names: list[str]
    anomaly_probability: list[float]
    predictions: list[bool]
    anomaly_indices: list[int]
    n_anomalies: int
    change_points: list[int]


@dataclass
class HMMUpdateResult:
    """Результат онлайн-обновления одной точкой.

    value: входное значение.
    current_state: состояние Viterbi на последнем шаге.
    state_name: "NORMAL" или "ANOMALY".
    anomaly_probability: P(ANOMALY | текущий буфер наблюдений).
    is_anomaly: anomaly_probability > threshold.
    n_updates: всего вызовов update() с последней калибровки.
    """

    value: float
    current_state: int
    state_name: str
    anomaly_probability: float
    is_anomaly: bool
    n_updates: int


@dataclass
class HMMState:
    """Внутреннее состояние модели для get_state() / status endpoint."""

    is_calibrated: bool
    n_states: int
    anomaly_threshold: float
    n_calibration: int
    n_updates: int
    transition_matrix: list[list[float]] | None
    means: list[float] | None
    stds: list[float] | None
    normal_state: int | None
    anomaly_state: int | None


# --------------------------------------------------------------------------- #
# Utility: log-domain numerics
# --------------------------------------------------------------------------- #


_NEG_INF = -1e308  # практический -∞ (не float('-inf') для совместимости с numpy)


def _logsumexp(log_probs: np.ndarray) -> float:
    """Numerically stable log(Σ exp(log_probs)).

    Uses the max-subtraction trick: log Σ exp(a_i) = m + log Σ exp(a_i - m).
    """
    m = float(np.max(log_probs))
    if m <= _NEG_INF:
        return _NEG_INF
    return m + float(np.log(np.sum(np.exp(log_probs - m))))


def _log_gaussian(x: float, mu: float, sigma: float) -> float:
    """log N(x; μ, σ) = -0.5 log(2π) - log(σ) - 0.5 ((x-μ)/σ)²."""
    z = (x - mu) / sigma
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * z * z


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #


class HMMDetector:
    """Gaussian HMM для детекции режимов (NORMAL / ANOMALY).

    Принимает последовательность скоров (любой из детекторов: Z-score,
    Isolation Forest, Kalman NIS, CUSUM S⁺, ESN reconstruction error).
    Возвращает вероятностное assignment каждой точки к режиму.

    Параметры HMM:
      A    : (n_states, n_states) матрица переходов
      mu   : (n_states,) средние эмиссий
      sigma: (n_states,) std эмиссий
      pi   : (n_states,) начальные вероятности

    Аномальное состояние = state с большим средним (learned, не hardcoded).
    """

    def __init__(self, config: HMMConfig | None = None) -> None:
        self._cfg = config or HMMConfig()
        # Параметры модели (инициализируются при calibrate)
        self._A: np.ndarray | None = None      # transition matrix [n_states, n_states]
        self._mu: np.ndarray | None = None     # emission means [n_states]
        self._sigma: np.ndarray | None = None  # emission stds [n_states]
        self._pi: np.ndarray | None = None     # initial probabilities [n_states]
        self._normal_state: int = 0
        self._anomaly_state: int = 1
        self._is_calibrated: bool = False
        self._n_calibration: int = 0
        self._n_updates: int = 0
        # Скользящий буфер для онлайн update
        self._buffer: list[float] = []
        self._buffer_maxlen: int = 200

    # ---------------------------------------------------------------------- #
    # Emission helpers
    # ---------------------------------------------------------------------- #

    def _log_emit(self, x: float, state: int) -> float:
        """log P(x | state) для Гауссовской эмиссии."""
        assert self._mu is not None and self._sigma is not None
        return _log_gaussian(x, float(self._mu[state]), float(self._sigma[state]))

    def _log_emit_all(self, obs: np.ndarray) -> np.ndarray:
        """log эмиссионная матрица: shape (T, n_states).

        log_emit[t, i] = log N(obs[t]; μ_i, σ_i).
        """
        assert self._mu is not None and self._sigma is not None
        T = len(obs)
        K = self._cfg.n_states
        log_b = np.empty((T, K))
        for i in range(K):
            z = (obs - self._mu[i]) / self._sigma[i]
            log_b[:, i] = -0.5 * np.log(2 * np.pi) - np.log(self._sigma[i]) - 0.5 * z * z
        return log_b

    # ---------------------------------------------------------------------- #
    # Forward-Backward algorithm (log-domain, scaled)
    # ---------------------------------------------------------------------- #

    def _forward(self, obs: np.ndarray, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        """Log-domain forward algorithm.

        Returns:
            log_alpha: (T, K) log P(y_{1:t}, q_t=i | θ)
            log_likelihood: log P(y_{1:T} | θ)
        """
        assert self._A is not None and self._pi is not None
        T, K = log_b.shape
        log_A = np.log(np.clip(self._A, 1e-300, None))
        log_alpha = np.full((T, K), _NEG_INF)
        log_alpha[0] = np.log(np.clip(self._pi, 1e-300, None)) + log_b[0]
        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_b[t, j]
        ll = _logsumexp(log_alpha[-1])
        return log_alpha, ll

    def _backward(self, obs: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        """Log-domain backward algorithm.

        Returns:
            log_beta: (T, K) log P(y_{t+1:T} | q_t=i, θ)
        """
        assert self._A is not None
        T, K = log_b.shape
        log_A = np.log(np.clip(self._A, 1e-300, None))
        log_beta = np.full((T, K), _NEG_INF)
        log_beta[-1] = 0.0  # log(1) = 0
        for t in range(T - 2, -1, -1):
            for i in range(K):
                log_beta[t, i] = _logsumexp(log_A[i] + log_b[t + 1] + log_beta[t + 1])
        return log_beta

    def _e_step(
        self, obs: np.ndarray, log_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """E-step: compute posteriors γ and ξ.

        γ[t, i]    = P(q_t=i | Y, θ)
        ξ[t, i, j] = P(q_t=i, q_{t+1}=j | Y, θ)
        """
        T, K = log_b.shape
        log_alpha, ll = self._forward(obs, log_b)
        log_beta = self._backward(obs, log_b)

        # γ
        log_gamma = log_alpha + log_beta
        # Normalise каждый временной шаг
        for t in range(T):
            log_gamma[t] -= _logsumexp(log_gamma[t])
        gamma = np.exp(log_gamma)

        # ξ: (T-1, K, K)
        assert self._A is not None
        log_A = np.log(np.clip(self._A, 1e-300, None))
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            log_xi_t = np.full((K, K), _NEG_INF)
            for i in range(K):
                for j in range(K):
                    log_xi_t[i, j] = (
                        log_alpha[t, i]
                        + log_A[i, j]
                        + log_b[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            # Normalise
            log_sum = _logsumexp(log_xi_t.ravel())
            xi[t] = np.exp(log_xi_t - log_sum)

        return gamma, xi, ll

    def _m_step(
        self, obs: np.ndarray, gamma: np.ndarray, xi: np.ndarray
    ) -> None:
        """M-step: обновить параметры θ = (A, μ, σ, π) из достаточных статистик."""
        K = self._cfg.n_states
        assert self._mu is not None and self._sigma is not None

        # π
        self._pi = gamma[0] / (gamma[0].sum() + 1e-300)

        # A: A[i,j] = Σ_t ξ[t,i,j] / Σ_t γ[t,i]
        self._A = (xi.sum(axis=0) + 1e-300) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
        # Renormalise строки
        row_sums = self._A.sum(axis=1, keepdims=True)
        self._A /= np.where(row_sums > 0, row_sums, 1.0)

        # μ, σ
        for i in range(K):
            w = gamma[:, i]
            w_sum = w.sum() + 1e-300
            self._mu[i] = (w * obs).sum() / w_sum
            var = (w * (obs - self._mu[i]) ** 2).sum() / w_sum
            self._sigma[i] = float(np.sqrt(max(var, self._cfg.min_sigma ** 2)))

    # ---------------------------------------------------------------------- #
    # Calibration (Baum-Welch EM)
    # ---------------------------------------------------------------------- #

    def calibrate(
        self,
        scores: list[float],
        anomaly_scores: list[float] | None = None,
    ) -> HMMCalibrationResult:
        """Оценить параметры HMM методом Baum-Welch EM.

        Args:
            scores: обучающая последовательность (обычно нормальные скоры).
            anomaly_scores: опционально — скоры аномальных периодов для
                            лучшей инициализации состояния ANOMALY.

        Raises:
            ValueError: если меньше 10 точек.
        """
        if len(scores) < 10:
            raise ValueError(f"Нужно ≥ 10 точек, получено {len(scores)}")

        obs = np.array(scores, dtype=float)
        K = self._cfg.n_states

        # Инициализация μ, σ: состояние 0 = нижние 50%, состояние 1 = верхние 50%
        median = float(np.median(obs))
        lower = obs[obs <= median]
        upper = obs[obs > median]

        if anomaly_scores is not None and len(anomaly_scores) >= 5:
            anom = np.array(anomaly_scores, dtype=float)
            mu0 = float(np.mean(lower)) if len(lower) > 0 else float(np.mean(obs))
            sigma0 = max(float(np.std(lower)) if len(lower) > 0 else float(np.std(obs)), self._cfg.min_sigma)
            mu1 = float(np.mean(anom))
            sigma1 = max(float(np.std(anom)), self._cfg.min_sigma)
        else:
            mu0 = float(np.mean(lower)) if len(lower) > 0 else float(np.mean(obs))
            sigma0 = max(float(np.std(lower)) if len(lower) > 0 else float(np.std(obs)), self._cfg.min_sigma)
            mu1 = float(np.mean(upper)) if len(upper) > 0 else mu0 + sigma0
            sigma1 = max(float(np.std(upper)) if len(upper) > 0 else sigma0, self._cfg.min_sigma)
            # Убедиться, что mu1 > mu0 (state 1 = более высокие скоры)
            if mu1 <= mu0:
                mu1 = mu0 + sigma0

        self._mu = np.array([mu0, mu1])
        self._sigma = np.array([sigma0, sigma1])

        # Инициализация A: высокая персистентность (стандарт для SRE метрик)
        self._A = np.array([[0.95, 0.05], [0.10, 0.90]])
        self._pi = np.array([0.9, 0.1])

        if K != 2:
            # Generic init для n_states != 2
            self._A = np.ones((K, K)) / K
            self._pi = np.ones(K) / K
            percentiles = np.linspace(0, 100, K + 1)
            boundaries = np.percentile(obs, percentiles)
            for i in range(K):
                mask = (obs >= boundaries[i]) & (obs <= boundaries[i + 1])
                subset = obs[mask] if mask.sum() > 0 else obs
                self._mu[i] = float(np.mean(subset))
                self._sigma[i] = max(float(np.std(subset)), self._cfg.min_sigma)

        # Если anomaly_scores предоставлены — включаем их в обучающую последовательность.
        # Это позволяет Baum-Welch увидеть оба режима и не коллапсировать состояние ANOMALY.
        if anomaly_scores is not None and len(anomaly_scores) >= 5:
            anom_arr = np.array(anomaly_scores, dtype=float)
            obs = np.concatenate([obs, anom_arr])

        # Baum-Welch EM
        prev_ll = None
        n_iter = 0
        converged = False
        ll = float("nan")

        for iteration in range(self._cfg.n_iter):
            log_b = self._log_emit_all(obs)
            gamma, xi, ll = self._e_step(obs, log_b)
            self._m_step(obs, gamma, xi)
            n_iter = iteration + 1

            if prev_ll is not None and abs(ll - prev_ll) < self._cfg.tol:
                converged = True
                break
            prev_ll = ll

        # Определяем, какой state = ANOMALY (с большим средним)
        self._normal_state = int(np.argmin(self._mu))
        self._anomaly_state = int(np.argmax(self._mu))

        self._is_calibrated = True
        self._n_calibration = len(scores)
        self._n_updates = 0
        self._buffer = list(scores[-self._buffer_maxlen :])  # seed буфер

        return HMMCalibrationResult(
            n_iter_done=n_iter,
            converged=converged,
            log_likelihood=float(ll),
            transition_matrix=self._A.tolist(),
            means=self._mu.tolist(),
            stds=self._sigma.tolist(),
            normal_state=self._normal_state,
            anomaly_state=self._anomaly_state,
        )

    # ---------------------------------------------------------------------- #
    # Viterbi decoding
    # ---------------------------------------------------------------------- #

    def viterbi(self, scores: list[float]) -> tuple[list[int], list[float]]:
        """Найти наиболее вероятную последовательность состояний (алгоритм Витерби).

        Args:
            scores: последовательность наблюдений.

        Returns:
            states: list[int] — Viterbi path (0 или 1 per timestep).
            path_log_probs: list[float] — lог-вероятность лучшего пути на каждом шаге.

        Raises:
            RuntimeError: если модель не откалибрована.
        """
        if not self._is_calibrated:
            raise RuntimeError("Требуется calibrate() перед viterbi()")
        obs = np.array(scores, dtype=float)
        T = len(obs)
        K = self._cfg.n_states
        log_A = np.log(np.clip(self._A, 1e-300, None))
        log_b = self._log_emit_all(obs)

        delta = np.full((T, K), _NEG_INF)
        psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(np.clip(self._pi, 1e-300, None)) + log_b[0]

        for t in range(1, T):
            for j in range(K):
                scores_prev = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores_prev))
                delta[t, j] = scores_prev[psi[t, j]] + log_b[t, j]

        # Traceback
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states.tolist(), delta[np.arange(T), states].tolist()

    # ---------------------------------------------------------------------- #
    # Batch decode
    # ---------------------------------------------------------------------- #

    def decode(self, scores: list[float]) -> HMMDecodeResult:
        """Полное декодирование: Viterbi + forward-backward posterior.

        Args:
            scores: последовательность скоров (≥ 2 точки).

        Returns:
            HMMDecodeResult с states, anomaly_probability, change_points.

        Raises:
            RuntimeError: если модель не откалибрована.
            ValueError: если меньше 2 точек.
        """
        if not self._is_calibrated:
            raise RuntimeError("Требуется calibrate() перед decode()")
        if len(scores) < 2:
            raise ValueError("Нужно ≥ 2 точки для декодирования")

        obs = np.array(scores, dtype=float)
        log_b = self._log_emit_all(obs)

        # Viterbi states
        states, _ = self.viterbi(scores)

        # Posterior P(ANOMALY_t | Y) из forward-backward γ
        gamma, _, _ = self._e_step(obs, log_b)
        anom_prob = gamma[:, self._anomaly_state].tolist()

        # is_anomaly по порогу
        predictions = [p > self._cfg.anomaly_threshold for p in anom_prob]
        anomaly_indices = [i for i, p in enumerate(predictions) if p]

        # Change points: где state[t] != state[t-1]
        change_points = [
            t for t in range(1, len(states)) if states[t] != states[t - 1]
        ]

        state_names = [
            "ANOMALY" if s == self._anomaly_state else "NORMAL" for s in states
        ]

        return HMMDecodeResult(
            states=states,
            state_names=state_names,
            anomaly_probability=anom_prob,
            predictions=predictions,
            anomaly_indices=anomaly_indices,
            n_anomalies=len(anomaly_indices),
            change_points=change_points,
        )

    # ---------------------------------------------------------------------- #
    # Online update
    # ---------------------------------------------------------------------- #

    def update(self, value: float) -> HMMUpdateResult:
        """Онлайн-обновление: добавить одну точку, переоценить текущий режим.

        Использует скользящий буфер + Viterbi для оценки текущего состояния.
        O(buffer_len) на вызов, O(buffer_maxlen) память.

        Args:
            value: новое наблюдение (скор аномальности).

        Raises:
            RuntimeError: если модель не откалибрована.
        """
        if not self._is_calibrated:
            raise RuntimeError("Требуется calibrate() перед update()")

        # Обновляем буфер (скользящее окно)
        self._buffer.append(value)
        if len(self._buffer) > self._buffer_maxlen:
            self._buffer = self._buffer[-self._buffer_maxlen :]

        self._n_updates += 1

        obs = np.array(self._buffer, dtype=float)
        log_b = self._log_emit_all(obs)

        # Posterior на последней точке
        gamma, _, _ = self._e_step(obs, log_b)
        anom_prob = float(gamma[-1, self._anomaly_state])

        # Viterbi state на последней точке
        states, _ = self.viterbi(self._buffer)
        current_state = states[-1]
        state_name = "ANOMALY" if current_state == self._anomaly_state else "NORMAL"

        return HMMUpdateResult(
            value=value,
            current_state=current_state,
            state_name=state_name,
            anomaly_probability=anom_prob,
            is_anomaly=anom_prob > self._cfg.anomaly_threshold,
            n_updates=self._n_updates,
        )

    # ---------------------------------------------------------------------- #
    # State inspection
    # ---------------------------------------------------------------------- #

    def get_state(self) -> HMMState:
        """Вернуть текущее состояние модели для мониторинга."""
        return HMMState(
            is_calibrated=self._is_calibrated,
            n_states=self._cfg.n_states,
            anomaly_threshold=self._cfg.anomaly_threshold,
            n_calibration=self._n_calibration,
            n_updates=self._n_updates,
            transition_matrix=self._A.tolist() if self._A is not None else None,
            means=self._mu.tolist() if self._mu is not None else None,
            stds=self._sigma.tolist() if self._sigma is not None else None,
            normal_state=self._normal_state if self._is_calibrated else None,
            anomaly_state=self._anomaly_state if self._is_calibrated else None,
        )

    def reset(self) -> None:
        """Сбросить состояние до некалиброванного."""
        self._A = None
        self._mu = None
        self._sigma = None
        self._pi = None
        self._is_calibrated = False
        self._n_calibration = 0
        self._n_updates = 0
        self._buffer = []
