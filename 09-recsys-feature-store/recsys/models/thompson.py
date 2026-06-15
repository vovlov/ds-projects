"""
Beta-Bernoulli Thompson Sampling для рекомендаций.
Beta-Bernoulli Thompson Sampling bandit for recommendations.

Идея: каждый arm (товар) представлен Beta-распределением posterior.
При recommendation семплируем θ_a ~ Beta(α_a, β_a) — exploration бесплатно.
При feedback: click → α_a += 1, skip → β_a += 1.

Thompson Sampling vs LinUCB (из bandit.py):
  - TS:     Bayesian, O(n_arms), binary rewards, лучше при высокой неопределённости
  - LinUCB: Frequentist, O(n_arms × d²), continuous CTR, требует контекстный вектор
  Используйте TS для click/no-click, LinUCB для CTR × context (user+item features).

Алгоритм (Russo et al. 2018 Tutorial, §2):
  Инициализация: α_a = α_prior, β_a = β_prior  (обычно 1.0 — uniform prior)
  Выбор arm:     ã = argmax_a θ̃_a, θ̃_a ~ Beta(α_a, β_a)
  Обновление:    reward=1 → α_a += 1
                 reward=0 → β_a += 1
  E[θ_a] = α_a / (α_a + β_a)  — posterior mean (exploitation)
  Var[θ_a] = α·β / ((α+β)²·(α+β+1))  — posterior variance (exploration signal)

Источники:
  Russo et al. 2018 FnT ML "A Tutorial on Thompson Sampling" (arxiv:1707.02038)
  Agrawal & Goyal 2012 AISTATS "Analysis of Thompson Sampling for MAB"
  Chapelle & Li 2011 NeurIPS "Empirical Evaluation of Thompson Sampling"
  Dynamic Prior TS for Cold-Start: arxiv:2602.00943 (2025)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


@dataclass
class ThompsonConfig:
    """Гиперпараметры Beta-Bernoulli Thompson Sampling.
    Hyperparameters for Beta-Bernoulli Thompson Sampling.

    alpha_prior: параметр α prior Beta-распределения (псевдо-успехи).
                 alpha_prior = beta_prior = 1.0 → равномерный prior (максимальная неопределённость).
                 alpha_prior > 1 → prior смещён в сторону клика (используй если знаешь CTR заранее).
    beta_prior:  параметр β prior (псевдо-неудачи).
    seed:        фиксирует numpy RNG для воспроизводимости тестов.
    """

    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    seed: int | None = None


@dataclass
class ArmPosterior:
    """Posterior Beta-распределение одного arm.
    Beta posterior for a single arm (item).

    alpha = alpha_prior + n_successes  (clicks)
    beta  = beta_prior  + n_failures   (skips)
    """

    arm_id: int
    alpha: float  # α = alpha_prior + n_successes
    beta: float  # β = beta_prior  + n_failures
    n_successes: int = 0
    n_failures: int = 0

    @property
    def n_pulls(self) -> int:
        """Суммарное число обновлений arm."""
        return self.n_successes + self.n_failures

    @property
    def posterior_mean(self) -> float:
        """E[θ] = α/(α+β) — оценка CTR через posterior mean."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_variance(self) -> float:
        """Var[θ] = αβ/((α+β)²(α+β+1)) — неопределённость оценки."""
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n * n * (n + 1))

    @property
    def posterior_std(self) -> float:
        """Стандартное отклонение posterior — прокси exploration bonus."""
        return math.sqrt(max(0.0, self.posterior_variance))


@dataclass
class ThompsonRecommendation:
    """Одна рекомендация с компонентами TS.
    Single recommendation with Thompson Sampling decomposition."""

    arm_id: int
    rank: int
    sampled_theta: float  # θ̃ ~ Beta(α, β) — exploration sample
    expected_reward: float  # E[θ] = α/(α+β) — posterior mean
    uncertainty: float  # std(θ) — exploration signal
    n_pulls: int


@dataclass
class ThompsonResult:
    """Результат Thompson Sampling рекомендации.
    Response from Thompson Sampling recommendation call."""

    recommendations: list[ThompsonRecommendation]
    n_arms_scored: int
    top_arm_id: int
    n_total_arms: int


@dataclass
class ThompsonFeedbackRecord:
    """Запись обратной связи для Thompson Sampling.
    Feedback payload — binary reward (click=1, skip=0)."""

    arm_id: int
    reward: float  # 1.0 (click/purchase) или 0.0 (skip)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ThompsonBandit:
    """
    Beta-Bernoulli Thompson Sampling bandit — numpy-only реализация.
    Beta-Bernoulli Thompson Sampling bandit — pure numpy, no external ML dependencies.

    Принимает binary rewards (1.0=клик, 0.0=пропуск).
    Новые arms автоматически регистрируются при первом появлении с prior Beta(α₀, β₀).
    New arms are auto-registered on first encounter with prior Beta(α₀, β₀).

    Подходит для сценариев с чистым explore/exploit без контекста (или когда контекст
    уже учтён в candidate_ids через pre-filtering). Для контекстного UCB используйте LinUCBBandit.
    """

    def __init__(self, config: ThompsonConfig | None = None) -> None:
        self.config = config or ThompsonConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._arms: dict[int, ArmPosterior] = {}
        self._total_recommendations: int = 0

    def _get_or_init_arm(self, arm_id: int) -> ArmPosterior:
        """Ленивая инициализация arm с prior. / Lazy arm initialization with prior."""
        if arm_id not in self._arms:
            self._arms[arm_id] = ArmPosterior(
                arm_id=arm_id,
                alpha=self.config.alpha_prior,
                beta=self.config.beta_prior,
            )
        return self._arms[arm_id]

    def recommend(
        self,
        candidate_ids: list[int],
        top_k: int = 10,
    ) -> ThompsonResult:
        """
        Ранжировать кандидатов по семплу из Beta-posterior.
        Rank candidates by Thompson sample θ̃ ~ Beta(α, β) from their posteriors.

        Exploration встроен в сэмплинг: arm с высокой неопределённостью → широкий Beta-spread
        → часто получает высокий θ̃ → попадает в топ → exploration без явного bonus-term.
        Exploration is implicit: uncertain arms have wide Beta → high θ̃ → natural exploration.

        Args:
            candidate_ids: список arm_id (product_id) для ранжирования
            top_k:         сколько рекомендаций вернуть

        Returns:
            ThompsonResult с рекомендациями, отсортированными по sampled_theta убыванию.
        """
        scored: list[ThompsonRecommendation] = []

        for arm_id in candidate_ids:
            arm = self._get_or_init_arm(arm_id)
            # Семплируем из текущего posterior — это и есть Thompson Sampling
            # θ̃ = sample from Beta(α, β) — exploration is baked in
            theta_sample = float(self._rng.beta(arm.alpha, arm.beta))
            scored.append(
                ThompsonRecommendation(
                    arm_id=arm_id,
                    rank=0,  # будет проставлен после сортировки
                    sampled_theta=round(theta_sample, 6),
                    expected_reward=round(arm.posterior_mean, 6),
                    uncertainty=round(arm.posterior_std, 6),
                    n_pulls=arm.n_pulls,
                )
            )

        scored.sort(key=lambda r: r.sampled_theta, reverse=True)
        top = scored[:top_k]
        for rank, rec in enumerate(top, start=1):
            rec.rank = rank

        self._total_recommendations += 1

        return ThompsonResult(
            recommendations=top,
            n_arms_scored=len(scored),
            top_arm_id=top[0].arm_id if top else -1,
            n_total_arms=len(self._arms),
        )

    def update(self, arm_id: int, reward: float) -> ArmPosterior:
        """
        Обновить Beta-posterior arm после получения обратной связи.
        Update Beta posterior: click → α += 1, skip → β += 1.

        Conjugate update: Beta prior + Bernoulli likelihood → Beta posterior.
        Это аналитическое обновление без gradient descent или матричных операций.

        Args:
            arm_id: id товара, получившего обратную связь
            reward: 1.0 (клик/покупка) или 0.0 (пропуск)

        Returns:
            Обновлённый ArmPosterior для подтверждения в API-ответе.
        """
        arm = self._get_or_init_arm(arm_id)
        if reward >= 0.5:
            # Treat reward >= 0.5 as success (бинарная интерпретация)
            arm.alpha += 1.0
            arm.n_successes += 1
        else:
            arm.beta += 1.0
            arm.n_failures += 1
        return arm

    def get_arm_stats(self) -> list[dict[str, Any]]:
        """Статистика по всем arms для диагностики и мониторинга.
        Per-arm Beta posterior statistics for monitoring and debugging."""
        stats = []
        for arm_id, arm in self._arms.items():
            stats.append(
                {
                    "arm_id": arm_id,
                    "alpha": round(arm.alpha, 4),
                    "beta": round(arm.beta, 4),
                    "n_pulls": arm.n_pulls,
                    "n_successes": arm.n_successes,
                    "n_failures": arm.n_failures,
                    "posterior_mean": round(arm.posterior_mean, 4),
                    "posterior_std": round(arm.posterior_std, 4),
                }
            )
        return sorted(stats, key=lambda s: s["n_pulls"], reverse=True)

    @property
    def n_arms(self) -> int:
        """Количество известных arms."""
        return len(self._arms)

    @property
    def total_recommendations(self) -> int:
        """Суммарное число вызовов recommend()."""
        return self._total_recommendations
