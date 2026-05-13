"""
LinUCB Contextual Bandit для рекомендаций с exploration-exploitation.
LinUCB Contextual Bandit for recommendations with exploration-exploitation.

Алгоритм LinUCB (Li et al. 2010 WWW):
    θ̂_a = A_a⁻¹ b_a          — ridge regression estimate per arm
    UCB_a(x) = θ̂_aᵀx + α·√(xᵀ A_a⁻¹ x)  — expected reward + exploration bonus
    A_a ← A_a + xxᵀ           — online update after feedback
    b_a ← b_a + r·x           — reward-weighted context accumulation

Применение: на каждом запросе ранжируем кандидаты по UCB;
при получении обратной связи (клик/покупка) обновляем A_a и b_a.
Это балансирует exploitation (рекомендуем то, что работало) и exploration
(пробуем новое, особенно в начале).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class BanditConfig:
    """Гиперпараметры LinUCB. / LinUCB hyperparameters.

    alpha: exploration coefficient — больше alpha → больше исследования новых arms.
           Рекомендуется 0.1–2.0; 1.0 — стандартный старт (Li et al. 2010).
    feature_dim: размерность контекстного вектора (user + item фичи).
    lambda_reg: регуляризация ridge regression (предотвращает вырожденность A).
    """

    alpha: float = 1.0
    feature_dim: int = 8
    lambda_reg: float = 1.0


@dataclass
class ArmState:
    """Состояние одного arm (товара) в LinUCB.
    Per-arm sufficient statistics for LinUCB updates."""

    arm_id: int
    A: np.ndarray  # (d, d) — design matrix, инициализируется λI
    b: np.ndarray  # (d,)  — reward-weighted context accumulator
    n_updates: int = 0
    total_reward: float = 0.0


@dataclass
class BanditRecommendation:
    """Одна рекомендация с UCB-компонентами.
    Single recommendation with UCB decomposition."""

    arm_id: int
    ucb_score: float
    expected_reward: float  # θ̂ᵀx
    exploration_bonus: float  # α·√(xᵀA⁻¹x)
    n_updates: int


@dataclass
class BanditResult:
    """Результат /bandit/recommend.
    Response from bandit recommendation endpoint."""

    recommendations: list[BanditRecommendation]
    n_arms_scored: int
    top_arm_id: int
    config_alpha: float


@dataclass
class FeedbackRecord:
    """Запись обратной связи для /bandit/feedback.
    Feedback payload for /bandit/feedback endpoint."""

    arm_id: int
    context: list[float]
    reward: float  # 0.0 (нет клика) или 1.0 (клик/покупка), или CTR 0–1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LinUCBBandit:
    """
    LinUCB Contextual Bandit — numpy-only реализация.
    LinUCB Contextual Bandit — pure numpy, no external ML dependencies.

    Реализует disjoint LinUCB (каждый arm имеет отдельную A_a и b_a).
    Implements disjoint LinUCB (separate A_a, b_a per arm) from Li et al. 2010.

    Новые arms автоматически добавляются при первом появлении.
    New arms are auto-registered on first encounter.
    """

    def __init__(self, config: BanditConfig | None = None) -> None:
        self.config = config or BanditConfig()
        self._arms: dict[int, ArmState] = {}
        self._total_recommendations: int = 0

    def _get_or_init_arm(self, arm_id: int) -> ArmState:
        """Ленивая инициализация arm. / Lazy arm initialization."""
        if arm_id not in self._arms:
            d = self.config.feature_dim
            self._arms[arm_id] = ArmState(
                arm_id=arm_id,
                A=np.eye(d) * self.config.lambda_reg,
                b=np.zeros(d),
            )
        return self._arms[arm_id]

    def _compute_ucb(self, arm: ArmState, context: np.ndarray) -> tuple[float, float, float]:
        """Вычислить UCB-score для arm и контекстного вектора.
        Compute UCB score = expected_reward + exploration_bonus."""
        A_inv = np.linalg.inv(arm.A)
        theta = A_inv @ arm.b
        expected = float(theta @ context)
        # Числовая защита: max(0,...) т.к. xᵀA⁻¹x ≥ 0 теоретически,
        # но floating-point может дать -ε
        variance = float(max(0.0, context @ A_inv @ context))
        bonus = self.config.alpha * math.sqrt(variance)
        ucb = expected + bonus
        return ucb, expected, bonus

    def recommend(
        self,
        candidate_ids: list[int],
        candidate_contexts: list[list[float]],
        top_k: int = 10,
    ) -> BanditResult:
        """
        Ранжировать кандидатов по UCB score.
        Rank candidates by UCB score and return top_k recommendations.

        Args:
            candidate_ids: список arm_id (product_id) для ранжирования
            candidate_contexts: контекстный вектор для каждого arm
                (обычно конкатенация user_features + item_features)
            top_k: сколько рекомендаций вернуть

        Returns:
            BanditResult с рекомендациями, отсортированными по UCB убыванию.
        """
        if len(candidate_ids) != len(candidate_contexts):
            raise ValueError("candidate_ids and candidate_contexts must have equal length")

        d = self.config.feature_dim
        scored: list[BanditRecommendation] = []

        for arm_id, ctx_raw in zip(candidate_ids, candidate_contexts):
            ctx = np.array(ctx_raw[:d], dtype=float)
            # Pad if context shorter than feature_dim
            if len(ctx) < d:
                ctx = np.pad(ctx, (0, d - len(ctx)))

            arm = self._get_or_init_arm(arm_id)
            ucb, expected, bonus = self._compute_ucb(arm, ctx)
            scored.append(
                BanditRecommendation(
                    arm_id=arm_id,
                    ucb_score=round(ucb, 6),
                    expected_reward=round(expected, 6),
                    exploration_bonus=round(bonus, 6),
                    n_updates=arm.n_updates,
                )
            )

        scored.sort(key=lambda r: r.ucb_score, reverse=True)
        top = scored[:top_k]
        self._total_recommendations += 1

        return BanditResult(
            recommendations=top,
            n_arms_scored=len(scored),
            top_arm_id=top[0].arm_id if top else -1,
            config_alpha=self.config.alpha,
        )

    def update(self, arm_id: int, context: list[float], reward: float) -> None:
        """
        Обновить статистику arm после получения обратной связи.
        Online update: A_a += xxᵀ, b_a += r·x.

        Args:
            arm_id: id товара, получившего обратную связь
            context: тот же контекстный вектор, что был при рекомендации
            reward: сигнал обратной связи (1.0=клик, 0.0=пропуск, или CTR 0–1)
        """
        d = self.config.feature_dim
        ctx = np.array(context[:d], dtype=float)
        if len(ctx) < d:
            ctx = np.pad(ctx, (0, d - len(ctx)))

        arm = self._get_or_init_arm(arm_id)
        arm.A += np.outer(ctx, ctx)
        arm.b += reward * ctx
        arm.n_updates += 1
        arm.total_reward += reward

    def get_arm_stats(self) -> list[dict[str, Any]]:
        """Статистика по всем arms для диагностики и мониторинга.
        Per-arm statistics for monitoring and debugging."""
        stats = []
        for arm_id, arm in self._arms.items():
            avg_reward = arm.total_reward / arm.n_updates if arm.n_updates > 0 else 0.0
            stats.append(
                {
                    "arm_id": arm_id,
                    "n_updates": arm.n_updates,
                    "total_reward": round(arm.total_reward, 4),
                    "avg_reward": round(avg_reward, 4),
                    "A_trace": round(float(np.trace(arm.A)), 4),
                }
            )
        return sorted(stats, key=lambda s: s["n_updates"], reverse=True)

    @property
    def n_arms(self) -> int:
        """Количество известных arms."""
        return len(self._arms)

    @property
    def total_recommendations(self) -> int:
        """Суммарное число вызовов recommend()."""
        return self._total_recommendations
