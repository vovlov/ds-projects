"""
Устранение смещения популярности через Inverse Propensity Scoring (IPS).
Popularity Debiasing via Inverse Propensity Scoring (IPS).

Проблема: популярные товары получают больше показов → больше взаимодействий →
модель их чаще рекомендует → замкнутый круг смещения (popularity feedback loop).

Решение: каждое взаимодействие с товаром i взвешивается как 1/p(i), где
p(i) ∝ popularity(i)^alpha — вероятность показа этого товара пользователю.

Sources:
  Schnabel et al. 2016 "Recommendations as Treatments" (ICML 2016)
  Swaminathan & Joachims 2015 "Counterfactual Risk Minimization" (ICML 2015, SNIPS variant)
  Chen et al. 2023 "Bias and Debias in Recommender System" (ACM TOIS survey)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class DebiasingConfig:
    """Конфигурация IPS-деbiasing / IPS debiasing configuration."""

    # Показатель степенного закона: 0=равномерное, 1=пропорционально популярности
    # Power law exponent: 0=uniform, 1=proportional to popularity
    alpha: float = 0.5
    # Порог клиппинга IPS весов (None = без клиппинга)
    # IPS weight clipping threshold (None = no clipping)
    clip_max: float | None = 10.0
    # Минимальная propensity — защита от деления на ~0
    # Minimum propensity to avoid near-zero division
    min_propensity: float = 1e-4


@dataclass
class PropensityStats:
    """Статистика распределения propensity / Propensity distribution statistics."""

    n_items: int
    mean_propensity: float
    std_propensity: float
    min_propensity: float
    max_propensity: float
    mean_ips_weight: float
    gini_coefficient: float  # мера неравенства популярностей / popularity inequality
    top10_concentration: float  # доля взаимодействий у топ-10% товаров / top-10% share


@dataclass
class IPSEvaluationResult:
    """Сравнение стандартных и IPS-скорректированных метрик.
    Comparison of standard vs IPS-corrected evaluation metrics."""

    ndcg_standard: float
    ndcg_ips: float  # IPS-corrected NDCG (IDCG нормализован через IPS-веса)
    precision_standard: float
    precision_ips: float
    # Насколько рекомендации смещены к популярным товарам (1.0 = нет смещения)
    # How biased recommendations are toward popular items (1.0 = unbiased)
    popularity_bias_ratio: float
    # Доля каталога покрытая в top-K рекомендациях
    # Fraction of catalog covered in top-K recommendations
    catalog_coverage: float
    n_users_evaluated: int


class PopularityDebiaser:
    """Оценивает propensity-веса из частот взаимодействий и корректирует метрики.

    Estimates propensity weights from interaction frequencies and corrects
    evaluation metrics for popularity bias using IPS weighting.

    Propensity: p(i) ∝ count(i)^alpha, normalized to sum to 1.
    IPS weight: w(i) = 1 / p(i), optionally clipped to [1, clip_max].
    """

    def __init__(self, config: DebiasingConfig | None = None) -> None:
        self.config = config or DebiasingConfig()
        self._propensities: dict[int, float] = {}
        self._item_counts: dict[int, int] = {}
        self._is_fitted: bool = False

    def fit(self, interactions_df: pl.DataFrame) -> "PopularityDebiaser":
        """Оценивает propensity каждого товара из частот взаимодействий.
        Estimates propensity of each item from interaction frequencies.

        Args:
            interactions_df: DataFrame с колонкой product_id.
        """
        counts = (
            interactions_df
            .group_by("product_id")
            .agg(pl.len().alias("count"))
        )

        item_ids = counts["product_id"].to_list()
        raw_counts = counts["count"].to_numpy().astype(float)

        # Степенной закон: p(i) ∝ count(i)^alpha
        # alpha=0: равномерное распределение (нет debiasing)
        # alpha=1: пропорционально популярности (максимальное debiasing)
        powered = raw_counts ** self.config.alpha
        total = powered.sum()

        for item_id, cnt, pw in zip(item_ids, raw_counts.tolist(), powered.tolist()):
            p = max(pw / total, self.config.min_propensity)
            self._propensities[int(item_id)] = p
            self._item_counts[int(item_id)] = int(cnt)

        self._is_fitted = True
        return self

    def get_propensity(self, item_id: int) -> float:
        """Propensity score для товара (оценка частоты показа).
        Propensity score for item (estimated exposure probability)."""
        if not self._is_fitted:
            raise RuntimeError("PopularityDebiaser not fitted. Call fit() first.")
        return self._propensities.get(item_id, self.config.min_propensity)

    def get_ips_weight(self, item_id: int) -> float:
        """IPS вес = 1/propensity, клиппинг для контроля дисперсии.
        IPS weight = 1/propensity, clipped for variance reduction."""
        w = 1.0 / self.get_propensity(item_id)
        if self.config.clip_max is not None:
            w = min(w, self.config.clip_max)
        return w

    def debias_scores(
        self,
        recommendations: list[tuple[int, float]],
        scale: float = 0.3,
    ) -> list[tuple[int, float]]:
        """Корректирует скоры рекомендаций: понижает популярные, повышает нишевые.
        Adjusts recommendation scores: down-weights popular, up-weights niche items.

        Debiased score = original_score * propensity^(-scale).
        scale=0: без изменений, scale=1: полный IPS, scale=0.3: мягкая коррекция.
        """
        if not self._is_fitted:
            raise RuntimeError("PopularityDebiaser not fitted. Call fit() first.")

        result = []
        for item_id, score in recommendations:
            p = self.get_propensity(item_id)
            # p^(-scale) > 1 для нишевых товаров — boost нишевых
            adjustment = p ** (-scale)
            if self.config.clip_max is not None:
                adjustment = min(adjustment, self.config.clip_max)
            result.append((item_id, score * adjustment))

        # Переранжируем по debiased скору
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def compute_propensity_stats(self) -> PropensityStats:
        """Статистика распределения propensity для мониторинга / Propensity distribution stats."""
        if not self._is_fitted:
            raise RuntimeError("PopularityDebiaser not fitted. Call fit() first.")

        props = np.array(list(self._propensities.values()))
        weights = np.array([1.0 / p for p in props])
        if self.config.clip_max is not None:
            weights = np.clip(weights, 1.0, self.config.clip_max)

        # Коэффициент Джини как мера неравенства популярностей
        # Gini coefficient as measure of popularity inequality
        sorted_props = np.sort(props)
        n = len(sorted_props)
        idx = np.arange(1, n + 1)
        gini = float((2 * (idx * sorted_props).sum() / (n * sorted_props.sum())) - (n + 1) / n)

        # Концентрация: топ-10% товаров захватывают какую долю взаимодействий?
        total_count = sum(self._item_counts.values())
        sorted_counts = sorted(self._item_counts.values(), reverse=True)
        top10_n = max(1, len(sorted_counts) // 10)
        top10_conc = sum(sorted_counts[:top10_n]) / total_count if total_count > 0 else 0.0

        return PropensityStats(
            n_items=len(self._propensities),
            mean_propensity=float(props.mean()),
            std_propensity=float(props.std()),
            min_propensity=float(props.min()),
            max_propensity=float(props.max()),
            mean_ips_weight=float(weights.mean()),
            gini_coefficient=gini,
            top10_concentration=top10_conc,
        )

    def evaluate_ips(
        self,
        recommendations_per_user: dict[int, list[tuple[int, float]]],
        relevant_per_user: dict[int, set[int]],
        top_k: int = 10,
    ) -> IPSEvaluationResult:
        """IPS-скорректированные метрики vs стандартные.
        IPS-corrected metrics vs standard metrics.

        Args:
            recommendations_per_user: user_id → [(item_id, score), ...]
            relevant_per_user: user_id → {relevant_item_id, ...}
            top_k: количество рекомендаций для оценки
        """
        if not self._is_fitted:
            raise RuntimeError("PopularityDebiaser not fitted. Call fit() first.")

        all_items = set(self._propensities.keys())

        std_ndcgs, ips_ndcgs = [], []
        std_precs, ips_precs = [], []
        rec_popularities = []

        for user_id, recs in recommendations_per_user.items():
            relevant = relevant_per_user.get(user_id, set())
            if not relevant:
                continue

            rec_ids = [pid for pid, _ in recs[:top_k]]

            # Популярность рекомендованных товаров (% каталога который популярнее)
            for pid in rec_ids:
                cnt = self._item_counts.get(pid, 0)
                rank_pct = sum(1 for c in self._item_counts.values() if c > cnt) / max(len(self._item_counts), 1)
                rec_popularities.append(rank_pct)

            hits_mask = [1 if pid in relevant else 0 for pid in rec_ids]
            ips_weights = [self.get_ips_weight(pid) for pid in rec_ids]

            # Стандартный NDCG@K
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits_mask))
            ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), top_k)))
            std_ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

            # IPS-скорректированный NDCG@K: relevance взвешена IPS
            # Нивелирует то, что модель "видела" популярные товары чаще
            ips_dcg = sum((h * w) / np.log2(i + 2) for i, (h, w) in enumerate(zip(hits_mask, ips_weights)))
            ips_ideal = sum(
                self.get_ips_weight(pid) / np.log2(i + 2)
                for i, pid in enumerate(sorted(relevant, key=self.get_ips_weight, reverse=True)[:top_k])
            )
            ips_ndcgs.append(ips_dcg / ips_ideal if ips_ideal > 0 else 0.0)

            # Precision
            std_precs.append(sum(hits_mask) / top_k)
            ips_prec = sum(h * w for h, w in zip(hits_mask, ips_weights))
            ips_norm = sum(ips_weights)
            ips_precs.append(ips_prec / ips_norm if ips_norm > 0 else 0.0)

        # Catalog coverage: сколько уникальных товаров рекомендуется
        all_recommended = {
            pid
            for recs in recommendations_per_user.values()
            for pid, _ in recs[:top_k]
        }
        coverage = len(all_recommended) / len(all_items) if all_items else 0.0

        # Popularity bias ratio: насколько рекомендации смещены к популярным товарам
        # 0 = только самые популярные, 0.5 = случайные, значения > 0.5 = нишевые
        mean_popularity_rank = float(np.mean(rec_popularities)) if rec_popularities else 0.5
        # Нормализуем: 0.5 = ожидаемое при равномерных рекомендациях
        popularity_bias_ratio = mean_popularity_rank  # выше = менее смещённые к популярным

        return IPSEvaluationResult(
            ndcg_standard=float(np.mean(std_ndcgs)) if std_ndcgs else 0.0,
            ndcg_ips=float(np.mean(ips_ndcgs)) if ips_ndcgs else 0.0,
            precision_standard=float(np.mean(std_precs)) if std_precs else 0.0,
            precision_ips=float(np.mean(ips_precs)) if ips_precs else 0.0,
            popularity_bias_ratio=popularity_bias_ratio,
            catalog_coverage=coverage,
            n_users_evaluated=len(std_ndcgs),
        )
