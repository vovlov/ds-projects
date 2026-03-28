"""
Контентная фильтрация для cold-start пользователей.
Content-based filtering — fallback for new users without history.

Строим профили товаров из категории и ценовой группы,
профили пользователей — из истории взаимодействий.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder


class ContentBasedRecommender:
    """Контентные рекомендации на основе косинусного сходства.
    Content-based recommender using item feature similarity."""

    def __init__(self) -> None:
        self.encoder: OneHotEncoder | None = None
        self.item_profiles: np.ndarray | None = None
        self.product_ids: list[int] = []
        self.product_to_idx: dict[int, int] = {}
        self.products_df: pl.DataFrame | None = None

    def fit(
        self, products_df: pl.DataFrame, interactions_df: pl.DataFrame | None = None
    ) -> ContentBasedRecommender:
        """Строим профили товаров из категорий и ценовых уровней.
        Build item profiles from category + price_tier features."""
        self.products_df = products_df
        self.product_ids = products_df["product_id"].to_list()
        self.product_to_idx = {pid: i for i, pid in enumerate(self.product_ids)}

        # One-hot кодирование категорий и ценовых групп
        features = products_df.select(["category", "price_tier"]).to_pandas()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.item_profiles = self.encoder.fit_transform(features)

        return self

    def _build_user_profile(
        self, interactions_df: pl.DataFrame, user_id: int
    ) -> np.ndarray | None:
        """Строим профиль пользователя как взвешенное среднее профилей товаров.
        Build user profile as weighted average of interacted item profiles."""
        if self.item_profiles is None:
            raise RuntimeError("Model not fitted / Модель не обучена")

        user_interactions = interactions_df.filter(pl.col("user_id") == user_id)
        if user_interactions.height == 0:
            return None

        profile = np.zeros(self.item_profiles.shape[1])
        total_weight = 0.0

        for row in user_interactions.iter_rows(named=True):
            pid = row["product_id"]
            if pid in self.product_to_idx:
                idx = self.product_to_idx[pid]
                weight = row["rating"]
                profile += weight * self.item_profiles[idx]
                total_weight += weight

        if total_weight == 0:
            return None

        return profile / total_weight

    def recommend(
        self,
        user_id: int,
        interactions_df: pl.DataFrame,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Рекомендации для пользователя с историей.
        Recommend items for a user based on their interaction history."""
        if self.item_profiles is None:
            raise RuntimeError("Model not fitted / Модель не обучена")

        user_profile = self._build_user_profile(interactions_df, user_id)
        if user_profile is None:
            return []

        # Косинусное сходство профиля пользователя со всеми товарами
        similarities = cosine_similarity(
            user_profile.reshape(1, -1), self.item_profiles
        )[0]

        # Исключаем уже просмотренные
        seen = set(
            interactions_df.filter(pl.col("user_id") == user_id)["product_id"].to_list()
        )

        scored = []
        for i, score in enumerate(similarities):
            pid = self.product_ids[i]
            if pid not in seen:
                scored.append((pid, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def recommend_for_new_user(
        self,
        preferences: dict[str, str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Рекомендации для нового пользователя (cold start).
        Recommend for a brand new user based on stated preferences.

        preferences — словарь вида {"category": "electronics", "price_tier": "mid"}
        """
        if self.item_profiles is None or self.encoder is None:
            raise RuntimeError("Model not fitted / Модель не обучена")

        # Создаём «идеальный товар» из предпочтений
        pref_df = pl.DataFrame({
            "category": [preferences.get("category", "electronics")],
            "price_tier": [preferences.get("price_tier", "mid")],
        }).to_pandas()

        pref_profile = self.encoder.transform(pref_df)

        # Сходство с каждым товаром
        similarities = cosine_similarity(pref_profile, self.item_profiles)[0]

        scored = [
            (self.product_ids[i], float(sim))
            for i, sim in enumerate(similarities)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def get_popular_items(
    interactions_df: pl.DataFrame, top_k: int = 10
) -> list[tuple[int, float]]:
    """Популярные товары — простейший cold-start fallback.
    Most popular items by average rating (weighted by count). Simplest baseline."""
    popular = (
        interactions_df.group_by("product_id")
        .agg([
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("rating").count().alias("n_ratings"),
        ])
        # Байесовское сглаживание: учитываем количество оценок
        .with_columns(
            (
                (pl.col("avg_rating") * pl.col("n_ratings") + 3.0 * 10)
                / (pl.col("n_ratings") + 10)
            ).alias("score")
        )
        .sort("score", descending=True)
        .head(top_k)
    )

    return list(
        zip(
            popular["product_id"].to_list(),
            popular["score"].to_list(),
        )
    )
