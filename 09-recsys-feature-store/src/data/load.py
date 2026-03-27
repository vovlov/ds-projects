"""
Генерация синтетических данных для e-commerce рекомендаций.
Synthetic e-commerce interaction data generator.

Создаём реалистичные данные: пользователи с разными предпочтениями,
товары по категориям, и взаимодействия с временными метками.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

# Константы для генерации / Generation constants
N_USERS = 500
N_PRODUCTS = 200
N_INTERACTIONS = 10_000

CATEGORIES = ["electronics", "clothing", "books", "food", "sports"]
PRICE_TIERS = ["low", "mid", "high"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Вероятности оценок зависят от «совпадения» пользователя и товара
# Rating probabilities depend on user-item affinity
SEED = 42


def generate_users(n_users: int = N_USERS, seed: int = SEED) -> pl.DataFrame:
    """Генерируем таблицу пользователей с демографией.
    Generate user table with demographics."""
    rng = np.random.default_rng(seed)

    user_ids = list(range(1, n_users + 1))
    age_groups = rng.choice(AGE_GROUPS, size=n_users).tolist()
    # Дни с момента регистрации — от 1 до 730 (два года)
    signup_days = rng.integers(1, 730, size=n_users).tolist()

    return pl.DataFrame(
        {
            "user_id": user_ids,
            "age_group": age_groups,
            "signup_days_ago": signup_days,
        }
    )


def generate_products(n_products: int = N_PRODUCTS, seed: int = SEED) -> pl.DataFrame:
    """Генерируем каталог товаров.
    Generate product catalog with categories and price tiers."""
    rng = np.random.default_rng(seed)

    product_ids = list(range(1, n_products + 1))
    categories = rng.choice(CATEGORIES, size=n_products).tolist()
    price_tiers = rng.choice(PRICE_TIERS, size=n_products).tolist()

    return pl.DataFrame(
        {
            "product_id": product_ids,
            "category": categories,
            "price_tier": price_tiers,
        }
    )


def generate_interactions(
    n_users: int = N_USERS,
    n_products: int = N_PRODUCTS,
    n_interactions: int = N_INTERACTIONS,
    seed: int = SEED,
) -> pl.DataFrame:
    """Генерируем взаимодействия пользователь-товар.
    Generate user-product interactions with ratings and timestamps.

    Используем power-law распределение: некоторые товары и пользователи
    значительно активнее остальных — как в реальном e-commerce.
    """
    rng = np.random.default_rng(seed)

    # Power-law: популярные товары получают больше оценок
    product_weights = rng.power(0.5, size=n_products)
    product_weights /= product_weights.sum()

    # Аналогично для пользователей — кто-то активнее
    user_weights = rng.power(0.6, size=n_users)
    user_weights /= user_weights.sum()

    user_ids = rng.choice(range(1, n_users + 1), size=n_interactions, p=user_weights).tolist()
    product_ids = rng.choice(
        range(1, n_products + 1), size=n_interactions, p=product_weights
    ).tolist()

    # Оценки: нормальное распределение вокруг 3.5 (лёгкий позитивный сдвиг)
    ratings_raw = rng.normal(loc=3.5, scale=1.2, size=n_interactions)
    ratings = np.clip(np.round(ratings_raw), 1, 5).astype(int).tolist()

    # Временные метки за последний год
    base_date = datetime(2026, 3, 1)
    timestamps = [
        (base_date - timedelta(days=int(rng.integers(0, 365)))).isoformat()
        for _ in range(n_interactions)
    ]

    return pl.DataFrame(
        {
            "user_id": user_ids,
            "product_id": product_ids,
            "rating": ratings,
            "timestamp": timestamps,
        }
    )


def load_all_data(
    seed: int = SEED,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Загружаем все три таблицы разом.
    Load all three tables at once. Convenience function."""
    users = generate_users(seed=seed)
    products = generate_products(seed=seed)
    interactions = generate_interactions(seed=seed)
    return users, products, interactions


if __name__ == "__main__":
    users, products, interactions = load_all_data()
    print(f"Users: {users.shape}")
    print(f"Products: {products.shape}")
    print(f"Interactions: {interactions.shape}")
    print(f"\nSample interactions:\n{interactions.head()}")
    print(f"\nRating distribution:\n{interactions['rating'].value_counts().sort('rating')}")
