"""
Офлайн-вычисление фичей для feature store.
Offline feature computation — batch processing with Polars.

Считаем агрегаты по пользователям и товарам,
сохраняем в Parquet для быстрого чтения.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from recsys.feature_store.registry import FeatureRegistry


def compute_user_features(
    interactions_df: pl.DataFrame,
    products_df: pl.DataFrame,
) -> pl.DataFrame:
    """Вычисляем пользовательские фичи из истории взаимодействий.
    Compute user-level features from interaction history.

    Фичи:
    - avg_rating: средний рейтинг, который ставит пользователь
    - n_purchases: количество взаимодействий
    - favorite_category: самая частая категория
    - recency: дней с последнего взаимодействия
    """
    # Базовые агрегаты по пользователю
    user_stats = interactions_df.group_by("user_id").agg(
        [
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("rating").count().alias("n_purchases"),
            pl.col("timestamp").max().alias("last_interaction"),
        ]
    )

    # Recency: сколько дней прошло с последнего взаимодействия
    user_stats = user_stats.with_columns(
        (
            pl.lit("2026-03-27T00:00:00").str.to_datetime()
            - pl.col("last_interaction").str.to_datetime()
        )
        .dt.total_days()
        .alias("recency_days")
    ).drop("last_interaction")

    # Любимая категория — нужен join с продуктами
    interactions_with_cat = interactions_df.join(
        products_df.select(["product_id", "category"]),
        on="product_id",
        how="left",
    )

    # Считаем количество покупок по категориям для каждого пользователя
    cat_counts = interactions_with_cat.group_by(["user_id", "category"]).agg(
        pl.col("rating").count().alias("cat_count")
    )

    # Берём категорию с максимальным количеством
    favorite_cat = (
        cat_counts.sort(["user_id", "cat_count"], descending=[False, True])
        .group_by("user_id")
        .first()
        .select(["user_id", pl.col("category").alias("favorite_category")])
    )

    # Собираем всё вместе
    user_features = user_stats.join(favorite_cat, on="user_id", how="left")

    return user_features


def compute_item_features(interactions_df: pl.DataFrame) -> pl.DataFrame:
    """Вычисляем товарные фичи из истории взаимодействий.
    Compute item-level features from interaction data.

    Фичи:
    - avg_rating: средний рейтинг товара
    - n_ratings: количество оценок
    - popularity_rank: ранг популярности (1 = самый популярный)
    """
    item_stats = interactions_df.group_by("product_id").agg(
        [
            pl.col("rating").mean().alias("avg_rating"),
            pl.col("rating").count().alias("n_ratings"),
        ]
    )

    # Ранг популярности по количеству оценок (desc)
    item_stats = item_stats.with_columns(
        pl.col("n_ratings").rank(method="ordinal", descending=True).alias("popularity_rank")
    )

    return item_stats


def save_features_parquet(df: pl.DataFrame, path: str | Path, name: str) -> Path:
    """Сохраняем фичи в Parquet-файл.
    Save computed features to Parquet format."""
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{name}.parquet"
    df.write_parquet(file_path)
    return file_path


def load_features_parquet(path: str | Path) -> pl.DataFrame:
    """Загружаем фичи из Parquet.
    Load features from a Parquet file."""
    return pl.read_parquet(path)


def populate_registry(
    registry: FeatureRegistry,
    user_features_df: pl.DataFrame,
    item_features_df: pl.DataFrame,
) -> FeatureRegistry:
    """Заполняем реестр вычисленными фичами.
    Populate feature registry with computed feature values."""

    # Регистрируем определения пользовательских фичей
    user_feature_defs = [
        ("avg_rating", "float", "Average rating given by user"),
        ("n_purchases", "int", "Number of user interactions"),
        ("favorite_category", "str", "Most frequent category"),
        ("recency_days", "int", "Days since last interaction"),
    ]
    for name, dtype, desc in user_feature_defs:
        if name not in [f.name for f in registry.list_features()]:
            registry.register_feature(name, dtype, desc, entity_type="user")

    # Регистрируем определения товарных фичей
    item_feature_defs = [
        ("item_avg_rating", "float", "Average item rating"),
        ("n_ratings", "int", "Number of ratings received"),
        ("popularity_rank", "int", "Popularity rank (1=most popular)"),
    ]
    for name, dtype, desc in item_feature_defs:
        if name not in [f.name for f in registry.list_features()]:
            registry.register_feature(name, dtype, desc, entity_type="item")

    # Заполняем значения для пользователей
    for row in user_features_df.iter_rows(named=True):
        entity_id = f"user_{row['user_id']}"
        registry.set_features(
            entity_id,
            {
                "avg_rating": row["avg_rating"],
                "n_purchases": row["n_purchases"],
                "favorite_category": row["favorite_category"],
                "recency_days": row["recency_days"],
            },
        )

    # Заполняем значения для товаров
    for row in item_features_df.iter_rows(named=True):
        entity_id = f"item_{row['product_id']}"
        registry.set_features(
            entity_id,
            {
                "item_avg_rating": row["avg_rating"],
                "n_ratings": row["n_ratings"],
                "popularity_rank": row["popularity_rank"],
            },
        )

    return registry


if __name__ == "__main__":
    from recsys.data.load import load_all_data

    users, products, interactions = load_all_data()

    user_feats = compute_user_features(interactions, products)
    item_feats = compute_item_features(interactions)

    print(f"User features:\n{user_feats.head()}\n")
    print(f"Item features:\n{item_feats.head()}\n")

    # Сохраняем
    data_dir = Path(__file__).parent.parent.parent / "data" / "features"
    save_features_parquet(user_feats, data_dir, "user_features")
    save_features_parquet(item_feats, data_dir, "item_features")
    print(f"Features saved to {data_dir}")
