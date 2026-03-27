# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "polars",
#     "numpy",
#     "matplotlib",
#     "seaborn",
# ]
# ///
"""
EDA для данных рекомендательной системы.
Exploratory data analysis for the recommendation engine dataset.

Запуск: uv run notebooks/eda.py (или python notebooks/eda.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.data.load import load_all_data

# Загружаем данные / Load data
users, products, interactions = load_all_data()

print("=" * 60)
print("DATASET OVERVIEW / ОБЗОР ДАННЫХ")
print("=" * 60)

print(f"\nUsers: {users.shape[0]} rows, {users.shape[1]} columns")
print(f"Products: {products.shape[0]} rows, {products.shape[1]} columns")
print(f"Interactions: {interactions.shape[0]} rows, {interactions.shape[1]} columns")

# --- Распределение оценок / Rating distribution ---
print("\n--- Rating Distribution / Распределение оценок ---")
rating_dist = interactions["rating"].value_counts().sort("rating")
print(rating_dist)

# --- Активность пользователей / User activity ---
user_activity = (
    interactions.group_by("user_id")
    .agg(pl.col("rating").count().alias("n_ratings"))
    .sort("n_ratings", descending=True)
)
print(f"\nUser activity stats / Статистика активности:")
print(f"  Mean ratings per user: {user_activity['n_ratings'].mean():.1f}")
print(f"  Median: {user_activity['n_ratings'].median():.1f}")
print(f"  Max: {user_activity['n_ratings'].max()}")
print(f"  Min: {user_activity['n_ratings'].min()}")

# --- Популярность товаров / Product popularity ---
product_pop = (
    interactions.group_by("product_id")
    .agg(pl.col("rating").count().alias("n_ratings"))
    .sort("n_ratings", descending=True)
)
print(f"\nProduct popularity stats / Статистика популярности:")
print(f"  Mean ratings per product: {product_pop['n_ratings'].mean():.1f}")
print(f"  Median: {product_pop['n_ratings'].median():.1f}")
print(f"  Top-5 products: {product_pop.head(5)['product_id'].to_list()}")

# --- Категории / Category breakdown ---
cat_stats = (
    interactions.join(
        products.select(["product_id", "category"]),
        on="product_id",
        how="left",
    )
    .group_by("category")
    .agg([
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").count().alias("n_interactions"),
    ])
    .sort("n_interactions", descending=True)
)
print(f"\nCategory stats / Статистика по категориям:")
print(cat_stats)

# --- Разреженность матрицы / Matrix sparsity ---
n_users = users.shape[0]
n_products = products.shape[0]
n_possible = n_users * n_products
n_actual = interactions.shape[0]
sparsity = 1 - (n_actual / n_possible)
print(f"\nMatrix sparsity / Разреженность матрицы: {sparsity:.2%}")
print(f"  ({n_actual} interactions out of {n_possible} possible)")

# --- Графики / Plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Распределение оценок / Rating distribution
axes[0, 0].bar(
    rating_dist["rating"].to_list(),
    rating_dist["count"].to_list(),
    color="steelblue",
)
axes[0, 0].set_title("Rating Distribution / Распределение оценок")
axes[0, 0].set_xlabel("Rating")
axes[0, 0].set_ylabel("Count")

# 2. Активность пользователей / User activity histogram
axes[0, 1].hist(
    user_activity["n_ratings"].to_list(),
    bins=30,
    color="coral",
    edgecolor="black",
)
axes[0, 1].set_title("User Activity / Активность пользователей")
axes[0, 1].set_xlabel("Number of ratings")
axes[0, 1].set_ylabel("Number of users")

# 3. Популярность товаров / Product popularity
axes[1, 0].hist(
    product_pop["n_ratings"].to_list(),
    bins=30,
    color="mediumseagreen",
    edgecolor="black",
)
axes[1, 0].set_title("Product Popularity / Популярность товаров")
axes[1, 0].set_xlabel("Number of ratings")
axes[1, 0].set_ylabel("Number of products")

# 4. Средний рейтинг по категориям / Average rating by category
axes[1, 1].barh(
    cat_stats["category"].to_list(),
    cat_stats["avg_rating"].to_list(),
    color="mediumpurple",
)
axes[1, 1].set_title("Avg Rating by Category / Средний рейтинг по категориям")
axes[1, 1].set_xlabel("Average Rating")
axes[1, 1].set_xlim(1, 5)

plt.tight_layout()
plt.savefig(str(Path(__file__).parent / "eda_plots.png"), dpi=150)
plt.show()
print("\nPlots saved to notebooks/eda_plots.png")
