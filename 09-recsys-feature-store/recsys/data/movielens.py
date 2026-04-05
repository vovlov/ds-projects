"""MovieLens-25M dataset loader for RecSys project.

Supports real MovieLens-25M CSV files (GroupLens / Kaggle) and provides
a mock generator for CI environments without network/file access.

Источник: https://grouplens.org/datasets/movielens/25m/
Reference: F. Maxwell Harper and Joseph A. Konstan. 2015.
    The MovieLens Datasets: History and Context.
    ACM TOIS, 5(4), Article 19.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Константы датасета / Dataset constants
# ---------------------------------------------------------------------------

MOVIELENS_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

# MovieLens-25M использует полузвёздные оценки от 0.5 до 5.0
# Half-star ratings are the key difference from integer 1-5 scales
MOVIELENS_RATING_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Реалистичное распределение оценок: пик на 4.0 (люди выбирают фильмы заранее)
# Users self-select movies they expect to enjoy — positive skew toward 4.0
MOVIELENS_RATING_PROBS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.15, 0.20, 0.22, 0.12, 0.08]

MOVIELENS_RATINGS_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
MOVIELENS_MOVIES_COLUMNS = ["movieId", "title", "genres"]


# ---------------------------------------------------------------------------
# Структура статистики / Stats dataclass
# ---------------------------------------------------------------------------


@dataclass
class MovieLensStats:
    """Статистика датасета MovieLens / Descriptive statistics for a dataset split."""

    n_users: int
    n_movies: int
    n_ratings: int
    sparsity: float
    avg_rating: float
    rating_distribution: dict[float, int] = field(default_factory=dict)
    top_genres: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Генерация mock-данных / Mock data generator
# ---------------------------------------------------------------------------


def generate_mock_movielens(
    n_users: int = 100,
    n_movies: int = 50,
    n_ratings: int = 1000,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate mock MovieLens-like data for CI/testing without network access.

    Генерирует синтетические данные в формате MovieLens-25M для использования
    в CI-среде без доступа к сети. Воспроизводит ключевые статистики реального
    датасета: положительный сдвиг оценок и полузвёздную шкалу.

    Args:
        n_users: Number of synthetic users to generate.
        n_movies: Number of synthetic movies to generate.
        n_ratings: Number of rating events to generate.
        seed: Random seed for reproducibility across runs.

    Returns:
        Tuple of (ratings_df, movies_df) with MovieLens column schema.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Power-law activity: некоторые пользователи и фильмы значительно активнее
    user_weights = rng.power(0.6, size=n_users)
    user_weights /= user_weights.sum()
    movie_weights = rng.power(0.5, size=n_movies)
    movie_weights /= movie_weights.sum()

    user_ids = rng.choice(range(1, n_users + 1), size=n_ratings, p=user_weights).tolist()
    movie_ids = rng.choice(range(1, n_movies + 1), size=n_ratings, p=movie_weights).tolist()

    ratings = rng.choice(MOVIELENS_RATING_VALUES, size=n_ratings, p=MOVIELENS_RATING_PROBS).tolist()

    # Unix timestamps: последние 5 лет (2020-2025)
    base_ts = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    end_ts = int(datetime(2025, 12, 31, tzinfo=UTC).timestamp())
    timestamps = rng.integers(base_ts, end_ts, size=n_ratings).tolist()

    ratings_df = pl.DataFrame(
        {
            "userId": user_ids,
            "movieId": movie_ids,
            "rating": ratings,
            "timestamp": timestamps,
        }
    )

    # Каталог фильмов с реалистичными жанрами и годами выпуска
    years = rng.integers(1975, 2024, size=n_movies)
    genres_list = [
        "|".join(sorted(random.sample(MOVIELENS_GENRES, k=random.randint(1, 3))))
        for _ in range(n_movies)
    ]
    movies_df = pl.DataFrame(
        {
            "movieId": list(range(1, n_movies + 1)),
            "title": [f"Movie {i} ({years[i - 1]})" for i in range(1, n_movies + 1)],
            "genres": genres_list,
        }
    )

    return ratings_df, movies_df


# ---------------------------------------------------------------------------
# Загрузка данных / Data loader
# ---------------------------------------------------------------------------


def load_movielens(
    ratings_path: Path | str | None = None,
    movies_path: Path | str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load MovieLens-25M dataset from CSV files with CI fallback to mock data.

    Загружает датасет MovieLens-25M из CSV файлов (ratings.csv, movies.csv).
    При отсутствии файлов автоматически генерирует mock-данные для CI.
    Graceful fallback позволяет одному и тому же коду работать локально
    с реальным датасетом и в CI без скачивания 250MB архива.

    Real dataset: https://grouplens.org/datasets/movielens/25m/
    Kaggle: https://www.kaggle.com/datasets/grouplens/movielens-25m-rating

    Args:
        ratings_path: Path to ratings.csv (userId,movieId,rating,timestamp).
            Pass None to use mock data (default for CI).
        movies_path: Path to movies.csv (movieId,title,genres).
            Pass None to use mock data (default for CI).

    Returns:
        Tuple of (ratings_df, movies_df) with MovieLens column schema.

    Raises:
        ValueError: If provided CSV files are missing expected columns.

    Example::

        # Real data (after downloading from GroupLens):
        ratings, movies = load_movielens(
            "data/ml-25m/ratings.csv",
            "data/ml-25m/movies.csv",
        )

        # CI / dev (no download required):
        ratings, movies = load_movielens()
    """
    # Если пути не переданы — используем mock для CI без сетевого доступа
    if ratings_path is None or movies_path is None:
        return generate_mock_movielens()

    ratings_path = Path(ratings_path)
    movies_path = Path(movies_path)

    # Файлы не найдены — автоматический fallback вместо падения
    if not ratings_path.exists() or not movies_path.exists():
        return generate_mock_movielens()

    ratings_df = pl.read_csv(ratings_path)
    movies_df = pl.read_csv(movies_path)

    # Валидация схемы только при наличии реальных файлов
    missing_rating_cols = set(MOVIELENS_RATINGS_COLUMNS) - set(ratings_df.columns)
    if missing_rating_cols:
        raise ValueError(f"ratings.csv missing columns: {missing_rating_cols}")

    missing_movie_cols = set(MOVIELENS_MOVIES_COLUMNS) - set(movies_df.columns)
    if missing_movie_cols:
        raise ValueError(f"movies.csv missing columns: {missing_movie_cols}")

    return ratings_df, movies_df


# ---------------------------------------------------------------------------
# Статистика / Statistics computation
# ---------------------------------------------------------------------------


def compute_movielens_stats(
    ratings_df: pl.DataFrame,
    movies_df: pl.DataFrame,
) -> MovieLensStats:
    """Compute descriptive statistics for a MovieLens dataset.

    Вычисляет ключевые метрики датасета: покрытие, разреженность матрицы,
    распределение оценок и топ жанры. Используется для EDA и мониторинга
    качества данных перед обучением.

    Args:
        ratings_df: DataFrame with MovieLens ratings schema.
        movies_df: DataFrame with MovieLens movies schema.

    Returns:
        MovieLensStats dataclass with all computed statistics.
    """
    n_users = ratings_df["userId"].n_unique()
    n_movies = movies_df["movieId"].n_unique()
    n_ratings = len(ratings_df)

    # Разреженность: доля возможных пар пользователь-фильм без оценки
    possible_pairs = n_users * n_movies
    sparsity = 1.0 - (n_ratings / possible_pairs) if possible_pairs > 0 else 1.0

    avg_rating = float(ratings_df["rating"].mean() or 0.0)

    rating_dist = ratings_df.group_by("rating").agg(pl.len().alias("count")).sort("rating")
    rating_distribution = {
        float(row["rating"]): int(row["count"]) for row in rating_dist.iter_rows(named=True)
    }

    # Подсчёт жанров — pipe-separated строки разбиваем вручную
    genre_counts: dict[str, int] = {}
    for genres_str in movies_df["genres"].to_list():
        for genre in str(genres_str).split("|"):
            genre = genre.strip()
            if genre and genre != "(no genres listed)":
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    top_genres = sorted(genre_counts, key=lambda g: genre_counts[g], reverse=True)[:10]

    return MovieLensStats(
        n_users=n_users,
        n_movies=n_movies,
        n_ratings=n_ratings,
        sparsity=sparsity,
        avg_rating=avg_rating,
        rating_distribution=rating_distribution,
        top_genres=top_genres,
    )


# ---------------------------------------------------------------------------
# Конвертация в формат RecSys / Schema conversion
# ---------------------------------------------------------------------------


def to_recsys_format(
    ratings_df: pl.DataFrame,
    movies_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert MovieLens DataFrames to the internal RecSys schema.

    Конвертирует данные MovieLens (userId/movieId) в формат, совместимый
    с CollaborativeRecommender и ContentBasedRecommender. Unix-timestamp
    конвертируется в ISO-строку. Жанры из pipe-separated строки → первичный жанр.
    price_tier назначается детерминированно через modulo (реальных цен нет).

    Args:
        ratings_df: MovieLens ratings with userId, movieId, rating, timestamp.
        movies_df: MovieLens movies with movieId, title, genres.

    Returns:
        Tuple of (interactions_df, products_df) matching existing RecSys schema:

        - interactions_df columns: user_id (int), product_id (int),
          rating (float), timestamp (str ISO)
        - products_df columns: product_id (int), category (str),
          price_tier (str: low/mid/high)
    """
    # Unix timestamp → ISO-8601 строка, совместимая с temporal split
    interactions_df = (
        ratings_df.rename({"userId": "user_id", "movieId": "product_id"})
        .with_columns(
            pl.from_epoch("timestamp", time_unit="s")
            .dt.strftime("%Y-%m-%dT%H:%M:%S")
            .alias("timestamp")
        )
        .select(["user_id", "product_id", "rating", "timestamp"])
    )

    # Первичный жанр как категория; "(no genres listed)" → "unknown"
    products_df = (
        movies_df.rename({"movieId": "product_id"})
        .with_columns(
            pl.when(pl.col("genres") == "(no genres listed)")
            .then(pl.lit("unknown"))
            .otherwise(pl.col("genres").str.split("|").list.first().str.to_lowercase())
            .alias("category")
        )
        .with_columns(
            # Детерминированное назначение ценовой категории через modulo
            # Семантически бессмысленно, но позволяет использовать
            # ContentBasedRecommender без изменений
            pl.when(pl.col("product_id") % 3 == 0)
            .then(pl.lit("low"))
            .when(pl.col("product_id") % 3 == 1)
            .then(pl.lit("mid"))
            .otherwise(pl.lit("high"))
            .alias("price_tier")
        )
        .select(["product_id", "category", "price_tier"])
    )

    return interactions_df, products_df
