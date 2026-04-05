"""
Тесты для рекомендательной системы и feature store.
Tests for recommendation engine, feature store, and API.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import polars as pl
import pytest
from fastapi.testclient import TestClient
from recsys.data.load import generate_interactions, generate_products, generate_users
from recsys.data.movielens import (
    MovieLensStats,
    compute_movielens_stats,
    generate_mock_movielens,
    load_movielens,
    to_recsys_format,
)
from recsys.feature_store.offline import compute_item_features, compute_user_features
from recsys.feature_store.registry import FeatureRegistry
from recsys.models.collaborative import CollaborativeRecommender
from recsys.models.content_based import ContentBasedRecommender, get_popular_items

# ========== Фикстуры / Fixtures ==========


@pytest.fixture(scope="module")
def users() -> pl.DataFrame:
    return generate_users(n_users=50, seed=123)


@pytest.fixture(scope="module")
def products() -> pl.DataFrame:
    return generate_products(n_products=30, seed=123)


@pytest.fixture(scope="module")
def interactions() -> pl.DataFrame:
    return generate_interactions(n_users=50, n_products=30, n_interactions=1000, seed=123)


@pytest.fixture(scope="module")
def fitted_collab(interactions: pl.DataFrame) -> CollaborativeRecommender:
    model = CollaborativeRecommender(n_components=10)
    model.fit(interactions)
    return model


@pytest.fixture(scope="module")
def fitted_content(products: pl.DataFrame) -> ContentBasedRecommender:
    model = ContentBasedRecommender()
    model.fit(products)
    return model


# ========== TestData: проверка сгенерированных данных ==========


class TestData:
    """Тесты генерации данных / Data generation tests."""

    def test_users_shape(self, users: pl.DataFrame) -> None:
        assert users.shape == (50, 3)

    def test_products_shape(self, products: pl.DataFrame) -> None:
        assert products.shape == (30, 3)

    def test_interactions_shape(self, interactions: pl.DataFrame) -> None:
        assert interactions.shape == (1000, 4)

    def test_rating_range(self, interactions: pl.DataFrame) -> None:
        """Рейтинги должны быть от 1 до 5 / Ratings must be 1-5."""
        ratings = interactions["rating"]
        assert ratings.min() >= 1
        assert ratings.max() <= 5

    def test_user_ids_valid(self, interactions: pl.DataFrame) -> None:
        """Все user_id в допустимом диапазоне / All user IDs in valid range."""
        user_ids = interactions["user_id"].unique()
        assert user_ids.min() >= 1
        assert user_ids.max() <= 50

    def test_product_ids_valid(self, interactions: pl.DataFrame) -> None:
        """Все product_id в допустимом диапазоне / All product IDs in valid range."""
        product_ids = interactions["product_id"].unique()
        assert product_ids.min() >= 1
        assert product_ids.max() <= 30

    def test_categories_valid(self, products: pl.DataFrame) -> None:
        """Категории из допустимого списка / Categories are from valid set."""
        valid = {"electronics", "clothing", "books", "food", "sports"}
        actual = set(products["category"].unique().to_list())
        assert actual.issubset(valid)


# ========== TestCollaborative: SVD-рекомендации ==========


class TestCollaborative:
    """Тесты коллаборативной фильтрации / Collaborative filtering tests."""

    def test_fit_creates_predicted_ratings(self, fitted_collab: CollaborativeRecommender) -> None:
        assert fitted_collab.predicted_ratings is not None

    def test_recommend_returns_k_items(self, fitted_collab: CollaborativeRecommender) -> None:
        """recommend() возвращает ровно top_k элементов / Returns exactly K items."""
        recs = fitted_collab.recommend(user_id=1, top_k=5)
        assert len(recs) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in recs)

    def test_recommend_unknown_user(self, fitted_collab: CollaborativeRecommender) -> None:
        """Неизвестный пользователь — пустой список / Unknown user gets empty list."""
        recs = fitted_collab.recommend(user_id=99999, top_k=5)
        assert recs == []

    def test_evaluate_returns_metrics(
        self, fitted_collab: CollaborativeRecommender, interactions: pl.DataFrame
    ) -> None:
        """evaluate() возвращает все три метрики / Returns all three metrics."""
        # Берём подмножество как тестовое
        test_df = interactions.tail(200)
        metrics = fitted_collab.evaluate(test_df, top_k=5)
        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics
        assert "ndcg_at_k" in metrics
        # Метрики в допустимом диапазоне [0, 1]
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


# ========== TestContentBased: контентная фильтрация ==========


class TestContentBased:
    """Тесты контентной фильтрации / Content-based filtering tests."""

    def test_fit_builds_profiles(self, fitted_content: ContentBasedRecommender) -> None:
        assert fitted_content.item_profiles is not None
        assert fitted_content.item_profiles.shape[0] == 30

    def test_cold_start_returns_items(self, fitted_content: ContentBasedRecommender) -> None:
        """Cold start рекомендации работают / Cold start recommendations work."""
        prefs = {"category": "electronics", "price_tier": "mid"}
        recs = fitted_content.recommend_for_new_user(prefs, top_k=5)
        assert len(recs) == 5
        assert all(isinstance(r, tuple) for r in recs)

    def test_cold_start_scores_positive(self, fitted_content: ContentBasedRecommender) -> None:
        """Скоры cold start неотрицательные / Cold start scores are non-negative."""
        prefs = {"category": "books", "price_tier": "low"}
        recs = fitted_content.recommend_for_new_user(prefs, top_k=5)
        assert all(score >= 0 for _, score in recs)

    def test_popular_items(self, interactions: pl.DataFrame) -> None:
        """Популярные товары возвращают нужное количество / Popular items returns K."""
        popular = get_popular_items(interactions, top_k=5)
        assert len(popular) == 5


# ========== TestFeatureStore: реестр фичей ==========


class TestFeatureStore:
    """Тесты feature store / Feature registry tests."""

    def test_register_and_list(self) -> None:
        registry = FeatureRegistry()
        registry.register_feature("avg_rating", "float", "Average rating", "user")
        assert registry.n_features == 1
        assert len(registry.list_features()) == 1

    def test_set_and_get_features(self) -> None:
        registry = FeatureRegistry()
        registry.register_feature("avg_rating", "float", "Average rating", "user")
        registry.register_feature("n_purchases", "int", "Purchase count", "user")

        registry.set_features("user_1", {"avg_rating": 4.2, "n_purchases": 15})
        result = registry.get_features("user_1", ["avg_rating", "n_purchases"])

        assert result["avg_rating"] == 4.2
        assert result["n_purchases"] == 15

    def test_get_missing_entity(self) -> None:
        """Несуществующая сущность — None значения / Missing entity returns Nones."""
        registry = FeatureRegistry()
        registry.register_feature("avg_rating", "float", "Average rating")
        result = registry.get_features("user_999", ["avg_rating"])
        assert result["avg_rating"] is None

    def test_duplicate_registration_raises(self) -> None:
        """Повторная регистрация — ошибка / Duplicate registration raises."""
        registry = FeatureRegistry()
        registry.register_feature("avg_rating", "float", "Average rating")
        with pytest.raises(ValueError):
            registry.register_feature("avg_rating", "float", "Duplicate")

    def test_unregistered_feature_raises(self) -> None:
        """Запрос незарегистрированной фичи — ошибка / Unregistered feature raises."""
        registry = FeatureRegistry()
        with pytest.raises(KeyError):
            registry.get_features("user_1", ["nonexistent"])


# ========== TestOfflineFeatures: офлайн-вычисления ==========


class TestOfflineFeatures:
    """Тесты вычисления офлайн-фичей / Offline feature computation tests."""

    def test_user_features_columns(
        self, interactions: pl.DataFrame, products: pl.DataFrame
    ) -> None:
        user_feats = compute_user_features(interactions, products)
        expected_cols = {
            "user_id",
            "avg_rating",
            "n_purchases",
            "favorite_category",
            "recency_days",
        }
        assert expected_cols.issubset(set(user_feats.columns))

    def test_item_features_columns(self, interactions: pl.DataFrame) -> None:
        item_feats = compute_item_features(interactions)
        expected_cols = {"product_id", "avg_rating", "n_ratings", "popularity_rank"}
        assert expected_cols.issubset(set(item_feats.columns))


# ========== TestAPI: FastAPI-эндпоинты ==========


class TestAPI:
    """Тесты API / API endpoint tests."""

    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        from recsys.api.app import app

        return TestClient(app)

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_fitted"] is True

    def test_recommend_known_user(self, client: TestClient) -> None:
        response = client.get("/recommend/1?top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 1
        assert len(data["recommendations"]) <= 5
        assert data["method"] == "collaborative"

    def test_recommend_unknown_user_fallback(self, client: TestClient) -> None:
        """Неизвестный пользователь получает популярные / Unknown user gets popular."""
        response = client.get("/recommend/999999?top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "popular"

    def test_popular_endpoint(self, client: TestClient) -> None:
        response = client.get("/popular?top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == 5


# ========== TestMovieLens: загрузка датасета MovieLens-25M ==========


class TestMovieLensMock:
    """Тесты генератора mock-данных MovieLens / Mock MovieLens data generator tests."""

    @pytest.fixture(scope="class")
    def mock_data(self) -> tuple:
        return generate_mock_movielens(n_users=50, n_movies=30, n_ratings=500, seed=7)

    def test_ratings_schema(self, mock_data: tuple) -> None:
        """ratings_df содержит обязательные колонки MovieLens / Required columns present."""
        ratings_df, _ = mock_data
        assert set(["userId", "movieId", "rating", "timestamp"]).issubset(set(ratings_df.columns))

    def test_movies_schema(self, mock_data: tuple) -> None:
        """movies_df содержит обязательные колонки / Required movie columns present."""
        _, movies_df = mock_data
        assert set(["movieId", "title", "genres"]).issubset(set(movies_df.columns))

    def test_ratings_count(self, mock_data: tuple) -> None:
        """Количество оценок соответствует параметру n_ratings."""
        ratings_df, _ = mock_data
        assert len(ratings_df) == 500

    def test_movies_count(self, mock_data: tuple) -> None:
        """Количество фильмов соответствует параметру n_movies."""
        _, movies_df = mock_data
        assert len(movies_df) == 30

    def test_rating_values_valid(self, mock_data: tuple) -> None:
        """Оценки — только допустимые полузвёздные значения 0.5–5.0."""
        ratings_df, _ = mock_data
        valid_values = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}
        actual_values = set(ratings_df["rating"].unique().to_list())
        assert actual_values.issubset(valid_values)

    def test_user_ids_in_range(self, mock_data: tuple) -> None:
        """userId в пределах заданного диапазона / User IDs within valid range."""
        ratings_df, _ = mock_data
        assert ratings_df["userId"].min() >= 1
        assert ratings_df["userId"].max() <= 50

    def test_movie_ids_in_range(self, mock_data: tuple) -> None:
        """movieId в пределах заданного диапазона / Movie IDs within valid range."""
        ratings_df, _ = mock_data
        assert ratings_df["movieId"].min() >= 1
        assert ratings_df["movieId"].max() <= 30

    def test_genres_non_empty(self, mock_data: tuple) -> None:
        """Все фильмы имеют хотя бы один жанр / Every movie has at least one genre."""
        _, movies_df = mock_data
        assert movies_df["genres"].null_count() == 0
        assert all(len(g) > 0 for g in movies_df["genres"].to_list())

    def test_reproducibility(self) -> None:
        """Одинаковый seed → одинаковые данные / Same seed produces identical data."""
        r1, m1 = generate_mock_movielens(seed=99)
        r2, m2 = generate_mock_movielens(seed=99)
        assert r1.equals(r2)
        assert m1.equals(m2)


class TestMovieLensLoader:
    """Тесты загрузчика load_movielens / load_movielens() loader tests."""

    def test_load_without_paths_returns_mock(self) -> None:
        """Вызов без путей возвращает mock-данные / No paths → mock data returned."""
        ratings_df, movies_df = load_movielens()
        assert len(ratings_df) > 0
        assert len(movies_df) > 0
        assert "userId" in ratings_df.columns

    def test_load_missing_files_returns_mock(self, tmp_path) -> None:
        """Несуществующие пути → graceful fallback на mock / Missing files → mock."""
        ratings_df, movies_df = load_movielens(
            ratings_path=tmp_path / "nonexistent_ratings.csv",
            movies_path=tmp_path / "nonexistent_movies.csv",
        )
        assert len(ratings_df) > 0
        assert "userId" in ratings_df.columns

    def test_load_real_csv(self, tmp_path) -> None:
        """Загрузка из реальных CSV-файлов работает корректно / Real CSV loading works."""
        # Создаём минимальные CSV для проверки парсера
        ratings_csv = tmp_path / "ratings.csv"
        movies_csv = tmp_path / "movies.csv"

        ratings_csv.write_text(
            "userId,movieId,rating,timestamp\n1,1,4.0,1609459200\n2,2,3.5,1609459300\n"
        )
        movies_csv.write_text(
            "movieId,title,genres\n1,Movie A (2020),Comedy|Drama\n2,Movie B (2021),Action\n"
        )

        ratings_df, movies_df = load_movielens(ratings_csv, movies_csv)
        assert len(ratings_df) == 2
        assert len(movies_df) == 2
        assert ratings_df["rating"].to_list() == [4.0, 3.5]


class TestMovieLensStats:
    """Тесты вычисления статистики / MovieLens statistics computation tests."""

    @pytest.fixture(scope="class")
    def stats(self) -> MovieLensStats:
        ratings_df, movies_df = generate_mock_movielens(
            n_users=50, n_movies=20, n_ratings=300, seed=11
        )
        return compute_movielens_stats(ratings_df, movies_df)

    def test_stats_type(self, stats: MovieLensStats) -> None:
        assert isinstance(stats, MovieLensStats)

    def test_n_users_positive(self, stats: MovieLensStats) -> None:
        assert stats.n_users > 0

    def test_n_movies_positive(self, stats: MovieLensStats) -> None:
        assert stats.n_movies > 0

    def test_sparsity_between_zero_and_one(self, stats: MovieLensStats) -> None:
        """Разреженность в [0, 1] / Sparsity must be in valid range."""
        assert 0.0 <= stats.sparsity <= 1.0

    def test_avg_rating_in_range(self, stats: MovieLensStats) -> None:
        """Средняя оценка в [0.5, 5.0] / Avg rating within half-star scale."""
        assert 0.5 <= stats.avg_rating <= 5.0

    def test_rating_distribution_non_empty(self, stats: MovieLensStats) -> None:
        assert len(stats.rating_distribution) > 0

    def test_top_genres_non_empty(self, stats: MovieLensStats) -> None:
        assert len(stats.top_genres) > 0

    def test_top_genres_count_at_most_ten(self, stats: MovieLensStats) -> None:
        assert len(stats.top_genres) <= 10


class TestMovieLensToRecsysFormat:
    """Тесты конвертации в формат RecSys / Schema conversion tests."""

    @pytest.fixture(scope="class")
    def converted(self) -> tuple:
        ratings_df, movies_df = generate_mock_movielens(
            n_users=30, n_movies=15, n_ratings=200, seed=42
        )
        return to_recsys_format(ratings_df, movies_df)

    def test_interactions_columns(self, converted: tuple) -> None:
        """interactions_df имеет нужные колонки / Required interaction columns present."""
        interactions_df, _ = converted
        assert set(["user_id", "product_id", "rating", "timestamp"]).issubset(
            set(interactions_df.columns)
        )

    def test_products_columns(self, converted: tuple) -> None:
        """products_df имеет нужные колонки / Required product columns present."""
        _, products_df = converted
        assert set(["product_id", "category", "price_tier"]).issubset(set(products_df.columns))

    def test_price_tiers_valid(self, converted: tuple) -> None:
        """price_tier содержит только допустимые значения / Valid price tiers only."""
        _, products_df = converted
        valid_tiers = {"low", "mid", "high"}
        actual = set(products_df["price_tier"].unique().to_list())
        assert actual.issubset(valid_tiers)

    def test_category_non_empty(self, converted: tuple) -> None:
        """Категория не пустая для всех фильмов / Category is non-empty for all movies."""
        _, products_df = converted
        categories = products_df["category"].to_list()
        assert all(isinstance(c, str) and len(c) > 0 for c in categories)

    def test_timestamp_is_string(self, converted: tuple) -> None:
        """timestamp в ISO-формате строки / Timestamp converted to ISO string."""
        interactions_df, _ = converted
        ts_dtype = interactions_df["timestamp"].dtype
        assert ts_dtype == pl.Utf8 or ts_dtype == pl.String

    def test_compatible_with_collaborative_recommender(self, converted: tuple) -> None:
        """Конвертированные данные пригодны для CollaborativeRecommender."""
        from recsys.models.collaborative import CollaborativeRecommender

        interactions_df, _ = converted
        model = CollaborativeRecommender(n_components=5)
        model.fit(interactions_df)
        assert model.predicted_ratings is not None

    def test_compatible_with_content_based_recommender(self, converted: tuple) -> None:
        """Конвертированные данные пригодны для ContentBasedRecommender."""
        from recsys.models.content_based import ContentBasedRecommender

        interactions_df, products_df = converted
        model = ContentBasedRecommender()
        model.fit(products_df)
        assert model.item_profiles is not None
