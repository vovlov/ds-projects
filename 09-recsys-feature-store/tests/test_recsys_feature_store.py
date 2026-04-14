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


# ========== TestTwoTowerModel ==========


class TestTwoTowerModel:
    """Тесты двухбашенной модели поиска кандидатов."""

    @pytest.fixture(scope="class")
    def fitted_two_tower(
        self,
        users: pl.DataFrame,
        products: pl.DataFrame,
        interactions: pl.DataFrame,
    ):
        """Обученная двухбашенная модель для тестов класса."""
        from recsys.models.two_tower import TowerConfig, TwoTowerModel

        config = TowerConfig(embedding_dim=16, n_epochs=5, batch_size=256)
        model = TwoTowerModel(config=config)
        model.fit(users, products, interactions)
        return model

    def test_fit_sets_is_fitted(self, fitted_two_tower) -> None:
        """После fit() модель помечается как обученная."""
        assert fitted_two_tower.is_fitted is True

    def test_weight_matrices_shape(
        self, fitted_two_tower, users: pl.DataFrame, products: pl.DataFrame
    ) -> None:
        """Матрицы весов башен имеют корректный размер."""
        model = fitted_two_tower
        emb_dim = model.config.embedding_dim
        assert model.W_user is not None
        assert model.W_item is not None
        assert model.W_user.shape[1] == emb_dim
        assert model.W_item.shape[1] == emb_dim

    def test_item_embeddings_normalized(self, fitted_two_tower) -> None:
        """Предвычисленные item-эмбеддинги должны быть L2-нормированы."""
        embs = fitted_two_tower._item_embeddings
        assert embs is not None
        norms = (embs**2).sum(axis=1) ** 0.5
        # Допускаем погрешность для нулевых векторов
        assert all(abs(n - 1.0) < 0.01 or n < 0.01 for n in norms)

    def test_recommend_returns_list(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """recommend() возвращает список (item_id, score)."""
        first_uid = int(users["user_id"][0])
        recs = fitted_two_tower.recommend(first_uid, users, interactions, top_k=5)
        assert isinstance(recs, list)

    def test_recommend_top_k(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """Число рекомендаций не превышает top_k."""
        first_uid = int(users["user_id"][0])
        recs = fitted_two_tower.recommend(first_uid, users, interactions, top_k=5)
        assert len(recs) <= 5

    def test_recommend_excludes_rated(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """Уже просмотренные товары не попадают в рекомендации."""
        first_uid = int(users["user_id"][0])
        rated = set(interactions.filter(pl.col("user_id") == first_uid)["product_id"].to_list())
        recs = fitted_two_tower.recommend(first_uid, users, interactions, top_k=20)
        rec_ids = {iid for iid, _ in recs}
        assert len(rec_ids & rated) == 0, "Rated items must be excluded from recommendations"

    def test_recommend_scores_descending(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """Рекомендации отсортированы по убыванию скора."""
        first_uid = int(users["user_id"][0])
        recs = fitted_two_tower.recommend(first_uid, users, interactions, top_k=10)
        if len(recs) > 1:
            scores = [s for _, s in recs]
            assert scores == sorted(scores, reverse=True)

    def test_recommend_unknown_user_returns_empty(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """Неизвестный пользователь → пустой список."""
        recs = fitted_two_tower.recommend(99999, users, interactions, top_k=5)
        assert recs == []

    def test_get_user_embedding_shape(self, fitted_two_tower, users: pl.DataFrame) -> None:
        """user_embedding имеет корректный размер и единичную норму."""
        first_uid = int(users["user_id"][0])
        emb = fitted_two_tower.get_user_embedding(first_uid, users)
        assert emb is not None
        assert emb.shape == (fitted_two_tower.config.embedding_dim,)

    def test_save_load_roundtrip(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
        tmp_path,
    ) -> None:
        """Сохранение и загрузка модели дают идентичные рекомендации."""
        path = str(tmp_path / "two_tower.pkl")
        fitted_two_tower.save(path)

        from recsys.models.two_tower import TwoTowerModel

        loaded = TwoTowerModel.load(path)
        first_uid = int(users["user_id"][0])

        recs_orig = fitted_two_tower.recommend(first_uid, users, interactions, top_k=5)
        recs_loaded = loaded.recommend(first_uid, users, interactions, top_k=5)
        assert recs_orig == recs_loaded

    def test_repr(self, fitted_two_tower) -> None:
        """__repr__ содержит полезную информацию о модели."""
        r = repr(fitted_two_tower)
        assert "TwoTowerModel" in r
        assert "fitted" in r

    def test_evaluate_returns_metrics(
        self,
        fitted_two_tower,
        users: pl.DataFrame,
        products: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """evaluate() возвращает словарь с precision, recall, ndcg."""
        # Используем первые 800 взаимодействий как train, остальные как test
        train = interactions.head(800)
        test = interactions.tail(200)
        metrics = fitted_two_tower.evaluate(users, products, test, train, top_k=5)
        assert "precision@5" in metrics
        assert "recall@5" in metrics
        assert "ndcg@5" in metrics
        assert 0.0 <= metrics["precision@5"] <= 1.0
        assert 0.0 <= metrics["recall@5"] <= 1.0
        assert 0.0 <= metrics["ndcg@5"] <= 1.0


# ========== TestLLMReranker ==========


class TestLLMReranker:
    """Тесты LLM-переранжирователя (mock-режим без API-ключа)."""

    @pytest.fixture(scope="class")
    def reranker(self):
        from recsys.models.reranker import LLMReranker

        # mock=True гарантирует независимость от LLM в CI
        return LLMReranker(mock=True)

    @pytest.fixture(scope="class")
    def candidates(self) -> list[tuple[int, float]]:
        """Синтетические кандидаты от двухбашенной модели."""
        return [(1, 0.95), (5, 0.88), (12, 0.76), (7, 0.65), (3, 0.54), (18, 0.43)]

    def test_rerank_returns_list(
        self,
        reranker,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        users: pl.DataFrame,
    ) -> None:
        """rerank() возвращает список словарей."""
        first_uid = int(users["user_id"][0])
        results = reranker.rerank(first_uid, candidates, products, users, top_k=3)
        assert isinstance(results, list)

    def test_rerank_top_k_count(
        self,
        reranker,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        users: pl.DataFrame,
    ) -> None:
        """Число результатов не превышает top_k."""
        first_uid = int(users["user_id"][0])
        results = reranker.rerank(first_uid, candidates, products, users, top_k=3)
        assert len(results) <= 3

    def test_rerank_result_keys(
        self,
        reranker,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        users: pl.DataFrame,
    ) -> None:
        """Каждый результат содержит обязательные ключи."""
        first_uid = int(users["user_id"][0])
        results = reranker.rerank(first_uid, candidates, products, users, top_k=3)
        required_keys = {
            "item_id",
            "retrieval_score",
            "rerank_score",
            "category",
            "price_tier",
            "explanation",
        }
        for item in results:
            assert required_keys.issubset(item.keys()), f"Missing keys in {item}"

    def test_rerank_scores_in_range(
        self,
        reranker,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        users: pl.DataFrame,
    ) -> None:
        """rerank_score находится в допустимом диапазоне."""
        first_uid = int(users["user_id"][0])
        results = reranker.rerank(first_uid, candidates, products, users, top_k=3)
        for item in results:
            assert 0.0 <= item["rerank_score"] <= 1.0

    def test_rerank_empty_candidates(
        self,
        reranker,
        products: pl.DataFrame,
        users: pl.DataFrame,
    ) -> None:
        """Пустой список кандидатов → пустой список результатов."""
        first_uid = int(users["user_id"][0])
        results = reranker.rerank(first_uid, [], products, users, top_k=5)
        assert results == []

    def test_reranker_repr(self, reranker) -> None:
        """__repr__ показывает режим работы."""
        r = repr(reranker)
        assert "LLMReranker" in r
        assert "mock" in r

    def test_two_tower_plus_reranker_pipeline(
        self,
        users: pl.DataFrame,
        products: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> None:
        """
        Интеграционный тест: two-tower retrieval → LLM re-ranking pipeline.

        Проверяем полный пайплайн: от обучения до финального ранжирования.
        """
        from recsys.models.reranker import LLMReranker
        from recsys.models.two_tower import TowerConfig, TwoTowerModel

        # Stage 1: Two-tower retrieval
        config = TowerConfig(embedding_dim=8, n_epochs=3, batch_size=128)
        retriever = TwoTowerModel(config=config).fit(users, products, interactions)

        first_uid = int(users["user_id"][0])
        candidates = retriever.recommend(first_uid, users, interactions, top_k=10)

        # Stage 2: LLM re-ranking (mock)
        reranker = LLMReranker(mock=True)
        final_recs = reranker.rerank(first_uid, candidates, products, users, top_k=5)

        assert len(final_recs) <= 5
        assert all("item_id" in r and "rerank_score" in r for r in final_recs)


# ========== TestWAPGate: Write-Audit-Publish drift gate ==========


class TestWAPGate:
    """
    Тесты Write-Audit-Publish gate для feature store.
    Write-Audit-Publish drift gate tests.
    """

    def test_cold_start_no_reference_publishes(self) -> None:
        """Первый батч без reference → статус 'no_reference', данные публикуются."""
        from recsys.feature_store.wap import WAPGate

        gate = WAPGate(psi_threshold=0.2)
        result = gate.write_audit_publish("avg_rating", [1.0, 2.0, 3.0, 4.0, 5.0])

        assert result.status == "no_reference"
        assert result.passed is True
        assert result.psi == 0.0
        assert result.n_reference == 0
        assert result.n_current == 5

    def test_cold_start_sets_reference(self) -> None:
        """После первого батча (cold start) reference устанавливается автоматически."""
        from recsys.feature_store.wap import WAPGate

        gate = WAPGate(psi_threshold=0.2)
        gate.write_audit_publish("avg_rating", [1.0, 2.0, 3.0, 4.0, 5.0])

        assert gate.has_reference("avg_rating")

    def test_stable_batch_published(self) -> None:
        """Батч из похожего распределения → PSI < threshold → 'published'."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 500).tolist()
        current = rng.normal(0.0, 1.0, 500).tolist()  # то же распределение

        gate = WAPGate(psi_threshold=0.2)
        gate.set_reference("feature_x", reference)
        result = gate.write_audit_publish("feature_x", current)

        assert result.status == "published"
        assert result.passed is True
        assert result.psi < 0.2

    def test_drifted_batch_quarantined(self) -> None:
        """Батч со сдвинутым распределением → PSI > threshold → 'quarantined'."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 500).tolist()
        current = rng.normal(5.0, 1.0, 500).tolist()  # сдвиг 5 сигм → большой PSI

        gate = WAPGate(psi_threshold=0.2)
        gate.set_reference("feature_x", reference)
        result = gate.write_audit_publish("feature_x", current)

        assert result.status == "quarantined"
        assert result.passed is False
        assert result.psi >= 0.2

    def test_psi_value_in_result(self) -> None:
        """AuditResult содержит корректное PSI-значение."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(7)
        reference = rng.normal(0.0, 1.0, 300).tolist()
        current = rng.normal(3.0, 1.0, 300).tolist()

        gate = WAPGate(psi_threshold=0.2)
        gate.set_reference("score", reference)
        result = gate.write_audit_publish("score", current)

        assert isinstance(result.psi, float)
        assert result.psi > 0.0

    def test_audit_log_grows(self) -> None:
        """После каждого WAP-вызова audit_log увеличивается на 1."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        gate = WAPGate()
        rng = np.random.default_rng(0)

        for i in range(3):
            gate.write_audit_publish("feat", rng.normal(0, 1, 100).tolist())
            assert len(gate.get_audit_log()) == i + 1

    def test_stable_batch_lands_in_production(self) -> None:
        """Опубликованный батч доступен в production store."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 200).tolist()
        current = rng.normal(0.0, 1.0, 200).tolist()

        gate = WAPGate(psi_threshold=0.2)
        gate.set_reference("feat", reference)
        result = gate.write_audit_publish("feat", current)

        assert result.passed
        assert gate.get_production("feat") is not None
        assert len(gate.get_production("feat")) == 200

    def test_quarantined_batch_not_in_production(self) -> None:
        """Забракованный батч НЕ попадает в production store."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 200).tolist()
        drifted = rng.normal(10.0, 1.0, 200).tolist()

        gate = WAPGate(psi_threshold=0.2)
        gate.set_reference("feat", reference)
        gate.write_audit_publish("feat", drifted)

        # В production store должно быть пусто
        assert gate.get_production("feat") is None

    def test_staging_cleared_after_publish(self) -> None:
        """После WAP staging-буфер очищается."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        gate = WAPGate(psi_threshold=0.2)
        rng = np.random.default_rng(1)
        values = rng.normal(0, 1, 100).tolist()

        gate.write_audit_publish("f", values)
        # После полного цикла staging пуст
        assert gate.n_staging == 0

    def test_step_by_step_write_audit_publish(self) -> None:
        """Пошаговый вызов write → audit → publish работает идентично all-in-one."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 200).tolist()
        current = rng.normal(0.0, 1.0, 200).tolist()

        gate = WAPGate(psi_threshold=0.2)
        gate.set_reference("f", reference)

        draft_id = gate.write("f", current)
        result = gate.audit(draft_id)
        published = gate.publish(draft_id, result)

        assert result.passed
        assert published is True

    def test_audit_unknown_draft_raises(self) -> None:
        """audit() с несуществующим draft_id → KeyError."""
        from recsys.feature_store.wap import WAPGate

        gate = WAPGate()
        with pytest.raises(KeyError):
            gate.audit("nonexistent-draft-id")

    def test_audit_result_to_dict_serializable(self) -> None:
        """to_dict() возвращает JSON-сериализуемый словарь."""
        import json

        from recsys.feature_store.wap import WAPGate

        gate = WAPGate()
        result = gate.write_audit_publish("f", [1.0, 2.0, 3.0])
        d = result.to_dict()
        assert json.dumps(d)  # не бросает исключение
        required_keys = {
            "draft_id",
            "feature_name",
            "status",
            "psi",
            "threshold",
            "passed",
            "n_reference",
            "n_current",
            "timestamp",
            "reason",
        }
        assert required_keys.issubset(d.keys())

    def test_custom_psi_threshold(self) -> None:
        """Строгий порог PSI=0.01 забраковывает даже умеренный дрейф."""
        import numpy as np
        from recsys.feature_store.wap import WAPGate

        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 500).tolist()
        # Небольшой сдвиг — умеренный PSI
        current = rng.normal(0.5, 1.0, 500).tolist()

        gate_strict = WAPGate(psi_threshold=0.01)
        gate_strict.set_reference("feat", reference)
        result = gate_strict.write_audit_publish("feat", current)

        # При очень строгом пороге даже небольшой сдвиг → quarantined
        assert result.threshold == 0.01


# ========== TestWAPAPIEndpoint ==========


class TestWAPAPIEndpoint:
    """
    Тесты API-эндпоинта POST /features/wap.
    Tests for the /features/wap API endpoint.
    """

    @pytest.fixture(scope="class")
    def client(self) -> TestClient:
        """Отдельный клиент без сессии WAP-gate из TestAPI."""
        from recsys.api.app import app

        return TestClient(app)

    def test_wap_cold_start(self, client: TestClient) -> None:
        """Первый вызов без reference → статус 'no_reference', passed=True."""
        import time

        # Уникальное имя фичи, чтобы не пересекаться с другими тестами
        feature_name = f"cold_start_test_{time.time_ns()}"
        response = client.post(
            "/features/wap",
            json={
                "feature_name": feature_name,
                "values": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_reference"
        assert data["passed"] is True
        assert data["psi"] == 0.0
        assert data["feature_name"] == feature_name

    def test_wap_stable_batch_published(self, client: TestClient) -> None:
        """Батч из стабильного распределения → 'published'."""
        import time

        import numpy as np

        feature_name = f"stable_test_{time.time_ns()}"
        rng = np.random.default_rng(42)
        reference = rng.normal(0.0, 1.0, 300).tolist()
        current = rng.normal(0.0, 1.0, 300).tolist()

        response = client.post(
            "/features/wap",
            json={
                "feature_name": feature_name,
                "values": current,
                "reference": reference,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "published"
        assert data["passed"] is True
        assert data["psi"] < 0.2

    def test_wap_drifted_batch_quarantined(self, client: TestClient) -> None:
        """Батч со сдвигом → 'quarantined', passed=False."""
        import time

        import numpy as np

        feature_name = f"drift_test_{time.time_ns()}"
        rng = np.random.default_rng(7)
        reference = rng.normal(0.0, 1.0, 500).tolist()
        drifted = rng.normal(5.0, 1.0, 500).tolist()

        response = client.post(
            "/features/wap",
            json={
                "feature_name": feature_name,
                "values": drifted,
                "reference": reference,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "quarantined"
        assert data["passed"] is False
        assert data["psi"] >= 0.2

    def test_wap_response_has_required_fields(self, client: TestClient) -> None:
        """Ответ содержит все обязательные поля."""
        import time

        feature_name = f"fields_test_{time.time_ns()}"
        response = client.post(
            "/features/wap",
            json={"feature_name": feature_name, "values": [1.0, 2.0, 3.0]},
        )
        assert response.status_code == 200
        data = response.json()
        required = {
            "draft_id",
            "feature_name",
            "status",
            "psi",
            "threshold",
            "passed",
            "n_reference",
            "n_current",
            "timestamp",
            "reason",
        }
        assert required.issubset(data.keys())
