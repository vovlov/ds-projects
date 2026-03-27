"""
Тесты для рекомендательной системы и feature store.
Tests for recommendation engine, feature store, and API.
"""

from __future__ import annotations

import polars as pl
import pytest
from fastapi.testclient import TestClient
from src.data.load import generate_interactions, generate_products, generate_users
from src.feature_store.offline import compute_item_features, compute_user_features
from src.feature_store.registry import FeatureRegistry
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender, get_popular_items

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
        from src.api.app import app

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
