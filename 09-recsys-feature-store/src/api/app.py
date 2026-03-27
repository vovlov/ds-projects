"""
FastAPI-сервис рекомендаций.
Recommendation API — serves personalized and popular item recommendations.

Для известных пользователей — коллаборативная фильтрация,
для новых — популярные товары (cold start fallback).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.data.load import load_all_data
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import get_popular_items

app = FastAPI(
    title="RecSys API",
    description="E-commerce recommendation engine / Рекомендательная система для e-commerce",
    version="0.1.0",
)


# --- Pydantic-модели ответов / Response models ---


class RecommendationItem(BaseModel):
    """Один рекомендованный товар / Single recommendation."""

    product_id: int
    score: float


class RecommendationResponse(BaseModel):
    """Ответ с рекомендациями / Recommendation response."""

    user_id: int
    recommendations: list[RecommendationItem]
    method: str  # "collaborative" или "popular"


class PopularResponse(BaseModel):
    """Ответ с популярными товарами / Popular items response."""

    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Статус здоровья сервиса / Health check response."""

    status: str
    n_users: int
    n_products: int
    model_fitted: bool


# --- Инициализация при старте / Startup initialization ---

# Глобальные объекты — инициализируем лениво
_recommender: CollaborativeRecommender | None = None
_interactions = None
_users = None
_products = None


def _get_recommender() -> CollaborativeRecommender:
    """Ленивая инициализация модели (singleton).
    Lazy model initialization — loads data and fits on first call."""
    global _recommender, _interactions, _users, _products

    if _recommender is None:
        _users, _products, _interactions = load_all_data()
        _recommender = CollaborativeRecommender(n_components=30)
        _recommender.fit(_interactions)

    return _recommender


# --- Эндпоинты / Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Проверка здоровья сервиса / Health check endpoint."""
    rec = _get_recommender()
    return HealthResponse(
        status="ok",
        n_users=len(rec.user_to_idx),
        n_products=len(rec.product_to_idx),
        model_fitted=rec.predicted_ratings is not None,
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int,
    top_k: int = Query(default=10, ge=1, le=100),
) -> RecommendationResponse:
    """Персональные рекомендации для пользователя.
    Get personalized recommendations. Falls back to popular items for unknown users."""
    rec = _get_recommender()

    if user_id in rec.user_to_idx:
        # Известный пользователь — коллаборативная фильтрация
        results = rec.recommend(user_id, top_k=top_k)
        method = "collaborative"
    else:
        # Новый пользователь — популярные товары
        if _interactions is None:
            raise HTTPException(status_code=500, detail="Data not loaded")
        results = get_popular_items(_interactions, top_k=top_k)
        method = "popular"

    items = [RecommendationItem(product_id=pid, score=round(score, 4)) for pid, score in results]

    return RecommendationResponse(
        user_id=user_id,
        recommendations=items,
        method=method,
    )


@app.get("/popular", response_model=PopularResponse)
def popular(
    top_k: int = Query(default=10, ge=1, le=100),
) -> PopularResponse:
    """Популярные товары (cold start / baseline).
    Get most popular items — works without user context."""
    _get_recommender()  # Убеждаемся что данные загружены

    if _interactions is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    results = get_popular_items(_interactions, top_k=top_k)
    items = [RecommendationItem(product_id=pid, score=round(score, 4)) for pid, score in results]

    return PopularResponse(recommendations=items)
