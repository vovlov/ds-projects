"""
FastAPI-сервис рекомендаций.
Recommendation API — serves personalized and popular item recommendations.

Для известных пользователей — коллаборативная фильтрация,
для новых — популярные товары (cold start fallback).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recsys.data.load import load_all_data
from recsys.feature_store.feast_bridge import (
    FeastBridge,
    FeatureRequest,
    OnlineFeatureResponse,
)
from recsys.feature_store.offline import (
    compute_item_features,
    compute_user_features,
    populate_registry,
)
from recsys.feature_store.registry import FeatureRegistry
from recsys.feature_store.wap import WAPGate
from recsys.models.collaborative import CollaborativeRecommender
from recsys.models.content_based import get_popular_items

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


class WAPRequest(BaseModel):
    """Запрос на WAP drift check для батча фичей.
    Request body for Write-Audit-Publish drift gate."""

    feature_name: str
    values: list[float]
    reference: list[float] | None = None
    psi_threshold: float = 0.2


class WAPResponse(BaseModel):
    """Результат WAP drift check.
    Write-Audit-Publish drift gate result."""

    draft_id: str
    feature_name: str
    status: str  # "published" | "quarantined" | "no_reference"
    psi: float
    threshold: float
    passed: bool
    n_reference: int
    n_current: int
    timestamp: str
    reason: str


class HealthResponse(BaseModel):
    """Статус здоровья сервиса / Health check response."""

    status: str
    n_users: int
    n_products: int
    model_fitted: bool


class OnlineFeatureRequest(BaseModel):
    """
    Запрос на получение online-фичей (Feast-совместимый API).
    Online feature request — Feast-compatible API.

    Пример / Example:
        {
          "entity_rows": [{"user_id": 1}, {"user_id": 2}],
          "features": ["user_features:avg_rating", "user_features:n_purchases"]
        }
    """

    entity_rows: list[dict[str, int | str | float]]
    features: list[str]  # "view:feature_name" format


class EntityFeatureRowResponse(BaseModel):
    """Строка с фичами для одной сущности / Feature row for one entity."""

    entity_key: str
    entity_row: dict[str, int | str | float]
    feature_values: dict[str, float | int | str | None]
    missing_features: list[str]


class OnlineFeaturesResponse(BaseModel):
    """
    Ответ Feast-compatible /features/online.
    Response from Feast-compatible online feature serving endpoint.
    """

    rows: list[EntityFeatureRowResponse]
    feature_refs: list[str]
    n_entities: int
    n_missing: int


# --- Инициализация при старте / Startup initialization ---

# Глобальные объекты — инициализируем лениво
_recommender: CollaborativeRecommender | None = None
_interactions = None
_users = None
_products = None

# WAP gate — singleton, хранит reference между вызовами
# WAP gate singleton — persists reference distributions across requests
_wap_gate: WAPGate | None = None

# Feast bridge — Feast-compatible online feature serving
_feast_bridge: FeastBridge | None = None


def _get_wap_gate(psi_threshold: float = 0.2) -> WAPGate:
    """Ленивая инициализация WAP gate / Lazy WAP gate initialization."""
    global _wap_gate
    if _wap_gate is None:
        _wap_gate = WAPGate(psi_threshold=psi_threshold)
    return _wap_gate


def _get_feast_bridge() -> FeastBridge:
    """
    Ленивая инициализация FeastBridge с вычисленными фичами.
    Lazy FeastBridge initialization — computes features from interaction data.
    """
    global _feast_bridge, _interactions, _users, _products
    if _feast_bridge is None:
        # Убедимся что данные загружены / ensure data is loaded
        _get_recommender()

        registry = FeatureRegistry()
        user_feats = compute_user_features(_interactions, _products)
        item_feats = compute_item_features(_interactions)
        populate_registry(registry, user_feats, item_feats)
        _feast_bridge = FeastBridge(registry=registry)
    return _feast_bridge


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


@app.post("/features/wap", response_model=WAPResponse)
def features_wap(request: WAPRequest) -> WAPResponse:
    """
    Write-Audit-Publish drift gate для батча фичей.
    WAP drift gate for a feature batch.

    Принимает батч новых значений фичи и проверяет PSI-дрейф
    относительно сохранённого reference-распределения.
    Если drift < psi_threshold — публикует; иначе — карантин.

    Accepts a new feature batch and checks PSI drift against stored reference.
    If PSI < psi_threshold → "published"; else → "quarantined".

    Поле reference (опционально):
    - Если передано — устанавливается как новый reference перед аудитом.
    - Если не передано — используется ранее установленное (или холодный старт).

    Optional `reference` field:
    - If provided — set as new reference before auditing.
    - If not provided — use previously stored reference (or cold start).
    """
    gate = _get_wap_gate(psi_threshold=request.psi_threshold)

    # Позволяем переопределить reference через запрос (например, из training pipeline)
    # Allow caller to update reference (e.g. after model retraining)
    if request.reference is not None:
        gate.set_reference(request.feature_name, request.reference)

    result = gate.write_audit_publish(request.feature_name, request.values)
    return WAPResponse(**result.to_dict())


@app.post("/features/online", response_model=OnlineFeaturesResponse)
def features_online(request: OnlineFeatureRequest) -> OnlineFeaturesResponse:
    """
    Feast-совместимое получение online-фичей.
    Feast-compatible online feature serving — get_online_features() API.

    Возвращает фичи для набора сущностей из внутреннего FeatureRegistry.
    Формат ссылки: "feature_view:feature_name" (например "user_features:avg_rating").
    Неизвестные сущности возвращают None для отсутствующих фичей.

    Returns features for a set of entities from the internal FeatureRegistry.
    Reference format: "feature_view:feature_name" (e.g. "user_features:avg_rating").
    Unknown entities return None for missing features.

    Поддерживаемые view / Supported views:
      - user_features: avg_rating, n_purchases, favorite_category, recency_days
      - item_features: item_avg_rating, n_ratings, popularity_rank
    """
    bridge = _get_feast_bridge()

    feast_request = FeatureRequest(
        entity_rows=[dict(row) for row in request.entity_rows],
        features=request.features,
    )

    try:
        response: OnlineFeatureResponse = bridge.get_online_features(feast_request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    rows = [
        EntityFeatureRowResponse(
            entity_key=r.entity_key,
            entity_row=r.entity_row,  # type: ignore[arg-type]
            feature_values=r.feature_values,  # type: ignore[arg-type]
            missing_features=r.missing_features,
        )
        for r in response.rows
    ]

    return OnlineFeaturesResponse(
        rows=rows,
        feature_refs=response.feature_refs,
        n_entities=response.n_entities,
        n_missing=response.n_missing,
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
