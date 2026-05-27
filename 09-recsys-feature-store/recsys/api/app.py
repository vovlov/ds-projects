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
from recsys.models.bandit import BanditConfig, BanditResult, LinUCBBandit
from recsys.models.collaborative import CollaborativeRecommender
from recsys.models.content_based import get_popular_items
from recsys.models.diversity import DiversityConfig, DiversityResult, MMRDiversifier

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

# LinUCB bandit — singleton, хранит arm statistics между запросами
# LinUCB bandit singleton — persists A, b matrices across requests
_bandit: LinUCBBandit | None = None

# MMR Diversifier singleton
_mmr_diversifier: MMRDiversifier | None = None


def _get_wap_gate(psi_threshold: float = 0.2) -> WAPGate:
    """Ленивая инициализация WAP gate / Lazy WAP gate initialization."""
    global _wap_gate
    if _wap_gate is None:
        _wap_gate = WAPGate(psi_threshold=psi_threshold)
    return _wap_gate


def _get_bandit(alpha: float = 1.0, feature_dim: int = 8) -> LinUCBBandit:
    """Ленивая инициализация LinUCB bandit / Lazy LinUCB initialization."""
    global _bandit
    if _bandit is None:
        _bandit = LinUCBBandit(BanditConfig(alpha=alpha, feature_dim=feature_dim))
    return _bandit


def _reset_bandit() -> None:
    """Сброс bandit для тестовой изоляции / Reset bandit for test isolation."""
    global _bandit
    _bandit = None


def _get_diversifier(embedding_dim: int = 8) -> MMRDiversifier:
    """Ленивая инициализация MMR diversifier / Lazy MMR diversifier initialization."""
    global _mmr_diversifier
    if _mmr_diversifier is None:
        _mmr_diversifier = MMRDiversifier(DiversityConfig(embedding_dim=embedding_dim))
    return _mmr_diversifier


def _reset_diversifier() -> None:
    """Сброс diversifier для тестовой изоляции / Reset for test isolation."""
    global _mmr_diversifier
    _mmr_diversifier = None


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


# ---------------------------------------------------------------------------
# LinUCB Contextual Bandit endpoints
# ---------------------------------------------------------------------------


class BanditRecommendationResponse(BaseModel):
    """Одна рекомендация с UCB-компонентами. / Single UCB recommendation."""

    arm_id: int
    ucb_score: float
    expected_reward: float
    exploration_bonus: float
    n_updates: int


class BanditRecommendRequest(BaseModel):
    """
    Запрос на ранжирование кандидатов через LinUCB.
    Request to rank candidate items via LinUCB bandit.

    Пример / Example:
        {
          "candidate_ids": [101, 202, 303],
          "candidate_contexts": [[0.8, 0.2, 0.5, 0.1, 0.9, 0.3, 0.4, 0.7], ...],
          "top_k": 3
        }

    candidate_contexts — конкатенация user_features + item_features.
    Длина каждого вектора должна соответствовать feature_dim bandit (по умолчанию 8).
    """

    candidate_ids: list[int]
    candidate_contexts: list[list[float]]
    top_k: int = 10
    alpha: float = 1.0
    feature_dim: int = 8


class BanditRecommendResponse(BaseModel):
    """Результат LinUCB ранжирования. / LinUCB ranking result."""

    recommendations: list[BanditRecommendationResponse]
    n_arms_scored: int
    top_arm_id: int
    config_alpha: float


class BanditFeedbackRequest(BaseModel):
    """
    Обратная связь для обновления LinUCB arm.
    Feedback payload to update LinUCB arm statistics.

    reward: 1.0 = клик/покупка, 0.0 = пропуск, или CTR ∈ [0,1].
    Контекст должен совпадать с тем, что передавался при recommend.
    """

    arm_id: int
    context: list[float]
    reward: float
    feature_dim: int = 8


class BanditFeedbackResponse(BaseModel):
    """Статус обновления arm. / Arm update status."""

    arm_id: int
    n_updates: int
    total_reward: float
    message: str


class BanditStatsResponse(BaseModel):
    """Статистика bandit для мониторинга. / Bandit stats for monitoring."""

    n_arms: int
    total_recommendations: int
    config_alpha: float
    arm_stats: list[dict]


@app.post("/bandit/recommend", response_model=BanditRecommendResponse)
def bandit_recommend(request: BanditRecommendRequest) -> BanditRecommendResponse:
    """
    Ранжировать кандидатов через LinUCB Contextual Bandit.
    Rank candidate items using LinUCB exploration-exploitation.

    На старте (cold start) все arms имеют одинаковое UCB → ранжирование
    по exploration_bonus (случайное исследование). После накопления фидбека
    модель переходит к exploitation лучших arms.

    Cold start: all arms start equal → ranked by exploration bonus.
    After feedback accumulates, model shifts to exploitation of high-reward arms.
    """
    if len(request.candidate_ids) != len(request.candidate_contexts):
        raise HTTPException(
            status_code=422,
            detail="candidate_ids and candidate_contexts must have equal length",
        )
    if not request.candidate_ids:
        raise HTTPException(status_code=422, detail="candidate_ids must not be empty")

    bandit = _get_bandit(alpha=request.alpha, feature_dim=request.feature_dim)
    result: BanditResult = bandit.recommend(
        candidate_ids=request.candidate_ids,
        candidate_contexts=request.candidate_contexts,
        top_k=request.top_k,
    )

    recs = [
        BanditRecommendationResponse(
            arm_id=r.arm_id,
            ucb_score=r.ucb_score,
            expected_reward=r.expected_reward,
            exploration_bonus=r.exploration_bonus,
            n_updates=r.n_updates,
        )
        for r in result.recommendations
    ]

    return BanditRecommendResponse(
        recommendations=recs,
        n_arms_scored=result.n_arms_scored,
        top_arm_id=result.top_arm_id,
        config_alpha=result.config_alpha,
    )


@app.post("/bandit/feedback", response_model=BanditFeedbackResponse)
def bandit_feedback(request: BanditFeedbackRequest) -> BanditFeedbackResponse:
    """
    Обновить LinUCB arm после получения обратной связи.
    Update LinUCB arm statistics after receiving user feedback.

    Вызывать после того, как пользователь среагировал (или нет) на рекомендацию.
    Call this after the user reacted (or ignored) a recommendation.

    reward=1.0 — клик/покупка; reward=0.0 — пропуск.
    """
    if not (0.0 <= request.reward <= 1.0):
        raise HTTPException(status_code=422, detail="reward must be in [0.0, 1.0]")

    bandit = _get_bandit(feature_dim=request.feature_dim)
    bandit.update(
        arm_id=request.arm_id,
        context=request.context,
        reward=request.reward,
    )

    arm_state = bandit._arms[request.arm_id]
    return BanditFeedbackResponse(
        arm_id=request.arm_id,
        n_updates=arm_state.n_updates,
        total_reward=round(arm_state.total_reward, 4),
        message=f"arm {request.arm_id} updated (n_updates={arm_state.n_updates})",
    )


@app.get("/bandit/stats", response_model=BanditStatsResponse)
def bandit_stats() -> BanditStatsResponse:
    """
    Статистика LinUCB bandit для мониторинга и отладки.
    LinUCB bandit statistics for monitoring and debugging.

    Показывает количество обновлений и среднее вознаграждение по каждому arm.
    Полезно для выявления arms с плохим качеством или недостаточным исследованием.
    """
    bandit = _get_bandit()
    return BanditStatsResponse(
        n_arms=bandit.n_arms,
        total_recommendations=bandit.total_recommendations,
        config_alpha=bandit.config.alpha,
        arm_stats=bandit.get_arm_stats(),
    )


# --- MMR Diversity Reranking ---


class DiverseRecommendRequest(BaseModel):
    """Запрос на MMR-диверсификацию списка рекомендаций.
    Request for MMR diversity reranking of a candidate list."""

    candidate_ids: list[int]
    relevance_scores: list[float]
    lambda_param: float = 0.5
    """Trade-off: 1.0 = pure relevance, 0.0 = pure diversity."""
    n_items: int = 10
    embedding_dim: int = 8
    categories: list[str] | None = None
    """Optional category label per candidate (same order as candidate_ids)."""
    price_tiers: list[str] | None = None
    """Optional price tier per candidate: 'low' | 'medium' | 'high'."""


class DiverseItemResponse(BaseModel):
    """Один товар в диверсифицированном списке / Single item in diverse list."""

    item_id: int
    relevance_score: float
    diversity_contribution: float
    mmr_score: float
    rank: int


class DiversityMetricsResponse(BaseModel):
    """Агрегированные метрики разнообразия / Aggregate diversity metrics."""

    intra_list_diversity: float
    coverage: float
    novelty: float
    effective_diversity: float


class DiverseRecommendResponse(BaseModel):
    """Результат MMR-диверсификации / MMR diversity reranking response."""

    items: list[DiverseItemResponse]
    metrics: DiversityMetricsResponse
    lambda_param: float
    n_candidates: int


@app.post("/recommend/diverse", response_model=DiverseRecommendResponse)
def recommend_diverse(request: DiverseRecommendRequest) -> DiverseRecommendResponse:
    """
    MMR-диверсификация списка кандидатов.
    MMR Diversity Reranking — balance relevance and novelty.

    Принимает отсортированный список кандидатов с оценками релевантности
    и переранжирует их, максимизируя разнообразие (Carbonell & Goldstein 1998).

    Takes a scored candidate list and reranks it to maximise the trade-off
    between relevance (lambda=1.0) and diversity (lambda=0.0).

    Полезно для борьбы с "пузырём фильтров" — пользователю предлагаются
    похожие, но не одинаковые товары.
    """
    if not request.candidate_ids:
        raise HTTPException(status_code=422, detail="candidate_ids must not be empty")

    if len(request.candidate_ids) != len(request.relevance_scores):
        raise HTTPException(
            status_code=422,
            detail="candidate_ids and relevance_scores must have equal length",
        )

    if not (0.0 <= request.lambda_param <= 1.0):
        raise HTTPException(status_code=422, detail="lambda_param must be in [0.0, 1.0]")

    diversifier = _get_diversifier(embedding_dim=request.embedding_dim)

    # Build content embeddings from optional metadata (or random if absent)
    item_embeddings = diversifier.build_item_embeddings(
        item_ids=request.candidate_ids,
        categories=request.categories,
        price_tiers=request.price_tiers,
        rng=None,
    )

    result: DiversityResult = diversifier.rerank(
        candidate_ids=request.candidate_ids,
        relevance_scores=request.relevance_scores,
        item_embeddings=item_embeddings,
        lambda_param=request.lambda_param,
        n_items=request.n_items,
    )

    return DiverseRecommendResponse(
        items=[
            DiverseItemResponse(
                item_id=item.item_id,
                relevance_score=item.relevance_score,
                diversity_contribution=item.diversity_contribution,
                mmr_score=item.mmr_score,
                rank=item.rank,
            )
            for item in result.items
        ],
        metrics=DiversityMetricsResponse(
            intra_list_diversity=result.metrics.intra_list_diversity,
            coverage=result.metrics.coverage,
            novelty=result.metrics.novelty,
            effective_diversity=result.metrics.effective_diversity,
        ),
        lambda_param=result.lambda_param,
        n_candidates=result.n_candidates,
    )
