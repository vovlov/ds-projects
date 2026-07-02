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
from recsys.models.debiasing import (
    DebiasingConfig,
    PopularityDebiaser,
)
from recsys.models.diversity import DiversityConfig, DiversityResult, MMRDiversifier
from recsys.models.session import SessionConfig, SessionRecommender, SessionResult
from recsys.models.thompson import ThompsonBandit, ThompsonConfig, ThompsonResult

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

# Session-based recommender singleton
_session_recommender: SessionRecommender | None = None

# Thompson Sampling bandit singleton — persists Beta posteriors across requests
_thompson_bandit: ThompsonBandit | None = None


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


def _get_session_recommender(
    embedding_dim: int = 32,
    decay_factor: float = 0.8,
    max_session_length: int = 20,
    n_items: int = 500,
) -> SessionRecommender:
    """Ленивая инициализация SessionRecommender / Lazy session recommender init."""
    global _session_recommender
    if _session_recommender is None:
        _session_recommender = SessionRecommender(
            SessionConfig(
                embedding_dim=embedding_dim,
                decay_factor=decay_factor,
                max_session_length=max_session_length,
                n_items=n_items,
            )
        )
    return _session_recommender


def _reset_session_recommender() -> None:
    """Сброс session recommender для тестовой изоляции / Reset for test isolation."""
    global _session_recommender
    _session_recommender = None


def _get_thompson_bandit(
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    seed: int | None = None,
) -> ThompsonBandit:
    """Ленивая инициализация Thompson bandit / Lazy Thompson bandit initialization."""
    global _thompson_bandit
    if _thompson_bandit is None:
        _thompson_bandit = ThompsonBandit(
            ThompsonConfig(alpha_prior=alpha_prior, beta_prior=beta_prior, seed=seed)
        )
    return _thompson_bandit


def _reset_thompson_bandit() -> None:
    """Сброс Thompson bandit для тестовой изоляции / Reset for test isolation."""
    global _thompson_bandit
    _thompson_bandit = None


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


# ---------------------------------------------------------------------------
# Session-Based Recommendations endpoints
# ---------------------------------------------------------------------------


class SessionInteractRequest(BaseModel):
    """
    Запрос на запись взаимодействия в сессию.
    Request to record a user-item interaction into the session.

    Пример / Example:
        { "user_id": 42, "item_id": 101 }
    """

    user_id: int
    item_id: int


class SessionInteractResponse(BaseModel):
    """Подтверждение записи взаимодействия. / Interaction recording confirmation."""

    user_id: int
    item_id: int
    session_length: int
    message: str


class SessionRecommendRequest(BaseModel):
    """
    Запрос на session-based рекомендации.
    Request for session-based next-item recommendations.

    candidate_ids: список кандидатов для ранжирования.
        None → все item_ids [0, n_items).
    exclude_seen: исключить товары уже просмотренные в сессии.
    decay_factor: вес новых взаимодействий. 0.8 по умолчанию.
    """

    user_id: int
    candidate_ids: list[int] | None = None
    top_k: int = 10
    exclude_seen: bool = True
    decay_factor: float = 0.8
    embedding_dim: int = 32


class SessionRecommendationResponse(BaseModel):
    """Одна рекомендация из session-based модели."""

    item_id: int
    score: float
    rank: int


class SessionRecommendResponse(BaseModel):
    """Ответ session-based рекомендательной системы."""

    user_id: int
    session_length: int
    recommendations: list[SessionRecommendationResponse]
    method: str
    session_vector_norm: float


class SessionStatusResponse(BaseModel):
    """Текущее состояние сессии пользователя."""

    user_id: int
    session_length: int
    item_history: list[int]
    last_updated: str


class SessionStatsResponse(BaseModel):
    """Агрегированная статистика всех сессий для мониторинга."""

    n_sessions: int
    n_known_items: int
    avg_session_length: float
    embedding_dim: int
    decay_factor: float


@app.post("/session/interact", response_model=SessionInteractResponse)
def session_interact(request: SessionInteractRequest) -> SessionInteractResponse:
    """
    Записать взаимодействие пользователя с товаром в сессию.
    Record a user-item interaction into the session window.

    Сессия создаётся автоматически при первом взаимодействии.
    Старые взаимодействия вытесняются при достижении max_session_length.

    Session is auto-created on first interaction.
    Old interactions are evicted when max_session_length is reached (sliding window).
    """
    rec = _get_session_recommender()
    state = rec.record_interaction(request.user_id, request.item_id)
    return SessionInteractResponse(
        user_id=request.user_id,
        item_id=request.item_id,
        session_length=len(state.item_history),
        message=f"interaction recorded (session_length={len(state.item_history)})",
    )


@app.post("/session/recommend", response_model=SessionRecommendResponse)
def session_recommend(request: SessionRecommendRequest) -> SessionRecommendResponse:
    """
    Рекомендовать следующие товары на основе истории сессии.
    Recommend next items using session-based decay-weighted embedding similarity.

    Алгоритм (GRU4Rec-инспированный, Hidasi et al. 2016):
        session_vec = Σ decay^(T-t) · emb[item_t] / Σ decay^(T-t)
        score(item) = cosine(session_vec, emb[item])

    При пустой сессии (cold start) — ранжирование по популярности.
    Empty session (cold start) → popular item fallback.
    """
    if request.top_k < 1:
        raise HTTPException(status_code=422, detail="top_k must be >= 1")

    rec = _get_session_recommender(
        embedding_dim=request.embedding_dim,
        decay_factor=request.decay_factor,
    )
    result: SessionResult = rec.recommend(
        user_id=request.user_id,
        candidate_ids=request.candidate_ids,
        top_k=request.top_k,
        exclude_seen=request.exclude_seen,
    )

    return SessionRecommendResponse(
        user_id=result.user_id,
        session_length=result.session_length,
        recommendations=[
            SessionRecommendationResponse(item_id=r.item_id, score=r.score, rank=r.rank)
            for r in result.recommendations
        ],
        method=result.method,
        session_vector_norm=result.session_vector_norm,
    )


@app.get("/session/status/{user_id}", response_model=SessionStatusResponse)
def session_status(user_id: int) -> SessionStatusResponse:
    """
    Текущее состояние сессии пользователя.
    Current session state for the given user.

    Возвращает 404 если сессия не существует (пользователь ещё не взаимодействовал).
    Returns 404 if the user has no session yet.
    """
    rec = _get_session_recommender()
    state = rec.get_session(user_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No session found for user_id={user_id}")

    return SessionStatusResponse(
        user_id=state.user_id,
        session_length=len(state.item_history),
        item_history=list(state.item_history),
        last_updated=state.last_updated,
    )


@app.get("/session/stats", response_model=SessionStatsResponse)
def session_stats() -> SessionStatsResponse:
    """
    Агрегированная статистика сессий для мониторинга.
    Aggregate session statistics for monitoring dashboards.
    """
    rec = _get_session_recommender()
    stats = rec.get_stats()
    return SessionStatsResponse(
        n_sessions=stats["n_sessions"],
        n_known_items=stats["n_known_items"],
        avg_session_length=stats["avg_session_length"],
        embedding_dim=stats["embedding_dim"],
        decay_factor=stats["decay_factor"],
    )


# ---------------------------------------------------------------------------
# Thompson Sampling (Beta-Bernoulli) Bandit endpoints
# ---------------------------------------------------------------------------


class ThompsonRecommendItem(BaseModel):
    """Одна рекомендация Thompson Sampling с компонентами posterior.
    Single Thompson Sampling recommendation with posterior decomposition."""

    arm_id: int
    rank: int
    sampled_theta: float
    expected_reward: float
    uncertainty: float
    n_pulls: int


class ThompsonRecommendRequest(BaseModel):
    """
    Запрос на ранжирование кандидатов через Thompson Sampling.
    Request to rank candidate items via Beta-Bernoulli Thompson Sampling.

    Пример / Example:
        {"candidate_ids": [101, 202, 303], "top_k": 3}

    Не требует контекстного вектора — exploration встроен в Beta-сэмплинг.
    No context vector needed — exploration is built into Beta sampling.
    """

    candidate_ids: list[int]
    top_k: int = 10
    alpha_prior: float = 1.0
    beta_prior: float = 1.0


class ThompsonRecommendResponse(BaseModel):
    """Результат Thompson Sampling ранжирования. / Thompson Sampling ranking result."""

    recommendations: list[ThompsonRecommendItem]
    n_arms_scored: int
    top_arm_id: int
    n_total_arms: int


class ThompsonFeedbackRequest(BaseModel):
    """
    Обратная связь для обновления Beta-posterior.
    Feedback to update Beta posterior: click → α += 1, skip → β += 1.

    reward: 1.0 = клик/покупка (успех), 0.0 = пропуск (неудача).
    Допустимы дробные значения — интерпретируются как reward ≥ 0.5 → success.
    """

    arm_id: int
    reward: float  # 1.0 (click) или 0.0 (skip)


class ThompsonFeedbackResponse(BaseModel):
    """Статус обновления posterior. / Posterior update status."""

    arm_id: int
    alpha: float
    beta: float
    n_pulls: int
    posterior_mean: float
    message: str


class ThompsonStatsResponse(BaseModel):
    """Статистика Thompson bandit для мониторинга. / Thompson bandit stats for monitoring."""

    n_arms: int
    total_recommendations: int
    config_alpha_prior: float
    config_beta_prior: float
    arm_stats: list[dict]


@app.post("/thompson/recommend", response_model=ThompsonRecommendResponse)
def thompson_recommend(request: ThompsonRecommendRequest) -> ThompsonRecommendResponse:
    """
    Ранжировать кандидатов через Beta-Bernoulli Thompson Sampling.
    Rank candidate items using Thompson Sampling exploration-exploitation.

    Каждый arm (item) представлен Beta-posterior: новые items → Beta(1,1) uniform prior
    → максимальная начальная неопределённость. После feedback posterior сужается.

    Cold start: uniform Beta(1,1) → arms ранжируются равномерно (exploration).
    После feedback: posterior_mean сближается с реальным CTR (exploitation).

    В отличие от LinUCB не требует контекстного вектора — используйте для
    сценариев с чистым click/no-click без user-item features.
    Unlike LinUCB, no context vector needed — use for pure click/no-click scenarios.
    """
    if not request.candidate_ids:
        raise HTTPException(status_code=422, detail="candidate_ids must not be empty")

    bandit = _get_thompson_bandit(
        alpha_prior=request.alpha_prior,
        beta_prior=request.beta_prior,
    )
    result: ThompsonResult = bandit.recommend(
        candidate_ids=request.candidate_ids,
        top_k=request.top_k,
    )

    recs = [
        ThompsonRecommendItem(
            arm_id=r.arm_id,
            rank=r.rank,
            sampled_theta=r.sampled_theta,
            expected_reward=r.expected_reward,
            uncertainty=r.uncertainty,
            n_pulls=r.n_pulls,
        )
        for r in result.recommendations
    ]

    return ThompsonRecommendResponse(
        recommendations=recs,
        n_arms_scored=result.n_arms_scored,
        top_arm_id=result.top_arm_id,
        n_total_arms=result.n_total_arms,
    )


@app.post("/thompson/feedback", response_model=ThompsonFeedbackResponse)
def thompson_feedback(request: ThompsonFeedbackRequest) -> ThompsonFeedbackResponse:
    """
    Обновить Beta-posterior arm после получения обратной связи.
    Update Beta posterior after receiving user feedback.

    Conjugate update: Beta + Bernoulli → Beta (аналитически, без gradient descent).
    Conjugate update: Beta + Bernoulli → Beta (analytical, no gradient descent needed).

    reward=1.0 → α += 1 (успех/клик)
    reward=0.0 → β += 1 (неудача/пропуск)
    """
    if not (0.0 <= request.reward <= 1.0):
        raise HTTPException(status_code=422, detail="reward must be in [0.0, 1.0]")

    bandit = _get_thompson_bandit()
    arm = bandit.update(arm_id=request.arm_id, reward=request.reward)

    return ThompsonFeedbackResponse(
        arm_id=request.arm_id,
        alpha=round(arm.alpha, 4),
        beta=round(arm.beta, 4),
        n_pulls=arm.n_pulls,
        posterior_mean=round(arm.posterior_mean, 4),
        message=f"arm {request.arm_id} updated (α={arm.alpha:.1f}, β={arm.beta:.1f})",
    )


@app.get("/thompson/stats", response_model=ThompsonStatsResponse)
def thompson_stats() -> ThompsonStatsResponse:
    """
    Статистика Thompson Sampling bandit для мониторинга.
    Thompson Sampling bandit statistics for monitoring dashboards.

    posterior_mean по каждому arm — оценка CTR; posterior_std — неопределённость.
    Arm с высоким std → ещё активно исследуется; с низким std → exploitation.
    Arms with high std → still being explored; low std → being exploited.
    """
    bandit = _get_thompson_bandit()
    return ThompsonStatsResponse(
        n_arms=bandit.n_arms,
        total_recommendations=bandit.total_recommendations,
        config_alpha_prior=bandit.config.alpha_prior,
        config_beta_prior=bandit.config.beta_prior,
        arm_stats=bandit.get_arm_stats(),
    )


# ==================== IPS Popularity Debiasing ====================

_debiaser: PopularityDebiaser | None = None


def _get_debiaser(alpha: float = 0.5, clip_max: float = 10.0) -> PopularityDebiaser:
    global _debiaser
    if _debiaser is None:
        _debiaser = PopularityDebiaser(DebiasingConfig(alpha=alpha, clip_max=clip_max))
    return _debiaser


def _reset_debiaser() -> None:
    global _debiaser
    _debiaser = None


class DebiasFitRequest(BaseModel):
    """Запрос на обучение IPS debiaser / IPS debiaser fit request."""

    interactions: list[dict]  # [{"user_id": int, "product_id": int, ...}]
    alpha: float = 0.5
    clip_max: float = 10.0


class DebiasFitResponse(BaseModel):
    """Результат обучения IPS debiaser / Fit result."""

    n_items: int
    mean_propensity: float
    gini_coefficient: float
    top10_concentration: float
    message: str


class DebiasScoresRequest(BaseModel):
    """Запрос на IPS-коррекцию скоров рекомендаций.
    Request to debias recommendation scores."""

    recommendations: list[dict]  # [{"product_id": int, "score": float}]
    scale: float = 0.3


class DebiasScoresResponse(BaseModel):
    """Debiased рекомендации (переранжированные) / Debiased recommendations."""

    recommendations: list[dict]  # [{"product_id": int, "score": float}]
    n_reranked: int


class PropensityStatsResponse(BaseModel):
    """Статистика propensity для мониторинга / Propensity stats for monitoring."""

    n_items: int
    mean_propensity: float
    std_propensity: float
    min_propensity: float
    max_propensity: float
    mean_ips_weight: float
    gini_coefficient: float
    top10_concentration: float
    is_fitted: bool


@app.post("/debias/fit", response_model=DebiasFitResponse)
def debias_fit(request: DebiasFitRequest) -> DebiasFitResponse:
    """
    Обучает IPS debiaser на данных взаимодействий.
    Fits IPS popularity debiaser on interaction data.

    Оценивает propensity каждого товара: p(i) ∝ count(i)^alpha.
    alpha=0: равномерное (нет debiasing), alpha=1: пропорционально популярности.

    Estimates item propensities: popular items get higher p(i) → lower IPS weight.
    """
    _reset_debiaser()
    debiaser = _get_debiaser(alpha=request.alpha, clip_max=request.clip_max)

    import polars as pl

    df = pl.DataFrame(request.interactions)
    if "product_id" not in df.columns:
        raise HTTPException(status_code=422, detail="interactions must contain 'product_id' field")

    debiaser.fit(df)
    stats = debiaser.compute_propensity_stats()

    return DebiasFitResponse(
        n_items=stats.n_items,
        mean_propensity=stats.mean_propensity,
        gini_coefficient=stats.gini_coefficient,
        top10_concentration=stats.top10_concentration,
        message=f"Fitted on {stats.n_items} items; Gini={stats.gini_coefficient:.3f}",
    )


@app.post("/debias/rerank", response_model=DebiasScoresResponse)
def debias_rerank(request: DebiasScoresRequest) -> DebiasScoresResponse:
    """
    Переранжирует рекомендации с IPS-коррекцией скоров.
    Re-ranks recommendations with IPS score correction.

    Нишевые товары получают boost пропорционально 1/propensity^scale.
    Popular items are down-weighted; niche items receive a score boost.
    """
    debiaser = _get_debiaser()
    if not debiaser._is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Debiaser not fitted. Call POST /debias/fit first.",
        )

    recs = [(int(r["product_id"]), float(r["score"])) for r in request.recommendations]
    debiased = debiaser.debias_scores(recs, scale=request.scale)

    return DebiasScoresResponse(
        recommendations=[{"product_id": pid, "score": score} for pid, score in debiased],
        n_reranked=len(debiased),
    )


@app.get("/debias/stats", response_model=PropensityStatsResponse)
def debias_stats() -> PropensityStatsResponse:
    """
    Статистика распределения propensity для мониторинга качества данных.
    Propensity distribution stats for data quality monitoring.

    Высокий Gini (> 0.7) → сильное смещение популярности → debiasing критичен.
    High Gini (> 0.7) → strong popularity skew → debiasing is critical.
    """
    debiaser = _get_debiaser()
    if not debiaser._is_fitted:
        return PropensityStatsResponse(
            n_items=0,
            mean_propensity=0.0,
            std_propensity=0.0,
            min_propensity=0.0,
            max_propensity=0.0,
            mean_ips_weight=0.0,
            gini_coefficient=0.0,
            top10_concentration=0.0,
            is_fitted=False,
        )

    stats = debiaser.compute_propensity_stats()
    return PropensityStatsResponse(
        n_items=stats.n_items,
        mean_propensity=stats.mean_propensity,
        std_propensity=stats.std_propensity,
        min_propensity=stats.min_propensity,
        max_propensity=stats.max_propensity,
        mean_ips_weight=stats.mean_ips_weight,
        gini_coefficient=stats.gini_coefficient,
        top10_concentration=stats.top10_concentration,
        is_fitted=True,
    )
