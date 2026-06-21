"""FastAPI NER service."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..active.pool import LabelingPool
from ..active.strategy import ActiveLearningConfig, SamplingStrategy, score_text
from ..data.collection5 import get_collection5_sample
from ..linking.knowledge_base import KnowledgeBase
from ..linking.linker import EntityLinker, LinkingConfig
from ..model.conformal import ConformalNERPredictor
from ..model.ner import extract_entities_from_bio, predict

app = FastAPI(
    title="NER Service API",
    description="Named Entity Recognition for Russian text",
    version="1.0.0",
)

# Автокалибровка на Collection5 при старте — гарантия coverage без ручного шага
_conformal = ConformalNERPredictor()
_cal_dataset = get_collection5_sample()
_cal_entities = []
for _sent in _cal_dataset:
    _tokens = [tok for tok, _ in _sent]
    _labels = [lbl for _, lbl in _sent]
    _cal_entities.extend(extract_entities_from_bio(_tokens, _labels))
if _cal_entities:
    _conformal.calibrate(_cal_entities)

# Active learning pool — singleton per process
_pool = LabelingPool()
_al_config = ActiveLearningConfig()

# Named Entity Linking — singleton KB + linker
_kb = KnowledgeBase()
_linker = EntityLinker(kb=_kb, config=LinkingConfig())


# ── Request/Response models ───────────────────────────────────────────────────


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Владимир Путин посетил Москву."])


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NERResponse(BaseModel):
    entities: list[EntityResponse]
    text: str


class ConformalEntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    nonconformity_score: float
    prediction_set: list[str]
    is_certain: bool
    coverage: float


class ConformalNERResponse(BaseModel):
    entities: list[ConformalEntityResponse]
    text: str
    q_hat: float
    calibrated: bool


# Active learning models
class ActiveAddRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Тексты для добавления в пул")
    strategy: str = Field(default="least_confidence", description="Стратегия выборки")


class ActiveAddResponse(BaseModel):
    ids: list[str]
    added: int
    strategy: str


class ActiveQueryRequest(BaseModel):
    batch_size: int = Field(default=10, ge=1, le=100)


class ActiveQueryItem(BaseModel):
    id: str
    text: str
    uncertainty_score: float
    n_entities: int
    strategy: str


class ActiveQueryResponse(BaseModel):
    items: list[ActiveQueryItem]
    strategy: str
    unlabeled_remaining: int


class AnnotationEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class ActiveLabelRequest(BaseModel):
    item_id: str
    annotations: list[AnnotationEntity]


class ActiveLabelResponse(BaseModel):
    id: str
    labeled: bool
    labeled_at: str | None


class ActiveStatusResponse(BaseModel):
    unlabeled_count: int
    queried_count: int
    labeled_count: int
    total_added: int
    strategy: str


class ActiveLabeledResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int


# ── Existing endpoints ────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "healthy", "conformal_calibrated": _conformal._calibrated}


@app.post("/predict", response_model=NERResponse)
def predict_entities(request: NERRequest):
    entities = predict(request.text)
    return NERResponse(
        entities=[
            EntityResponse(text=e.text, label=e.label, start=e.start, end=e.end) for e in entities
        ],
        text=request.text,
    )


@app.post("/predict/batch")
def predict_batch(requests: list[NERRequest]):
    results = []
    for req in requests:
        entities = predict(req.text)
        results.append(
            {
                "entities": [
                    {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
                    for e in entities
                ],
                "text": req.text,
            }
        )
    return results


@app.post("/predict/conformal", response_model=ConformalNERResponse)
def predict_conformal(request: NERRequest):
    """
    NER с конформными множествами предсказаний.

    Каждая сущность содержит prediction_set — набор возможных меток,
    гарантированно включающий истинную метку с вероятностью ≥ coverage.
    is_certain=True означает единственный кандидат (высокая уверенность).
    """
    results = _conformal.predict_text(request.text)
    return ConformalNERResponse(
        entities=[
            ConformalEntityResponse(
                text=r.text,
                label=r.label,
                start=r.start,
                end=r.end,
                nonconformity_score=r.nonconformity_score,
                prediction_set=r.prediction_set,
                is_certain=r.is_certain,
                coverage=r.coverage,
            )
            for r in results
        ],
        text=request.text,
        q_hat=_conformal.q_hat,
        calibrated=_conformal._calibrated,
    )


# ── Active learning endpoints ─────────────────────────────────────────────────


@app.post("/active/pool/add", response_model=ActiveAddResponse)
def active_pool_add(request: ActiveAddRequest):
    """
    Добавить тексты в пул активного обучения.

    Каждый текст прогоняется через конформный NER — nonconformity_score
    используется как прокси неопределённости без дополнительного inference.
    Тексты без найденных сущностей получают score=0 (нет сигнала для обучения).
    """
    try:
        strategy = SamplingStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Неизвестная стратегия: {request.strategy}. "
            f"Допустимые: {[s.value for s in SamplingStrategy]}",
        ) from None

    config = ActiveLearningConfig(strategy=strategy)
    uncertainty_scores: list[float] = []

    for text in request.texts:
        conformal_results = _conformal.predict_text(text)
        nc_scores = [r.nonconformity_score for r in conformal_results]
        scored = score_text(text, nc_scores, config)
        uncertainty_scores.append(scored.score)

    ids = _pool.add_texts(request.texts, uncertainty_scores, str(strategy))
    return ActiveAddResponse(ids=ids, added=len(ids), strategy=str(strategy))


@app.post("/active/pool/query", response_model=ActiveQueryResponse)
def active_pool_query(request: ActiveQueryRequest):
    """
    Запросить топ-N наиболее неопределённых текстов для аннотации.

    Возвращает items, отсортированные по убыванию uncertainty_score.
    После вызова тексты переходят в статус 'queried' — ожидают разметки.
    """
    batch = _pool.query(request.batch_size)

    items = [
        ActiveQueryItem(
            id=item.id,
            text=item.text,
            uncertainty_score=item.uncertainty_score,
            n_entities=0,
            strategy=item.strategy,
        )
        for item in batch.items
    ]

    return ActiveQueryResponse(
        items=items,
        strategy=batch.strategy,
        unlabeled_remaining=batch.unlabeled_remaining,
    )


@app.post("/active/pool/label", response_model=ActiveLabelResponse)
def active_pool_label(request: ActiveLabelRequest):
    """
    Принять разметку аннотатора для указанного item_id.

    annotations — список именованных сущностей в формате {text, label, start, end}.
    После вызова item переходит в статус 'labeled' и доступен через /active/pool/labeled.
    """
    annotations = [a.model_dump() for a in request.annotations]
    result = _pool.label(request.item_id, annotations)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"item_id {request.item_id!r} не найден в очереди аннотации. "
            "Сначала вызовите /active/pool/query.",
        )

    return ActiveLabelResponse(
        id=result.id,
        labeled=True,
        labeled_at=result.labeled_at,
    )


@app.get("/active/pool/status", response_model=ActiveStatusResponse)
def active_pool_status():
    """Статус пула: кол-во текстов в каждом состоянии."""
    status = _pool.status(str(_al_config.strategy))
    return ActiveStatusResponse(
        unlabeled_count=status.unlabeled_count,
        queried_count=status.queried_count,
        labeled_count=status.labeled_count,
        total_added=status.total_added,
        strategy=status.strategy,
    )


@app.get("/active/pool/labeled", response_model=ActiveLabeledResponse)
def active_pool_labeled():
    """Вернуть все размеченные примеры (готово для fine-tuning)."""
    labeled = _pool.get_labeled()
    items = [
        {
            "id": item.id,
            "text": item.text,
            "annotations": item.annotations,
            "labeled_at": item.labeled_at,
        }
        for item in labeled
    ]
    return ActiveLabeledResponse(items=items, total=len(items))


@app.post("/active/pool/reset")
def active_pool_reset():
    """Сбросить пул (для тестирования / новой сессии аннотации)."""
    _pool.reset()
    return {"reset": True}


# ── Named Entity Linking endpoints ───────────────────────────────────────────


class LinkRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Газпром подписал договор с Яндексом в Москве."])


class LinkedEntityResponse(BaseModel):
    mention: str
    entity_type: str
    entity_id: str | None
    canonical_name: str | None
    confidence: float
    is_linked: bool
    start: int
    end: int


class LinkResponse(BaseModel):
    text: str
    entities: list[LinkedEntityResponse]
    n_linked: int
    n_total: int


class KBStatsResponse(BaseModel):
    n_entities: int
    by_type: dict[str, int]
    n_aliases: int


class KBSearchRequest(BaseModel):
    mention: str = Field(..., min_length=1, examples=["Газпром"])
    entity_type: str | None = Field(
        default=None, description="Фильтр по типу: PER/ORG/LOC или None для всех"
    )


class KBSearchResponse(BaseModel):
    mention: str
    candidates: list[dict]
    top_match: dict | None


@app.post("/link/entities", response_model=LinkResponse)
def link_entities(request: LinkRequest):
    """
    NER + Named Entity Linking: извлечь сущности и связать с базой знаний.

    Двухэтапный пайплайн:
    1. Rule-based NER → (mention, label, start, end)
    2. EntityLinker → canonical_name из Wikidata-style KB

    Бизнес-применение: юридический отдел — "Газпром" в договоре → Q102048
    (каноническое ПАО Газпром), что исключает путаницу дочерних структур.
    """
    raw_entities = predict(request.text)
    linked: list[LinkedEntityResponse] = []

    for entity in raw_entities:
        result = _linker.link_mention(entity.text, entity.label)
        linked.append(
            LinkedEntityResponse(
                mention=entity.text,
                entity_type=entity.label,
                entity_id=result.entity_id,
                canonical_name=result.canonical_name,
                confidence=result.confidence,
                is_linked=result.is_linked,
                start=entity.start,
                end=entity.end,
            )
        )

    n_linked = sum(1 for e in linked if e.is_linked)
    return LinkResponse(text=request.text, entities=linked, n_linked=n_linked, n_total=len(linked))


@app.get("/kb/stats", response_model=KBStatsResponse)
def kb_stats():
    """Статистика базы знаний: кол-во сущностей и псевдонимов по типам."""
    stats = _kb.stats()
    return KBStatsResponse(
        n_entities=stats.n_entities,
        by_type=stats.by_type,
        n_aliases=stats.n_aliases,
    )


@app.post("/kb/search", response_model=KBSearchResponse)
def kb_search(request: KBSearchRequest):
    """
    Прямой поиск упоминания в базе знаний без NER-шага.

    Возвращает топ-N кандидатов с confidence-скорами для debugging.
    Опциональный фильтр entity_type (PER/ORG/LOC) сужает поиск.
    """
    entity_type = request.entity_type or "ORG"
    result = _linker.link_mention(request.mention, entity_type)

    top_match = None
    if result.is_linked and result.candidates:
        top_match = result.candidates[0]

    return KBSearchResponse(
        mention=request.mention,
        candidates=result.candidates,
        top_match=top_match,
    )
