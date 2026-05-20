"""FastAPI NER service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..active_learning.sampler import ActiveLearner, SamplingStrategy
from ..data.collection5 import get_collection5_sample
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


@app.get("/health")
def health():
    stats = _active_learner.get_stats()
    return {
        "status": "healthy",
        "conformal_calibrated": _conformal._calibrated,
        "active_learning": {
            "pending": stats.pending,
            "annotated": stats.annotated,
            "recalibrations": stats.recalibrations,
        },
    }


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


# ── Active Learning singleton ────────────────────────────────────────────────

_active_learner = ActiveLearner(
    conformal_predictor=_conformal, strategy=SamplingStrategy.UNCERTAINTY
)


def _reset_active_learner() -> None:
    """Сброс для тестовой изоляции."""
    global _active_learner
    _active_learner = ActiveLearner(
        conformal_predictor=_conformal, strategy=SamplingStrategy.UNCERTAINTY
    )


# ── Active Learning request/response models ───────────────────────────────────


class SampleRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Тексты для оценки неопределённости")
    n: int = Field(default=5, ge=1, le=50, description="Сколько кандидатов вернуть")
    strategy: str = Field(default="uncertainty", description="uncertainty|margin|entropy|random")


class AnnotationEntity(BaseModel):
    text: str
    label: str
    start: int = 0
    end: int = 0


class AnnotateRequest(BaseModel):
    candidate_id: str
    annotation: list[AnnotationEntity]


class CandidateResponse(BaseModel):
    candidate_id: str
    text: str
    uncertainty_score: float
    sampling_reason: str
    predicted_entities: list[dict]
    strategy: str
    annotated: bool


class SampleResponse(BaseModel):
    candidates: list[CandidateResponse]
    strategy: str
    total_selected: int


class ActiveStatsResponse(BaseModel):
    total_candidates: int
    annotated: int
    pending: int
    avg_uncertainty: float
    recalibrations: int
    strategy_used: str


@app.post("/active/sample", response_model=SampleResponse, status_code=200)
def active_sample(request: SampleRequest):
    """
    Выбрать тексты для аннотации по мере неопределённости модели.

    Возвращает top-n кандидатов, ранжированных по убыванию uncertainty_score.
    Тексты без сущностей (cold-start) всегда получают score=1.0.
    """
    strategy_map = {
        "uncertainty": SamplingStrategy.UNCERTAINTY,
        "margin": SamplingStrategy.MARGIN,
        "entropy": SamplingStrategy.ENTROPY,
        "random": SamplingStrategy.RANDOM,
    }
    strategy = strategy_map.get(request.strategy, SamplingStrategy.UNCERTAINTY)
    _active_learner._strategy = strategy

    candidates = _active_learner.select_candidates(request.texts, n=request.n)
    return SampleResponse(
        candidates=[CandidateResponse(**c.to_dict()) for c in candidates],
        strategy=request.strategy,
        total_selected=len(candidates),
    )


@app.post("/active/annotate", status_code=200)
def active_annotate(request: AnnotateRequest):
    """
    Принять аннотацию эксперта и обновить калибровку.

    После каждой аннотации конформный предсказатель рекалибруется на
    накопленном наборе аннотаций — q_hat обновляется без переобучения.
    """
    annotation_dicts = [
        {"text": e.text, "label": e.label, "start": e.start, "end": e.end}
        for e in request.annotation
    ]
    success = _active_learner.receive_annotation(request.candidate_id, annotation_dicts)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {request.candidate_id} not found in queue",
        )

    stats = _active_learner.get_stats()
    return {
        "accepted": True,
        "candidate_id": request.candidate_id,
        "recalibrations": stats.recalibrations,
        "q_hat": _conformal.q_hat,
        "pending_in_queue": stats.pending,
    }


@app.get("/active/queue")
def active_queue():
    """Вернуть очередь текстов, ожидающих аннотации (по убыванию uncertainty)."""
    queue = _active_learner.get_queue()
    return {
        "pending": len(queue),
        "candidates": [c.to_dict() for c in queue],
    }


@app.get("/active/stats", response_model=ActiveStatsResponse)
def active_stats():
    """Статистика сессии активного обучения."""
    return ActiveStatsResponse(**_active_learner.get_stats().to_dict())


@app.post("/active/reset")
def active_reset():
    """Сбросить состояние Active Learner (для тестов / новой сессии аннотации)."""
    _reset_active_learner()
    return {"reset": True}


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
