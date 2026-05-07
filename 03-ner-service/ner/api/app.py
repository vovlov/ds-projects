"""FastAPI NER service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

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
