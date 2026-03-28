"""FastAPI NER service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..model.ner import predict

app = FastAPI(
    title="NER Service API",
    description="Named Entity Recognition for Russian text",
    version="1.0.0",
)


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


@app.get("/health")
def health():
    return {"status": "healthy"}


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
