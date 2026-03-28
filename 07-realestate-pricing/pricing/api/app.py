"""FastAPI-сервис оценки стоимости недвижимости.

Эндпоинт /estimate принимает характеристики квартиры и возвращает:
- estimated_price — точечная оценка
- confidence_interval — диапазон (модель не идеальна, честно об этом говорим)
- top_factors — какие признаки больше всего повлияли на цену

Доверительный интервал считаем как +/- MAPE от обучения. Это упрощение —
в продакшене лучше использовать quantile regression или conformal prediction,
но для MVP достаточно.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Price Estimation API",
    description="Оценка стоимости московской недвижимости на основе CatBoost",
    version="1.0.0",
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

_model = None
_results = None


def _load_artifacts() -> tuple:
    global _model, _results
    if _model is None:
        model_path = ARTIFACTS_DIR / "model.pkl"
        results_path = ARTIFACTS_DIR / "results.pkl"
        if not model_path.exists():
            raise HTTPException(500, "Model not found. Run train.py first.")
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        if results_path.exists():
            with open(results_path, "rb") as f:
                _results = pickle.load(f)
    return _model, _results


class PropertyInput(BaseModel):
    """Характеристики квартиры для оценки."""

    sqft: int = Field(..., ge=15, le=500, description="Площадь, кв.м", examples=[65])
    bedrooms: int = Field(..., ge=1, le=10, description="Комнат", examples=[2])
    bathrooms: int = Field(..., ge=1, le=5, description="Санузлов", examples=[1])
    year_built: int = Field(..., ge=1900, le=2026, description="Год постройки", examples=[2015])
    lot_size: int = Field(..., ge=0, le=2000, description="Площадь участка, кв.м", examples=[0])
    garage: int = Field(..., ge=0, le=1, description="Гараж (0/1)", examples=[1])
    neighborhood: str = Field(..., description="Район Москвы", examples=["Хамовники"])
    condition: str = Field(..., description="Состояние", examples=["хорошее"])


class PriceEstimate(BaseModel):
    """Результат оценки."""

    estimated_price: int = Field(..., description="Оценка стоимости, руб")
    confidence_low: int = Field(..., description="Нижняя граница, руб")
    confidence_high: int = Field(..., description="Верхняя граница, руб")
    top_factors: list[dict[str, str | float]] = Field(
        ..., description="Топ-5 факторов, повлиявших на цену"
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/estimate", response_model=PriceEstimate)
def estimate(prop: PropertyInput) -> PriceEstimate:
    """Оценить стоимость квартиры."""
    model, results = _load_artifacts()

    from ..data.load import CURRENT_YEAR

    # Собираем признаки в том же порядке, что при обучении
    age = CURRENT_YEAR - prop.year_built
    has_garage = "yes" if prop.garage == 1 else "no"

    # CatBoost ожидает массив признаков в порядке MODEL_FEATURES
    # MODEL_FEATURES = [sqft, bedrooms, bathrooms, year_built, lot_size, age,
    #                   neighborhood, condition, has_garage]
    features = np.array(
        [
            prop.sqft,
            prop.bedrooms,
            prop.bathrooms,
            prop.year_built,
            prop.lot_size,
            age,
            prop.neighborhood,
            prop.condition,
            has_garage,
        ],
        dtype=object,
    )

    prediction = float(model.predict(features.reshape(1, -1))[0])
    estimated = max(int(round(prediction, -4)), 1_000_000)  # не ниже 1М

    # Доверительный интервал на основе MAPE модели
    mape = results.get("mape", 0.10) if results else 0.10
    margin = max(mape, 0.05)  # минимум 5%
    confidence_low = int(round(estimated * (1 - margin), -4))
    confidence_high = int(round(estimated * (1 + margin), -4))

    # Объяснение: top-5 признаков по важности
    top_factors = []
    if results and "feature_importances" in results:
        sorted_imp = sorted(
            results["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, importance in sorted_imp[:5]:
            top_factors.append({"feature": name, "importance": round(importance, 2)})

    return PriceEstimate(
        estimated_price=estimated,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        top_factors=top_factors,
    )
