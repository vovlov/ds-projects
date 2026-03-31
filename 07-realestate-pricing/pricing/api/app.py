"""FastAPI-сервис оценки стоимости недвижимости.

Эндпоинт /estimate принимает характеристики квартиры и возвращает:
- estimated_price — точечная оценка
- confidence_interval — диапазон (модель не идеальна, честно об этом говорим)
- top_factors — топ-5 признаков, повлиявших на цену (из SHAP)
- shap_waterfall — полный SHAP waterfall: базовое значение + вклад каждого признака

Доверительный интервал считаем как +/- MAPE от обучения. Это упрощение —
в продакшене лучше использовать quantile regression или conformal prediction,
но для MVP достаточно.

SHAP waterfall — ключевое улучшение над классическими feature importances:
глобальная важность говорит "площадь важна в среднем", а SHAP говорит
"для ЭТОЙ квартиры район добавил +3.2М, а год постройки -400К".
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
    version="2.0.0",
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


class SHAPContribution(BaseModel):
    """Вклад одного признака в предсказание (одна полоска waterfall chart)."""

    feature: str = Field(..., description="Название признака")
    value: str | float | int = Field(..., description="Значение признака для данного объекта")
    contribution: float = Field(..., description="Вклад в рублях (+ повышает цену, - понижает)")
    direction: str = Field(..., description="positive | negative")


class SHAPWaterfall(BaseModel):
    """Полный SHAP waterfall: базовое значение + вклад каждого признака.

    base_value — это E[f(x)], средняя цена по обучающей выборке.
    prediction = base_value + sum(contributions).
    """

    base_value: float = Field(..., description="E[f(x)] — средняя оценка по датасету, руб")
    contributions: list[SHAPContribution] = Field(
        ..., description="Вклады признаков, отсортированы по |contribution|"
    )
    prediction: float = Field(..., description="base_value + sum(contributions), руб")


class PriceEstimate(BaseModel):
    """Результат оценки."""

    estimated_price: int = Field(..., description="Оценка стоимости, руб")
    confidence_low: int = Field(..., description="Нижняя граница, руб")
    confidence_high: int = Field(..., description="Верхняя граница, руб")
    top_factors: list[dict[str, str | float]] = Field(
        ..., description="Топ-5 факторов по |SHAP contribution|"
    )
    shap_waterfall: SHAPWaterfall | None = Field(
        None, description="SHAP waterfall для визуализации объяснения (None если SHAP недоступен)"
    )


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "healthy", "model_loaded": _model is not None}


@app.post("/estimate", response_model=PriceEstimate)
def estimate(prop: PropertyInput) -> PriceEstimate:
    """Оценить стоимость квартиры с SHAP waterfall объяснением."""
    model, results = _load_artifacts()

    from ..data.load import CURRENT_YEAR

    # Собираем признаки в том же порядке, что при обучении
    age = CURRENT_YEAR - prop.year_built
    has_garage = "yes" if prop.garage == 1 else "no"

    # Значения признаков — нужны для отображения в SHAP waterfall
    feature_values: dict[str, str | float | int] = {
        "sqft": prop.sqft,
        "bedrooms": prop.bedrooms,
        "bathrooms": prop.bathrooms,
        "year_built": prop.year_built,
        "lot_size": prop.lot_size,
        "age": age,
        "neighborhood": prop.neighborhood,
        "condition": prop.condition,
        "has_garage": has_garage,
    }

    # CatBoost ожидает массив признаков в порядке MODEL_FEATURES
    # MODEL_FEATURES = [sqft, bedrooms, bathrooms, year_built, lot_size, age,
    #                   neighborhood, condition, has_garage]
    features = np.array(list(feature_values.values()), dtype=object)

    prediction = float(model.predict(features.reshape(1, -1))[0])
    estimated = max(int(round(prediction, -4)), 1_000_000)  # не ниже 1М

    # Доверительный интервал на основе MAPE модели
    mape = results.get("mape", 0.10) if results else 0.10
    margin = max(mape, 0.05)  # минимум 5%
    confidence_low = int(round(estimated * (1 - margin), -4))
    confidence_high = int(round(estimated * (1 + margin), -4))

    # SHAP waterfall: per-prediction объяснение через CatBoost built-in SHAP.
    # Это лучше глобальных feature importances: показывает вклад для конкретной квартиры.
    shap_waterfall = None
    feature_names: list[str] = results.get("feature_names", []) if results else []

    if feature_names:
        try:
            from ..models.explain import explain_prediction

            shap_result = explain_prediction(model, features, feature_names)
            contributions = [
                SHAPContribution(
                    feature=fname,
                    value=feature_values.get(fname, ""),
                    contribution=contrib,
                    direction="positive" if contrib >= 0 else "negative",
                )
                for fname, contrib in shap_result["contributions"].items()
            ]
            shap_waterfall = SHAPWaterfall(
                base_value=shap_result["bias"],
                contributions=contributions,
                prediction=shap_result["prediction"],
            )
        except Exception as e:
            # SHAP может упасть если модель не CatBoost или Pool несовместим.
            # Не прерываем ответ — waterfall просто будет None.
            logger.warning("SHAP explain failed: %s", e)

    # top_factors: из SHAP (если доступен) или из глобальных importances (fallback)
    top_factors: list[dict[str, str | float]] = []
    if shap_waterfall:
        for c in shap_waterfall.contributions[:5]:
            top_factors.append(
                {
                    "feature": c.feature,
                    "value": c.value,
                    "contribution": c.contribution,
                    "direction": c.direction,
                }
            )
    elif results and "feature_importances" in results:
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
        shap_waterfall=shap_waterfall,
    )
