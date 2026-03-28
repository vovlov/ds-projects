"""
Объяснимость модели — без SHAP, на нативных средствах CatBoost.

Почему не SHAP: библиотека shap тянёт llvmlite, который ломается на Apple Silicon
и конфликтует с другими зависимостями. CatBoost предоставляет свой метод
get_feature_importance с разными типами — PredictionValuesChange (аналог gain)
и ShapValues (да, CatBoost реализует SHAP внутри без внешней зависимости).

Для бизнеса важно не просто сказать "цена 12М", а объяснить ПОЧЕМУ:
"район +3М, площадь +2М, состояние -500К". Это делает explain_prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    importance_type: str = "FeatureImportance",
) -> dict[str, float]:
    """Получить важность признаков из обученной модели.

    Работает с CatBoost (feature_importances_) и LightGBM (feature_importances_).
    Возвращает отсортированный словарь {название: значение}.
    """
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    else:
        raise ValueError(f"Model {type(model).__name__} has no feature_importances_")

    importances = dict(zip(feature_names, raw))
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


def explain_prediction(
    model: Any,
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Объяснить конкретное предсказание — вклад каждого признака.

    Для CatBoost: используем встроенный calc_feature_contribution (SHAP-подобно).
    Для LightGBM: используем predict с pred_contrib=True.
    Для остальных: fallback на feature_importances_ * (значение - среднее).
    """
    features_2d = features.reshape(1, -1) if features.ndim == 1 else features

    model_name = type(model).__name__

    if model_name == "CatBoostRegressor":
        # CatBoost: встроенный SHAP без внешней библиотеки
        # get_feature_importance(type="ShapValues") возвращает [n_samples, n_features + 1]
        # последний столбец — bias (базовое значение)
        shap_values = model.get_feature_importance(
            data=model.get_params().get("pool", None) or _make_pool(model, features_2d),
            type="ShapValues",
        )
        contributions = shap_values[0, :-1]  # без bias
        bias = shap_values[0, -1]
    elif model_name == "LGBMRegressor":
        # LightGBM: pred_contrib
        contrib = model.predict(features_2d, pred_contrib=True)
        contributions = contrib[0, :-1]
        bias = contrib[0, -1]
    else:
        # Fallback: importance * direction (грубая оценка)
        prediction = float(model.predict(features_2d)[0])
        imp = model.feature_importances_
        contributions = imp / imp.sum() * prediction
        bias = 0.0

    result: dict[str, float] = {}
    for name, value in zip(feature_names, contributions):
        result[name] = round(float(value), 0)

    # Сортируем по абсолютному вкладу — бизнесу важнее знать,
    # что БОЛЬШЕ ВСЕГО влияет, неважно в какую сторону
    sorted_result = dict(
        sorted(result.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return {
        "contributions": sorted_result,
        "bias": round(float(bias), 0),
        "prediction": round(float(bias + sum(contributions)), 0),
    }


def _make_pool(model: Any, features: np.ndarray) -> Any:
    """Создать CatBoost Pool для вычисления SHAP."""
    from catboost import Pool

    cat_features = model.get_param("cat_features") or []
    return Pool(features, cat_features=cat_features)
