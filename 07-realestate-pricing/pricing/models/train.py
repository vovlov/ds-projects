"""
Обучение моделей CatBoost и LightGBM для оценки стоимости недвижимости.

Подход к оптимизации: Optuna с 5-fold CV, 20 trials. Для задачи регрессии
(в отличие от классификации в проекте 01) используем RMSE как основную метрику,
но также считаем MAE, MAPE и R^2 — каждая метрика рассказывает свою историю:
- RMSE штрафует за большие ошибки (важно — не хотим оценить 30М квартиру в 10М)
- MAE — медианная ошибка, устойчива к выбросам
- MAPE — относительная ошибка, понятна бизнесу ("ошибаемся на 8%")
- R^2 — доля объяснённой дисперсии

MLflow логирует каждый trial, чтобы можно было вернуться и понять,
какие гиперпараметры работают лучше.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import optuna
import polars as pl
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score

from ..data.load import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

# Признаки для модели: числовые + категориальные.
# price_per_sqft НЕ включаем — он вычисляется из таргета (data leakage).
MODEL_FEATURES = [f for f in NUMERICAL_FEATURES if f != "price_per_sqft"] + CATEGORICAL_FEATURES


def _prepare_xy(
    df: pl.DataFrame,
    features: list[str] | None = None,
    encode_cats: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Подготовить X, y для обучения.

    CatBoost умеет работать с категориями нативно, поэтому encode_cats=False.
    Для LightGBM нужно закодировать — используем label encoding (порядковый),
    потому что one-hot на 15 районов создаст слишком разреженную матрицу.
    """
    if features is None:
        features = MODEL_FEATURES
    features = [f for f in features if f in df.columns]

    if encode_cats:
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in features]
        df_enc = df.clone()
        for col in cat_cols:
            # Label encoding: Polars делает это элегантно через rank
            mapping = (
                df_enc.select(col)
                .unique()
                .sort(col)
                .with_row_index("_code")
            )
            df_enc = df_enc.join(mapping, on=col, how="left").drop(col).rename({"_code": col})
        X = df_enc.select(features).to_numpy().astype(np.float64)
    else:
        X = df.select(features).to_pandas()
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")
        feature_names = list(X.columns)
        X = X.values
        features = feature_names

    y = df[TARGET].to_numpy().astype(np.float64)
    return X, y, features


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Вычислить все метрики регрессии."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_catboost(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    n_trials: int = 20,
    experiment_name: str = "realestate-catboost",
) -> dict[str, Any]:
    """Обучить CatBoost с подбором гиперпараметров через Optuna.

    CatBoost выбран как основная модель, потому что:
    1. Нативно работает с категориями (не нужен one-hot для 15 районов)
    2. Ordered boosting уменьшает overfitting на маленьких данных (1000 строк)
    3. Встроенная feature importance — нужна для объяснимости
    """
    X_train, y_train, feature_names = _prepare_xy(df_train, encode_cats=False)
    X_test, y_test, _ = _prepare_xy(df_test, encode_cats=False)

    cat_indices = [
        i for i, f in enumerate(feature_names)
        if f in CATEGORICAL_FEATURES
    ]

    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": 0,
            "random_seed": 42,
            "loss_function": "RMSE",
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train):
            model = CatBoostRegressor(**params, cat_features=cat_indices)
            model.fit(X_train[train_idx], y_train[train_idx])
            y_pred = model.predict(X_train[val_idx])
            rmse = float(np.sqrt(mean_squared_error(y_train[val_idx], y_pred)))
            scores.append(rmse)
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["verbose"] = 0
    best_params["random_seed"] = 42
    best_params["loss_function"] = "RMSE"

    with mlflow.start_run(run_name="catboost-best") as run:
        model = CatBoostRegressor(**best_params, cat_features=cat_indices)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)

        mlflow.log_params(best_params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        importances = dict(zip(feature_names, model.feature_importances_))
        mlflow.catboost.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    return {
        "model": model,
        "params": best_params,
        "feature_names": feature_names,
        "feature_importances": importances,
        "run_id": run_id,
        **metrics,
    }


def train_lightgbm(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    n_trials: int = 20,
    experiment_name: str = "realestate-lightgbm",
) -> dict[str, Any]:
    """Обучить LightGBM с подбором гиперпараметров.

    LightGBM быстрее CatBoost (~10x на этом датасете), поэтому подходит
    как baseline для сравнения. Используем label encoding для категорий.
    """
    X_train, y_train, feature_names = _prepare_xy(df_train, encode_cats=True)
    X_test, y_test, _ = _prepare_xy(df_test, encode_cats=True)

    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "verbose": -1,
            "random_state": 42,
        }

        model = LGBMRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        # cross_val_score для регрессии: neg_root_mean_squared_error
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["verbose"] = -1
    best_params["random_state"] = 42

    with mlflow.start_run(run_name="lightgbm-best") as run:
        model = LGBMRegressor(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = _compute_metrics(y_test, y_pred)

        mlflow.log_params(best_params)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        importances = dict(zip(feature_names, model.feature_importances_))
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    return {
        "model": model,
        "params": best_params,
        "feature_names": feature_names,
        "feature_importances": importances,
        "run_id": run_id,
        **metrics,
    }
