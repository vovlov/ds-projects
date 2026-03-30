"""
Обучение моделей CatBoost и LightGBM для предсказания оттока клиентов.

Задача бинарной классификации: уйдёт ли клиент (Churn=1) или останется (Churn=0).
Основная метрика — ROC AUC (классы несбалансированы: ~26% churn), дополнительно
отслеживаем F1-score, потому что бизнесу важнее поймать уходящих (recall),
чем минимизировать ложные тревоги.

MLflow логирует каждый best run: параметры, метрики, артефакт модели — так можно
вернуться и понять, почему три недели назад модель была лучше.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import optuna
import polars as pl
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

from ..data.load import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

# Инженерные признаки из add_features() — добавляем к базовым.
# TenureGroup — категориальный (new/mid/long), остальные числовые.
_CAT_ENGINEERED = ["TenureGroup"]
_NUM_ENGINEERED = ["AvgMonthlySpend", "ExpectedTotalCharges", "NumServices"]

CAT_MODEL_FEATURES = CATEGORICAL_FEATURES + _CAT_ENGINEERED
NUM_MODEL_FEATURES = NUMERICAL_FEATURES + _NUM_ENGINEERED

# Порядок: сначала числовые, потом категориальные — удобно для CatBoost cat_indices.
MODEL_FEATURES = NUM_MODEL_FEATURES + CAT_MODEL_FEATURES


def _prepare_xy(
    df: pl.DataFrame,
    encode_cats: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Подготовить X, y для обучения.

    CatBoost умеет работать с категориями нативно (encode_cats=False) —
    не нужен one-hot для 15+ телеком-услуг. LightGBM требует числа,
    поэтому используем label encoding (encode_cats=True).
    """
    features = [f for f in MODEL_FEATURES if f in df.columns]

    if encode_cats:
        cat_cols = [c for c in CAT_MODEL_FEATURES if c in features]
        df_enc = df.clone()
        for col in cat_cols:
            # Label encoding через Polars: строим mapping «значение → порядковый номер»
            mapping = df_enc.select(col).unique().sort(col).with_row_index("_code")
            df_enc = df_enc.join(mapping, on=col, how="left").drop(col).rename({"_code": col})
        X = df_enc.select(features).to_numpy().astype(np.float64)
    else:
        X = df.select(features).to_pandas()
        # Polars String → pandas object; category dtype нужен CatBoost для распознавания
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")
        features = list(X.columns)
        X = X.values

    y = df[TARGET].to_numpy().astype(np.int32)
    return X, y, features


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Вычислить метрики бинарной классификации."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


def train_catboost(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    n_trials: int = 30,
    experiment_name: str = "churn-catboost",
) -> dict[str, Any]:
    """Обучить CatBoost с подбором гиперпараметров через Optuna.

    CatBoost выбран как основная модель благодаря нативной поддержке категорий
    (15 телеком-услуг) и ordered boosting — уменьшает overfitting при
    несбалансированных классах без явного class_weight.
    """
    X_train, y_train, feature_names = _prepare_xy(df_train, encode_cats=False)
    X_test, y_test, _ = _prepare_xy(df_test, encode_cats=False)

    cat_indices = [i for i, f in enumerate(feature_names) if f in CAT_MODEL_FEATURES]

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
            "loss_function": "Logloss",
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train):
            model = CatBoostClassifier(**params, cat_features=cat_indices)
            model.fit(X_train[train_idx], y_train[train_idx])
            y_prob = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(float(roc_auc_score(y_train[val_idx], y_prob)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"verbose": 0, "random_seed": 42, "loss_function": "Logloss"})

    with mlflow.start_run(run_name="catboost-best") as run:
        model = CatBoostClassifier(**best_params, cat_features=cat_indices)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

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
        "report": report,
        "run_id": run_id,
        **metrics,
    }


def train_lightgbm(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    n_trials: int = 30,
    experiment_name: str = "churn-lightgbm",
) -> dict[str, Any]:
    """Обучить LightGBM с подбором гиперпараметров.

    LightGBM быстрее CatBoost (~10x на этом датасете), поэтому 30 trials
    Optuna проходят быстрее. Используем label encoding для категорий.
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

        model = LGBMClassifier(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"verbose": -1, "random_state": 42})

    with mlflow.start_run(run_name="lightgbm-best") as run:
        model = LGBMClassifier(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

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
        "report": report,
        "run_id": run_id,
        **metrics,
    }
