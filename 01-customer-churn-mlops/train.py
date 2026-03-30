"""Main training script — run with `uv run python 01-customer-churn-mlops/train.py`."""

from __future__ import annotations

import pickle
from pathlib import Path

import mlflow
from churn.data.load import TARGET, prepare_dataset
from churn.models.registry import register_best_model
from churn.models.train import train_catboost, train_lightgbm
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# SQLite tracking URI нужен для Model Registry (file store его не поддерживает).
# Файл mlruns.db создаётся рядом с артефактами — удобно держать вместе.
MLFLOW_TRACKING_URI = f"sqlite:///{ARTIFACTS_DIR}/mlruns.db"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Указываем tracking URI до первого mlflow-вызова
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("Loading and preparing data...")
    df = prepare_dataset(DATA_DIR / "raw.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df[TARGET].mean():.2%}")

    # Stratified split — важно при несбалансированных классах
    indices = list(range(len(df)))
    y = df[TARGET].to_numpy()
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    df_train = df[train_idx]
    df_test = df[test_idx]
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # Train CatBoost
    print("\n--- Training CatBoost (Optuna, 30 trials) ---")
    cb_results = train_catboost(df_train, df_test, n_trials=30)
    print(f"CatBoost F1: {cb_results['f1_score']:.4f}, AUC: {cb_results['roc_auc']:.4f}")

    # Train LightGBM
    print("\n--- Training LightGBM (Optuna, 30 trials) ---")
    lgb_results = train_lightgbm(df_train, df_test, n_trials=30)
    print(f"LightGBM F1: {lgb_results['f1_score']:.4f}, AUC: {lgb_results['roc_auc']:.4f}")

    # Select best model by ROC AUC
    if cb_results["roc_auc"] >= lgb_results["roc_auc"]:
        best = cb_results
        best["model_type"] = "CatBoost"
        best_name = "churn-catboost"
        print("\n✅ Best model: CatBoost")
    else:
        best = lgb_results
        best["model_type"] = "LightGBM"
        best_name = "churn-lightgbm"
        print("\n✅ Best model: LightGBM")

    # Save pickle artifacts (для FastAPI — быстрая загрузка без MLflow сервера)
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    with open(ARTIFACTS_DIR / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "f1_score": best["f1_score"],
                "roc_auc": best["roc_auc"],
                "params": best["params"],
                "feature_importances": best["feature_importances"],
                "report": best["report"],
                "model_type": best["model_type"],
            },
            f,
        )

    # Регистрируем лучшую модель в Model Registry
    print(f"\nRegistering {best['model_type']} in MLflow Model Registry...")
    version = register_best_model(
        run_id=best["run_id"],
        artifact_path="model",
        model_name=best_name,
        metrics={"roc_auc": best["roc_auc"], "f1_score": best["f1_score"]},
    )
    if version:
        print(f"  ✅ Registered as {best_name} v{version}")
    else:
        print("  ⚠️  Registry unavailable — model saved to pickle only")

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
    print(f"  model.pkl   — trained {best['model_type']} model")
    print("  results.pkl — metrics and feature importances")
    print("\nFinal metrics:")
    print(f"  F1 Score: {best['f1_score']:.4f}")
    print(f"  ROC AUC:  {best['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
