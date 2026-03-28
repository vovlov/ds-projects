"""Main training script — run with `uv run python 01-customer-churn-mlops/train.py`."""

from __future__ import annotations

import pickle
from pathlib import Path

from churn.data.load import TARGET, prepare_dataset
from churn.models.train import train_catboost, train_lightgbm
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Loading and preparing data...")
    df = prepare_dataset(DATA_DIR / "raw.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df[TARGET].mean():.2%}")

    # Stratified train/test split
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

    # Select best model
    if cb_results["f1_score"] >= lgb_results["f1_score"]:
        best = cb_results
        best["model_type"] = "CatBoost"
        print("\n✅ Best model: CatBoost")
    else:
        best = lgb_results
        best["model_type"] = "LightGBM"
        print("\n✅ Best model: LightGBM")

    # Save model and results
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

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
    print(f"  model.pkl  — trained {best['model_type']} model")
    print("  results.pkl — metrics and feature importances")
    print("\nFinal metrics:")
    print(f"  F1 Score: {best['f1_score']:.4f}")
    print(f"  ROC AUC:  {best['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
