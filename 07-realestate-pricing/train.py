"""Main training script — run with `uv run python 07-realestate-pricing/train.py`."""

from __future__ import annotations

import pickle
from pathlib import Path

from pricing.data.load import TARGET, load_dataset
from pricing.models.train import train_catboost, train_lightgbm
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Generating dataset...")
    df = load_dataset(n_rows=1000, seed=42)
    print(f"Dataset shape: {df.shape}")
    print(f"Price range: {df[TARGET].min():,} - {df[TARGET].max():,} RUB")
    print(f"Mean price: {df[TARGET].mean():,.0f} RUB")

    # Train/test split (80/20)
    indices = list(range(len(df)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    df_train = df[train_idx]
    df_test = df[test_idx]
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # Train CatBoost
    print("\n--- Training CatBoost (Optuna, 20 trials) ---")
    cb_results = train_catboost(df_train, df_test, n_trials=20)
    print(f"CatBoost RMSE: {cb_results['rmse']:,.0f}, R2: {cb_results['r2']:.4f}")

    # Train LightGBM
    print("\n--- Training LightGBM (Optuna, 20 trials) ---")
    lgb_results = train_lightgbm(df_train, df_test, n_trials=20)
    print(f"LightGBM RMSE: {lgb_results['rmse']:,.0f}, R2: {lgb_results['r2']:.4f}")

    # Select best model by RMSE (lower is better)
    if cb_results["rmse"] <= lgb_results["rmse"]:
        best = cb_results
        best["model_type"] = "CatBoost"
        print("\nBest model: CatBoost")
    else:
        best = lgb_results
        best["model_type"] = "LightGBM"
        print("\nBest model: LightGBM")

    # Save artifacts
    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(best["model"], f)

    with open(ARTIFACTS_DIR / "results.pkl", "wb") as f:
        pickle.dump(
            {
                "rmse": best["rmse"],
                "mae": best["mae"],
                "mape": best["mape"],
                "r2": best["r2"],
                "params": best["params"],
                "feature_importances": best["feature_importances"],
                "feature_names": best["feature_names"],
                "model_type": best["model_type"],
            },
            f,
        )

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
    print(f"  model.pkl   — trained {best['model_type']} model")
    print("  results.pkl — metrics and feature importances")
    print("\nFinal metrics:")
    print(f"  RMSE: {best['rmse']:,.0f} RUB")
    print(f"  MAE:  {best['mae']:,.0f} RUB")
    print(f"  MAPE: {best['mape']:.2%}")
    print(f"  R2:   {best['r2']:.4f}")


if __name__ == "__main__":
    main()
