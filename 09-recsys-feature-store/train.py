"""Train recommendation models and compute evaluation metrics."""

from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl
from recsys.data.load import load_all_data
from recsys.models.collaborative import CollaborativeRecommender
from recsys.models.content_based import ContentBasedRecommender

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    users, products, interactions = load_all_data()
    print(f"Users: {users.shape[0]}, Products: {products.shape[0]}")
    print(f"Interactions: {interactions.shape[0]}")

    # Train/test split: last 20% interactions per user
    interactions_sorted = interactions.sort(["user_id", "timestamp"])

    train_parts = []
    test_parts = []
    for user_id in interactions_sorted["user_id"].unique().to_list():
        user_data = interactions_sorted.filter(pl.col("user_id") == user_id)
        n = len(user_data)
        split = int(n * 0.8)
        if split < 1:
            train_parts.append(user_data)
            continue
        train_parts.append(user_data[:split])
        test_parts.append(user_data[split:])

    train_df = pl.concat(train_parts)
    test_df = pl.concat(test_parts) if test_parts else train_df

    print(f"Train: {train_df.shape[0]}, Test: {test_df.shape[0]}")

    # Collaborative filtering
    print("\n--- Collaborative Filtering (SVD) ---")
    cf = CollaborativeRecommender(n_components=50)
    cf.fit(train_df)

    metrics = cf.evaluate(test_df, top_k=10, threshold=3.5)
    print(f"Precision@10: {metrics['precision_at_k']:.4f}")
    print(f"Recall@10:    {metrics['recall_at_k']:.4f}")
    print(f"NDCG@10:      {metrics['ndcg_at_k']:.4f}")

    # Content-based
    print("\n--- Content-Based (cold start) ---")
    cb = ContentBasedRecommender()
    cb.fit(products, train_df)

    # Sample recommendation
    sample_recs = cf.recommend(0, top_k=5)
    print(f"\nSample recs for user 0: {sample_recs}")

    # Save
    with open(ARTIFACTS_DIR / "collaborative.pkl", "wb") as f:
        pickle.dump(cf, f)
    with open(ARTIFACTS_DIR / "content_based.pkl", "wb") as f:
        pickle.dump(cb, f)
    with open(ARTIFACTS_DIR / "metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
