"""Synthetic fraud detection dataset with graph structure."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def generate_synthetic_transactions(
    n_nodes: int = 500,
    n_transactions: int = 2000,
    fraud_rate: float = 0.05,
    seed: int = 42,
) -> dict:
    """Generate synthetic transaction graph data.

    Returns dict with:
        - nodes: list of dicts (id, features, is_fraud)
        - edges: list of tuples (src, dst, amount, timestamp)
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Generate node features
    nodes = []
    for i in range(n_nodes):
        is_fraud = rng.random() < fraud_rate
        if is_fraud:
            # Fraudulent nodes have different feature distribution
            avg_amount = np_rng.lognormal(mean=8, sigma=1.5)
            n_txn = np_rng.poisson(lam=15)
            account_age = np_rng.exponential(scale=30)
        else:
            avg_amount = np_rng.lognormal(mean=5, sigma=0.8)
            n_txn = np_rng.poisson(lam=5)
            account_age = np_rng.exponential(scale=365)

        nodes.append(
            {
                "id": i,
                "avg_amount": float(avg_amount),
                "n_transactions": int(n_txn),
                "account_age_days": float(account_age),
                "is_fraud": int(is_fraud),
            }
        )

    # Generate edges (transactions between nodes)
    edges = []
    for _ in range(n_transactions):
        src = rng.randint(0, n_nodes - 1)
        # Fraudulent nodes tend to transact with each other
        if nodes[src]["is_fraud"] and rng.random() < 0.4:
            fraud_nodes = [n["id"] for n in nodes if n["is_fraud"] and n["id"] != src]
            dst = rng.choice(fraud_nodes) if fraud_nodes else rng.randint(0, n_nodes - 1)
        else:
            dst = rng.randint(0, n_nodes - 1)

        if src == dst:
            dst = (dst + 1) % n_nodes

        amount = float(np_rng.lognormal(mean=5, sigma=1))
        timestamp = float(np_rng.uniform(0, 365))

        edges.append((src, dst, amount, timestamp))

    return {"nodes": nodes, "edges": edges}


def get_feature_matrix(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and labels y from graph data."""
    X = np.array(
        [[n["avg_amount"], n["n_transactions"], n["account_age_days"]] for n in data["nodes"]]
    )
    y = np.array([n["is_fraud"] for n in data["nodes"]])
    return X, y


def get_edge_index(data: dict) -> np.ndarray:
    """Get edge index array (2 x n_edges) for GNN."""
    src = [e[0] for e in data["edges"]]
    dst = [e[1] for e in data["edges"]]
    return np.array([src, dst])
