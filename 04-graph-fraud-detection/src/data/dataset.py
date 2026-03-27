"""
Синтетический генератор графа транзакций для задачи fraud detection.

Почему синтетика, а не реальные данные?
1. Реальные данные о мошенничестве — конфиденциальны.
2. Синтетика позволяет контролировать fraud rate и структуру графа.
3. Модель, работающая на синтетике, переносится на реальные данные
   (Elliptic Bitcoin, IEEE-CIS Fraud Detection).

Ключевая идея: мошенники не действуют изолированно. Они образуют кластеры —
переводят деньги друг другу, используют общие подставные счета. Граф это ловит,
а табличная модель — нет.
"""

from __future__ import annotations

import random

import numpy as np


def generate_synthetic_transactions(
    n_nodes: int = 500,
    n_transactions: int = 2000,
    fraud_rate: float = 0.05,
    seed: int = 42,
) -> dict:
    """Сгенерировать граф транзакций с мошенническими узлами.

    Мошеннические узлы отличаются от нормальных:
    - Средняя сумма транзакции выше (lognormal mean=8 vs 5)
    - Больше транзакций (poisson λ=15 vs 5)
    - Аккаунт моложе (exponential scale=30 vs 365 дней)
    - 40% их транзакций идут к другим мошенникам (кластеризация)
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    nodes = []
    for i in range(n_nodes):
        is_fraud = rng.random() < fraud_rate

        if is_fraud:
            # Мошенник: крупные суммы, частые транзакции, свежий аккаунт
            avg_amount = np_rng.lognormal(mean=8, sigma=1.5)
            n_txn = np_rng.poisson(lam=15)
            account_age = np_rng.exponential(scale=30)
        else:
            # Обычный клиент: умеренные суммы, редкие транзакции, старый аккаунт
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

    # Рёбра графа — транзакции между узлами.
    # Мошенники с вероятностью 40% переводят деньги другим мошенникам.
    # Это создаёт кластерную структуру, которую GNN может выучить.
    edges = []
    for _ in range(n_transactions):
        src = rng.randint(0, n_nodes - 1)

        if nodes[src]["is_fraud"] and rng.random() < 0.4:
            fraud_nodes = [n["id"] for n in nodes if n["is_fraud"] and n["id"] != src]
            dst = rng.choice(fraud_nodes) if fraud_nodes else rng.randint(0, n_nodes - 1)
        else:
            dst = rng.randint(0, n_nodes - 1)

        # Убираем self-loops — транзакция самому себе не имеет смысла
        if src == dst:
            dst = (dst + 1) % n_nodes

        amount = float(np_rng.lognormal(mean=5, sigma=1))
        timestamp = float(np_rng.uniform(0, 365))
        edges.append((src, dst, amount, timestamp))

    return {"nodes": nodes, "edges": edges}


def get_feature_matrix(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Табличное представление: матрица признаков X (n_nodes × 3) и метки y."""
    X = np.array(
        [[n["avg_amount"], n["n_transactions"], n["account_age_days"]] for n in data["nodes"]]
    )
    y = np.array([n["is_fraud"] for n in data["nodes"]])
    return X, y


def get_edge_index(data: dict) -> np.ndarray:
    """Индекс рёбер в формате PyTorch Geometric: (2 × n_edges)."""
    src = [e[0] for e in data["edges"]]
    dst = [e[1] for e in data["edges"]]
    return np.array([src, dst])
