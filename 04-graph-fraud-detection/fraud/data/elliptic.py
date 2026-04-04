"""
Загрузчик датасета Elliptic Bitcoin Dataset.
Elliptic Bitcoin Dataset loader.

Датасет Elliptic — крупнейший публичный граф транзакций Bitcoin с метками.
The Elliptic dataset is the largest public labeled Bitcoin transaction graph.

Структура / Structure:
  - 203 769 транзакций (узлы), 234 355 рёбер
  - 21% illicit (мошеннические), 79% licit (легальные) из размеченных
  - 49 временны́х шагов
  - 166 признаков: 94 локальных + 72 агрегированных по соседям

Если реальные CSV-файлы недоступны — генерирует mock-данные того же формата,
что позволяет запускать тесты без Kaggle-доступа.

If real CSV files are unavailable — generates mock data in the same format,
enabling tests to run without Kaggle access.

Реальные данные: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Real data: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Реальный датасет содержит ровно 166 признаков (94 локальных + 72 агрегированных).
# Первый признак — временно́й шаг (1–49).
ELLIPTIC_N_FEATURES = 166
ELLIPTIC_N_TIME_STEPS = 49


def generate_mock_elliptic(
    n_nodes: int = 500,
    n_edges: int = 800,
    seed: int = 42,
) -> dict:
    """Сгенерировать mock-данные в формате Elliptic Bitcoin Dataset.

    Generate mock data matching Elliptic Bitcoin Dataset format.

    Используется для тестов и когда реальный датасет недоступен.
    Used for tests and when the real dataset is unavailable.

    Сохраняет ключевые свойства реального датасета:
    - Доля мошенничества ~21% (как в Elliptic)
    - 166 признаков
    - Формат меток: 1=illicit, 2=licit, 0=unknown

    Returns:
        dict с ключами / dict with keys:
            node_ids: ndarray of shape (n_nodes,)
            features: ndarray of shape (n_nodes, 166)
            labels: ndarray (1=illicit, 2=licit, 0=unknown)
            edges: ndarray of shape (2, n_edges) — [src, dst]
            is_mock: True
    """
    rng = np.random.default_rng(seed)

    # ~21% illicit, как в реальном Elliptic
    fraud_rate = 0.21
    n_fraud = int(n_nodes * fraud_rate)

    # Метки: 1=illicit (мошеннические), 2=licit (легальные)
    labels = np.array([1] * n_fraud + [2] * (n_nodes - n_fraud), dtype=np.int32)
    rng.shuffle(labels)

    # Признак 0 — временно́й шаг (1–49), остальные — анонимизированы в реальных данных
    time_steps = rng.integers(1, ELLIPTIC_N_TIME_STEPS + 1, size=(n_nodes, 1)).astype(np.float32)
    other_features = rng.standard_normal((n_nodes, ELLIPTIC_N_FEATURES - 1)).astype(np.float32)
    features = np.hstack([time_steps, other_features])

    node_ids = np.arange(n_nodes, dtype=np.int64)

    # Рёбра без self-loops
    src = rng.integers(0, n_nodes, size=n_edges * 2)
    dst = rng.integers(0, n_nodes, size=n_edges * 2)
    no_self_loop = src != dst
    src, dst = src[no_self_loop][:n_edges], dst[no_self_loop][:n_edges]

    return {
        "node_ids": node_ids,
        "features": features,
        "labels": labels,
        "edges": np.stack([src, dst], axis=0),
        "is_mock": True,
    }


def load_elliptic_dataset(data_dir: str | Path | None = None) -> dict:
    """Загрузить датасет Elliptic Bitcoin из CSV или сгенерировать mock.

    Load Elliptic Bitcoin Dataset from CSV files or generate mock data.

    Ожидаемые файлы в data_dir (скачать с Kaggle):
    Expected files in data_dir (download from Kaggle):
      - elliptic_txs_features.csv  (no header; col0=txId, cols1–166=features)
      - elliptic_txs_edgelist.csv  (header: txId1,txId2)
      - elliptic_txs_classes.csv   (header: txId,class; class: 1=illicit,2=licit,unknown)

    Args:
        data_dir: путь к директории с CSV-файлами.
                  Path to directory with CSV files.
                  None → используется mock-данные.
                  None → mock data is used.

    Returns:
        dict с ключами / dict with keys:
            node_ids: ndarray — оригинальные txId из CSV (или 0..N-1 для mock)
            features: ndarray of shape (n_nodes, 166)
            labels: ndarray (1=illicit, 2=licit, 0=unknown)
            edges: ndarray of shape (2, n_edges)
            is_mock: bool
    """
    if data_dir is None:
        return generate_mock_elliptic()

    import pandas as pd  # pandas опциональный для тестов без CSV

    data_dir = Path(data_dir)
    features_path = data_dir / "elliptic_txs_features.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"
    classes_path = data_dir / "elliptic_txs_classes.csv"

    if not all(p.exists() for p in [features_path, edges_path, classes_path]):
        return generate_mock_elliptic()

    # Загрузка признаков (нет заголовка: col0=txId, cols1-166=features)
    features_df = pd.read_csv(features_path, header=None)
    node_ids_raw = features_df.iloc[:, 0].values.astype(np.int64)
    features = features_df.iloc[:, 1:].values.astype(np.float32)

    # Индекс: txId → позиция в массиве
    id_map: dict = {int(txid): i for i, txid in enumerate(node_ids_raw)}

    # Загрузка меток (unknown → 0)
    classes_df = pd.read_csv(classes_path)
    classes_df["class"] = classes_df["class"].replace("unknown", "0").astype(int)
    label_map: dict = dict(zip(classes_df["txId"].astype(int), classes_df["class"], strict=False))
    labels = np.array([label_map.get(int(txid), 0) for txid in node_ids_raw], dtype=np.int32)

    # Загрузка рёбер: маппинг txId → индекс
    edges_df = pd.read_csv(edges_path)
    valid = edges_df["txId1"].astype(int).isin(id_map) & edges_df["txId2"].astype(int).isin(id_map)
    edges_df = edges_df[valid]
    src = np.array([id_map[int(t)] for t in edges_df["txId1"]], dtype=np.int64)
    dst = np.array([id_map[int(t)] for t in edges_df["txId2"]], dtype=np.int64)

    return {
        "node_ids": node_ids_raw,
        "features": features,
        "labels": labels,
        "edges": np.stack([src, dst], axis=0),
        "is_mock": False,
    }


def get_labeled_split(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть только размеченные узлы (licit + illicit, без unknown).

    Return only labeled nodes (licit + illicit, excluding unknown).

    Конвертация меток: Elliptic использует 1/2, мы используем 0/1.
    Label conversion: Elliptic uses 1/2, we use binary 0/1.

    Args:
        data: вывод load_elliptic_dataset() / output of load_elliptic_dataset()

    Returns:
        X: ndarray shape (n_labeled, n_features)
        y: binary labels — 0=licit, 1=illicit
    """
    labels = data["labels"]
    features = data["features"]

    # Только размеченные: 1 (illicit) и 2 (licit); 0 — unknown, пропускаем
    mask = labels > 0
    X = features[mask]
    # 1 (illicit) → 1 (fraud), 2 (licit) → 0 (normal)
    y = (labels[mask] == 1).astype(np.int32)

    return X, y
