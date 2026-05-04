"""Temporal graph features for fraud detection.

Почему временны́е признаки критичны для fraud detection?
Мошенники маскируются под легальных пользователей по статическим признакам,
но выдают себя *когда* и *как быстро* действуют:
- burst pattern: 20 транзакций за час, затем тишина
- temporal clustering: активность скачет, а не равномерна
- amount concentration: несколько крупных платежей vs. много мелких

Реализовано на чистом numpy — нет зависимости от PyTorch (macOS x86_64).
Источники: temporal GNN survey (arxiv 2302.01018), FinGuard-GNN 2025.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalConfig:
    """Configuration for temporal feature extraction."""

    time_window: float = 30.0  # дней — окно "недавней" активности
    decay_factor: float = 0.1  # экспоненциальное затухание веса старых рёбер
    min_edges_for_burst: int = 2  # минимум рёбер для вычисления burst score


@dataclass
class NodeTemporalFeatures:
    """Временны́е признаки одного узла.

    Attributes:
        velocity_ratio: доля транзакций в окне time_window (0-1).
        burst_score: CV временны́х интервалов — высокий = нерегулярность.
        amount_hhi: Herfindahl-Hirschman Index сумм — высокий = концентрация.
        recent_amount_ratio: доля объёма в окне time_window (0-1).
        neighbor_fraud_ratio: доля соседей-мошенников в графе (0-1).
        hub_proximity: log(1 + средняя степень соседей).
    """

    velocity_ratio: float = 0.0
    burst_score: float = 0.0
    amount_hhi: float = 0.0
    recent_amount_ratio: float = 0.0
    neighbor_fraud_ratio: float = 0.0
    hub_proximity: float = 0.0

    def to_array(self) -> np.ndarray:
        """Вернуть признаки как вектор (6,)."""
        return np.array(
            [
                self.velocity_ratio,
                self.burst_score,
                self.amount_hhi,
                self.recent_amount_ratio,
                self.neighbor_fraud_ratio,
                self.hub_proximity,
            ],
            dtype=np.float32,
        )


TEMPORAL_FEATURE_NAMES = [
    "velocity_ratio",
    "burst_score",
    "amount_hhi",
    "recent_amount_ratio",
    "neighbor_fraud_ratio",
    "hub_proximity",
]


class TemporalFeatureExtractor:
    """Извлекает временны́е граф-признаки для всех узлов.

    Использует только numpy и структуру данных из dataset.py:
      data["nodes"]: list[{id, avg_amount, n_transactions, account_age_days, is_fraud}]
      data["edges"]: list[(src, dst, amount, timestamp)]

    Args:
        config: настройки окна и затухания.
    """

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()

    def _build_adjacency(self, data: dict) -> dict[int, set[int]]:
        """Построить неориентированный граф смежности."""
        adj: dict[int, set[int]] = {n["id"]: set() for n in data["nodes"]}
        for src, dst, _amt, _ts in data["edges"]:
            adj[src].add(dst)
            adj[dst].add(src)
        return adj

    def _node_edges(self, data: dict, node_id: int) -> list[tuple]:
        """Все рёбра (входящие + исходящие) для узла."""
        return [(s, d, a, t) for s, d, a, t in data["edges"] if s == node_id or d == node_id]

    def compute_node_features(self, data: dict, node_id: int) -> NodeTemporalFeatures:
        """Вычислить 6 временны́х признаков для одного узла.

        Args:
            data: граф транзакций из generate_synthetic_transactions().
            node_id: индекс узла.

        Returns:
            NodeTemporalFeatures с заполненными полями.
        """
        edges = self._node_edges(data, node_id)
        feat = NodeTemporalFeatures()

        if edges:
            timestamps = [e[3] for e in edges]
            amounts = [e[2] for e in edges]
            t_max = max(timestamps)
            window_start = t_max - self.config.time_window

            # Velocity ratio — какой % транзакций пришёлся на последнее окно
            recent_mask = [t >= window_start for t in timestamps]
            feat.velocity_ratio = sum(recent_mask) / len(timestamps)

            # Burst score — CV интервалов (высокий = нерегулярная активность)
            if len(timestamps) >= self.config.min_edges_for_burst:
                sorted_ts = sorted(timestamps)
                gaps = np.diff(sorted_ts)
                mean_gap = float(np.mean(gaps))
                feat.burst_score = float(np.std(gaps)) / (mean_gap + 1e-8)

            # Amount HHI — концентрация сумм (1.0 = одна транзакция на всё)
            total_amount = sum(amounts)
            if total_amount > 0:
                shares = [a / total_amount for a in amounts]
                feat.amount_hhi = float(sum(s**2 for s in shares))

            # Recent amount ratio — какой % объёма в окне
            recent_amounts = [a for a, m in zip(amounts, recent_mask) if m]
            feat.recent_amount_ratio = sum(recent_amounts) / (total_amount + 1e-8)

        return feat

    def compute_neighborhood_features(
        self,
        data: dict,
        node_id: int,
        adj: dict[int, set[int]],
    ) -> tuple[float, float]:
        """Вычислить признаки соседства для узла.

        Args:
            data: граф транзакций.
            node_id: индекс узла.
            adj: предвычисленная смежность.

        Returns:
            (neighbor_fraud_ratio, hub_proximity)
        """
        fraud_ids = {n["id"] for n in data["nodes"] if n.get("is_fraud", 0)}
        neighbors = adj.get(node_id, set())

        if not neighbors:
            return 0.0, 0.0

        fraud_ratio = len(neighbors & fraud_ids) / len(neighbors)
        avg_neighbor_degree = float(np.mean([len(adj.get(nb, set())) for nb in neighbors]))
        hub_proximity = float(np.log1p(avg_neighbor_degree))

        return fraud_ratio, hub_proximity

    def extract(self, data: dict) -> np.ndarray:
        """Извлечь временны́е признаки для всех узлов.

        Args:
            data: граф транзакций.

        Returns:
            Матрица (n_nodes, 6) с признаками TEMPORAL_FEATURE_NAMES.
        """
        n_nodes = len(data["nodes"])
        result = np.zeros((n_nodes, 6), dtype=np.float32)
        adj = self._build_adjacency(data)

        for i, node in enumerate(data["nodes"]):
            node_id = node["id"]
            feat = self.compute_node_features(data, node_id)
            fraud_ratio, hub_prox = self.compute_neighborhood_features(data, node_id, adj)
            feat.neighbor_fraud_ratio = fraud_ratio
            feat.hub_proximity = hub_prox
            result[i] = feat.to_array()

        return result

    def augment_features(self, X: np.ndarray, data: dict) -> np.ndarray:
        """Дополнить табличные признаки временны́ми граф-признаками.

        Args:
            X: базовые признаки (n_nodes, n_base_features).
            data: граф транзакций.

        Returns:
            Расширенная матрица (n_nodes, n_base_features + 6).
        """
        temporal = self.extract(data)
        return np.hstack([X, temporal])


def explain_temporal_features(features: NodeTemporalFeatures) -> dict[str, str]:
    """Человекочитаемые объяснения временны́х признаков.

    Полезно для аудита и compliance (EU AI Act Article 13 — transparency).

    Args:
        features: временны́е признаки узла.

    Returns:
        Словарь признак → текстовое объяснение уровня риска.
    """
    explanations: dict[str, str] = {}

    if features.velocity_ratio > 0.8:
        explanations["velocity_ratio"] = "HIGH: >80% транзакций в последнем окне — burst activity"
    elif features.velocity_ratio > 0.5:
        explanations["velocity_ratio"] = "MEDIUM: заметный всплеск активности"
    else:
        explanations["velocity_ratio"] = "LOW: равномерная активность"

    if features.burst_score > 3.0:
        explanations["burst_score"] = "HIGH: CV > 3 — крайне нерегулярные интервалы"
    elif features.burst_score > 1.0:
        explanations["burst_score"] = "MEDIUM: нерегулярные временны́е интервалы"
    else:
        explanations["burst_score"] = "LOW: регулярная активность"

    if features.amount_hhi > 0.7:
        explanations["amount_hhi"] = "HIGH: HHI > 0.7 — одна-две транзакции доминируют"
    elif features.amount_hhi > 0.3:
        explanations["amount_hhi"] = "MEDIUM: умеренная концентрация сумм"
    else:
        explanations["amount_hhi"] = "LOW: равномерное распределение сумм"

    if features.neighbor_fraud_ratio > 0.3:
        explanations["neighbor_fraud_ratio"] = f"HIGH: {features.neighbor_fraud_ratio:.0%} соседей — мошенники"
    elif features.neighbor_fraud_ratio > 0.1:
        explanations["neighbor_fraud_ratio"] = "MEDIUM: часть соседей помечена как fraud"
    else:
        explanations["neighbor_fraud_ratio"] = "LOW: соседи без fraud-меток"

    return explanations
