"""Fraud ring detection via Label Propagation.

Зачем детекция колец важна?
Мошенничество редко индивидуально — 60-80% потерь от организованных атак
(fraud rings, account farms). Индивидуальная оценка транзакций упускает паттерн:
каждый участник кольца может иметь низкий score, но как сеть — высокий.

Алгоритм: Label Propagation (Raghavan et al. 2007, Physical Review E 76, 036106).
Асинхронное обновление меток O(E) — без NetworkX, только numpy для shuffle+rng.
Tie-breaking: лексикографически наименьшая метка → детерминированный результат
при фиксированном seed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CommunityConfig:
    """Конфигурация детектора мошеннических колец."""

    max_iterations: int = 100
    fraud_ratio_high: float = 0.3  # >30% мошенников → высокий риск
    fraud_ratio_medium: float = 0.1  # >10% → средний риск
    min_ring_size: int = 2  # минимум узлов для "кольца" (не одиночка)
    seed: int = 42


@dataclass
class CommunityResult:
    """Описание одного сообщества (потенциального мошеннического кольца).

    Attributes:
        community_id: уникальный идентификатор (ранг по убыванию размера).
        size: количество узлов.
        fraud_ratio: доля labeled узлов с known fraud=True (0-1).
        risk_level: 'high' / 'medium' / 'low'.
        node_ids: отсортированный список node_id в сообществе.
    """

    community_id: int
    size: int
    fraud_ratio: float
    risk_level: str
    node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "community_id": self.community_id,
            "size": self.size,
            "fraud_ratio": round(self.fraud_ratio, 4),
            "risk_level": self.risk_level,
            "node_ids": self.node_ids,
        }


@dataclass
class DetectionResult:
    """Результат детекции мошеннических колец.

    Attributes:
        communities: все обнаруженные сообщества, отсортированные по убыванию размера.
        suspicious_rings: подмножество с risk_level != 'low' и size >= min_ring_size.
        n_iterations: количество итераций до сходимости.
        converged: True если Label Propagation сошёлся до max_iter.
        total_nodes: общее количество проанализированных узлов.
    """

    communities: list[CommunityResult]
    suspicious_rings: list[CommunityResult]
    n_iterations: int
    converged: bool
    total_nodes: int

    @property
    def n_communities(self) -> int:
        return len(self.communities)

    def to_dict(self) -> dict:
        return {
            "n_communities": self.n_communities,
            "communities": [c.to_dict() for c in self.communities],
            "suspicious_rings": [c.to_dict() for c in self.suspicious_rings],
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "total_nodes": self.total_nodes,
        }


class FraudRingDetector:
    """Детектор мошеннических колец через Label Propagation.

    Fraud scoring: внутри сообщества fraud_ratio = labeled_fraud / total_labeled.
    Узлы без known fraud label не влияют на знаменатель (честная оценка по evidence).
    Высокий fraud_ratio → кольцо с координированным мошенничеством.
    """

    def __init__(self, config: CommunityConfig | None = None) -> None:
        self.config = config or CommunityConfig()

    def detect(
        self,
        node_ids: list[str],
        edges: list[tuple[str, str]],
        fraud_labels: dict[str, bool] | None = None,
    ) -> DetectionResult:
        """Обнаружить мошеннические кольца в транзакционном графе.

        Args:
            node_ids: список всех узлов (аккаунты / транзакции).
            edges: рёбра как (from_id, to_id) — граф неориентированный.
            fraud_labels: известные метки {node_id: is_fraud}.

        Returns:
            DetectionResult с сообществами и подозрительными кольцами.

        Raises:
            ValueError: если node_ids пуст.
        """
        if not node_ids:
            raise ValueError("node_ids cannot be empty")

        fraud_labels = fraud_labels or {}

        adj: dict[str, set[str]] = defaultdict(set)
        for node in node_ids:
            if node not in adj:
                adj[node] = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        labels, n_iter, converged = self._label_propagation(node_ids, adj)

        communities_map: dict[str, list[str]] = defaultdict(list)
        for node, label in labels.items():
            communities_map[label].append(node)

        communities: list[CommunityResult] = []
        for community_id, (_, members) in enumerate(
            sorted(communities_map.items(), key=lambda x: -len(x[1]))
        ):
            labeled_in_comm = [n for n in members if n in fraud_labels]
            fraud_count = sum(1 for n in labeled_in_comm if fraud_labels[n])

            fraud_ratio = fraud_count / len(labeled_in_comm) if labeled_in_comm else 0.0
            risk_level = self._risk_level(fraud_ratio)

            communities.append(
                CommunityResult(
                    community_id=community_id,
                    size=len(members),
                    fraud_ratio=fraud_ratio,
                    risk_level=risk_level,
                    node_ids=sorted(members),
                )
            )

        suspicious = [
            c
            for c in communities
            if c.risk_level != "low" and c.size >= self.config.min_ring_size
        ]

        return DetectionResult(
            communities=communities,
            suspicious_rings=suspicious,
            n_iterations=n_iter,
            converged=converged,
            total_nodes=len(node_ids),
        )

    def _label_propagation(
        self,
        node_ids: list[str],
        adj: dict[str, set[str]],
    ) -> tuple[dict[str, str], int, bool]:
        """Асинхронный Label Propagation с детерминированным tie-breaking.

        Tie-breaking по минимальной метке делает алгоритм детерминированным
        при фиксированном seed — важно для воспроизводимых аудит-логов (EU AI Act).
        """
        import numpy as np

        rng = np.random.RandomState(self.config.seed)

        labels = {n: n for n in node_ids}

        for iteration in range(self.config.max_iterations):
            old_labels = labels.copy()

            shuffled = list(node_ids)
            rng.shuffle(shuffled)

            for node in shuffled:
                neighbors = adj[node]
                if not neighbors:
                    continue

                label_counts: dict[str, int] = defaultdict(int)
                for neighbor in neighbors:
                    label_counts[labels[neighbor]] += 1

                max_count = max(label_counts.values())
                candidates = [lb for lb, cnt in label_counts.items() if cnt == max_count]
                labels[node] = min(candidates)

            if labels == old_labels:
                return labels, iteration + 1, True

        return labels, self.config.max_iterations, False

    def _risk_level(self, fraud_ratio: float) -> str:
        if fraud_ratio >= self.config.fraud_ratio_high:
            return "high"
        elif fraud_ratio >= self.config.fraud_ratio_medium:
            return "medium"
        return "low"
