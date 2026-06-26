"""Graph centrality features for fraud detection.

Почему centrality критична для fraud detection?
- PageRank: money mule получает от многих источников → высокий PR без высокого degree.
- Betweenness: координатор fraud ring стоит на пути между многими парами узлов.
- Clustering coefficient: члены плотного кольца имеют высокий CC (много общих соседей).
- k-core: глубоко встроенные узлы сети — устойчивые fraud rings, нечувствительные
  к удалению периферии (Malliaros et al. 2020).

Все метрики реализованы на чистом numpy / stdlib — нет зависимости от NetworkX.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CentralityConfig:
    """Параметры экстрактора centrality-признаков."""

    damping: float = 0.85  # PageRank damping (Brin & Page 1998)
    pagerank_max_iter: int = 100
    pagerank_tol: float = 1e-6
    betweenness_k_sources: int = 10  # BFS-сэмплирование для O(k·E) аппроксимации
    seed: int = 42


@dataclass
class NodeCentralityFeatures:
    """Шесть centrality-признаков одного узла.

    Attributes:
        pagerank: вес PageRank (Brin & Page 1998) — influence в ориентированном графе.
        in_degree_centrality: нормированный in-degree [0,1] — «сколько отправляют этому узлу».
        out_degree_centrality: нормированный out-degree [0,1] — «скольким этот узел отправляет».
        betweenness_approx: приближённый betweenness [0,1] — роль посредника.
        clustering_coefficient: локальный CC [0,1] — плотность треугольников в окрестности.
        k_core_number: нормированный k-core номер [0,1] — глубина в ядре сети.
    """

    pagerank: float = 0.0
    in_degree_centrality: float = 0.0
    out_degree_centrality: float = 0.0
    betweenness_approx: float = 0.0
    clustering_coefficient: float = 0.0
    k_core_number: float = 0.0

    def to_array(self) -> np.ndarray:
        """Конвертировать в float32-вектор для подачи в модель."""
        return np.array(
            [
                self.pagerank,
                self.in_degree_centrality,
                self.out_degree_centrality,
                self.betweenness_approx,
                self.clustering_coefficient,
                self.k_core_number,
            ],
            dtype=np.float32,
        )


CENTRALITY_FEATURE_NAMES = [
    "pagerank",
    "in_degree_centrality",
    "out_degree_centrality",
    "betweenness_approx",
    "clustering_coefficient",
    "k_core_number",
]


@dataclass
class CentralityExtractResult:
    """Результат извлечения centrality-признаков для всего графа."""

    features: dict[str, NodeCentralityFeatures]  # node_id → features
    n_nodes: int
    n_edges: int
    max_pagerank: float
    max_k_core_raw: int  # до нормализации


class CentralityFeatureExtractor:
    """Извлекатель graph centrality признаков без внешних зависимостей.

    Алгоритмы:
        PageRank: power iteration, O(iter · E).
        Degree: O(E) — простой подсчёт in/out рёбер.
        Betweenness: BFS-сэмплирование k случайных источников, O(k · (V+E)).
        Clustering: O(V · d²) — перебор пар соседей (d = средняя степень).
        k-core: итеративное «срезание» листьев, O(V + E) амортизированно.
    """

    def __init__(self, config: CentralityConfig | None = None) -> None:
        self.config = config or CentralityConfig()

    def _build_graph(
        self,
        node_ids: list[str],
        edges: list[tuple[str, str]],
    ) -> tuple[int, dict[str, int], dict, dict, dict]:
        """Построить adjacency structures из node_ids и ориентированных рёбер."""
        n = len(node_ids)
        idx: dict[str, int] = {node: i for i, node in enumerate(node_ids)}

        out_adj: dict[int, set[int]] = defaultdict(set)
        in_adj: dict[int, set[int]] = defaultdict(set)
        undirected: dict[int, set[int]] = defaultdict(set)

        for u_id, v_id in edges:
            if u_id in idx and v_id in idx and u_id != v_id:
                u, v = idx[u_id], idx[v_id]
                out_adj[u].add(v)
                in_adj[v].add(u)
                undirected[u].add(v)
                undirected[v].add(u)

        return n, idx, out_adj, in_adj, undirected

    def _pagerank(
        self,
        n: int,
        out_adj: dict[int, set[int]],
        in_adj: dict[int, set[int]],
    ) -> np.ndarray:
        """Power iteration PageRank с обработкой dangling nodes."""
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.array([1.0])

        d = self.config.damping
        pr = np.full(n, 1.0 / n)
        out_deg = np.array([len(out_adj.get(i, set())) for i in range(n)], dtype=float)

        for _ in range(self.config.pagerank_max_iter):
            pr_new = np.full(n, (1.0 - d) / n)

            # Dangling-node mass redistributed uniformly (Langville & Meyer 2006).
            dangling_sum = float(np.sum(pr[out_deg == 0]))
            pr_new += d * dangling_sum / n

            for v in range(n):
                for u in in_adj.get(v, set()):
                    if out_deg[u] > 0:
                        pr_new[v] += d * pr[u] / out_deg[u]

            if float(np.max(np.abs(pr_new - pr))) < self.config.pagerank_tol:
                pr = pr_new
                break
            pr = pr_new

        total = pr.sum()
        return pr / total if total > 0 else pr

    def _degree_centrality(
        self,
        n: int,
        out_adj: dict[int, set[int]],
        in_adj: dict[int, set[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Нормированный in/out degree (деление на N-1)."""
        if n <= 1:
            return np.zeros(n), np.zeros(n)

        norm = float(n - 1)
        in_deg = np.array([len(in_adj.get(i, set())) for i in range(n)], dtype=float) / norm
        out_deg = np.array([len(out_adj.get(i, set())) for i in range(n)], dtype=float) / norm
        return in_deg, out_deg

    def _betweenness_approx(
        self,
        n: int,
        undirected: dict[int, set[int]],
    ) -> np.ndarray:
        """Приближённый betweenness через BFS от k случайных источников.

        Нормализация: делим на k · (N-1) · (N-2) / 2 — средний объём путей
        при случайной выборке источников (Brandes 2001, Journal of Mathematical Sociology).
        """
        if n <= 2:
            return np.zeros(n)

        rng = np.random.default_rng(self.config.seed)
        k = min(self.config.betweenness_k_sources, n)
        sources = rng.choice(n, size=k, replace=False).tolist()

        betweenness = np.zeros(n)

        for s in sources:
            dist: list[int] = [-1] * n
            dist[s] = 0
            pred: dict[int, list[int]] = defaultdict(list)
            sigma = np.zeros(n)
            sigma[s] = 1.0

            queue: deque[int] = deque([s])
            order: list[int] = []

            while queue:
                u = queue.popleft()
                order.append(u)
                for v in undirected.get(u, set()):
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    if dist[v] == dist[u] + 1:
                        sigma[v] += sigma[u]
                        pred[v].append(u)

            delta = np.zeros(n)
            for v in reversed(order):
                if v == s:
                    continue
                for u in pred.get(v, []):
                    if sigma[v] > 0:
                        delta[u] += (sigma[u] / sigma[v]) * (1.0 + delta[v])
            betweenness += delta

        denom = float(k) * float(n - 1) * float(n - 2) / 2.0
        if denom > 0:
            betweenness /= denom

        return np.clip(betweenness, 0.0, 1.0)

    def _clustering_coefficient(
        self,
        n: int,
        undirected: dict[int, set[int]],
    ) -> np.ndarray:
        """Локальный clustering coefficient для каждого узла.

        CC(v) = 2·t(v) / (k(v) · (k(v) − 1)), где t(v) = число треугольников через v.
        """
        cc = np.zeros(n)
        for v in range(n):
            neighbors = list(undirected.get(v, set()))
            k = len(neighbors)
            if k < 2:
                continue
            neighbor_set = undirected.get(v, set())
            triangles = sum(
                1
                for i in range(len(neighbors))
                for j in range(i + 1, len(neighbors))
                if neighbors[j] in undirected.get(neighbors[i], set())
                and neighbors[i] in neighbor_set
            )
            cc[v] = 2.0 * triangles / (k * (k - 1))
        return cc

    def _k_core(
        self,
        n: int,
        undirected: dict[int, set[int]],
    ) -> tuple[np.ndarray, int]:
        """K-core decomposition через итеративное «срезание» листьев.

        Возвращает (нормализованные номера, максимальный k-core до нормализации).
        Алгоритм: если degree(v) < k, удалить v из k-core и уменьшить degree соседей.
        """
        if n == 0:
            return np.array([]), 0

        core = np.array([len(undirected.get(v, set())) for v in range(n)], dtype=int)

        changed = True
        while changed:
            changed = False
            for v in range(n):
                k = core[v]
                if k == 0:
                    continue
                effective = sum(1 for u in undirected.get(v, set()) if core[u] >= k)
                new_k = min(k, effective)
                if new_k < core[v]:
                    core[v] = new_k
                    changed = True

        max_core = int(core.max()) if n > 0 else 0
        core_f = core.astype(float)
        if max_core > 0:
            core_f /= float(max_core)

        return core_f, max_core

    def extract(
        self,
        node_ids: list[str],
        edges: list[tuple[str, str]],
    ) -> CentralityExtractResult:
        """Извлечь centrality-признаки для всех узлов графа.

        Args:
            node_ids: список идентификаторов узлов.
            edges: рёбра (from_id, to_id) — могут ссылаться на несуществующие узлы (игнорируются).

        Returns:
            CentralityExtractResult с dict node_id → NodeCentralityFeatures.
        """
        n, idx, out_adj, in_adj, undirected = self._build_graph(node_ids, edges)

        pr = self._pagerank(n, out_adj, in_adj)
        in_deg, out_deg = self._degree_centrality(n, out_adj, in_adj)
        between = self._betweenness_approx(n, undirected)
        cluster = self._clustering_coefficient(n, undirected)
        kcore, max_kcore_raw = self._k_core(n, undirected)

        features: dict[str, NodeCentralityFeatures] = {}
        for node_id, i in idx.items():
            features[node_id] = NodeCentralityFeatures(
                pagerank=float(pr[i]) if n > 0 else 0.0,
                in_degree_centrality=float(in_deg[i]) if n > 0 else 0.0,
                out_degree_centrality=float(out_deg[i]) if n > 0 else 0.0,
                betweenness_approx=float(between[i]) if n > 0 else 0.0,
                clustering_coefficient=float(cluster[i]) if n > 0 else 0.0,
                k_core_number=float(kcore[i]) if n > 0 else 0.0,
            )

        valid_edges = [
            (u, v) for u, v in edges if u in idx and v in idx and u != v
        ]

        return CentralityExtractResult(
            features=features,
            n_nodes=n,
            n_edges=len(valid_edges),
            max_pagerank=float(pr.max()) if n > 0 else 0.0,
            max_k_core_raw=max_kcore_raw,
        )

    def augment_features(
        self,
        X: np.ndarray,
        node_ids: list[str],
        edges: list[tuple[str, str]],
    ) -> np.ndarray:
        """Дополнить матрицу признаков X centrality-вектором (строка = узел).

        X должен иметь len(node_ids) строк — порядок узлов совпадает.
        """
        result = self.extract(node_ids, edges)
        centrality_matrix = np.zeros((len(node_ids), 6), dtype=np.float32)
        for i, node in enumerate(node_ids):
            if node in result.features:
                centrality_matrix[i] = result.features[node].to_array()
        return np.hstack([X, centrality_matrix])


def explain_centrality_features(
    features: NodeCentralityFeatures,
    pr_threshold: float = 0.01,
    between_threshold: float = 0.05,
    cluster_threshold: float = 0.5,
    kcore_threshold: float = 0.7,
) -> dict[str, str]:
    """Человекочитаемые объяснения centrality-рисков (EU AI Act Article 13).

    Пороги по умолчанию настроены для типичных fraud-графов (500 узлов, sparse).
    В production пороги вычисляются как top-10% распределения по всему графу.
    """
    flags: dict[str, str] = {}

    if features.pagerank > pr_threshold:
        flags["pagerank"] = (
            f"High influence node (PR={features.pagerank:.4f}) — "
            "potential money mule aggregator or hub account"
        )
    if features.in_degree_centrality > 0.3:
        flags["in_degree"] = (
            f"High in-degree ({features.in_degree_centrality:.2f}) — "
            "many accounts send to this node (money mule pattern)"
        )
    if features.betweenness_approx > between_threshold:
        flags["betweenness"] = (
            f"High betweenness ({features.betweenness_approx:.3f}) — "
            "broker/intermediary role in transaction network"
        )
    if features.clustering_coefficient > cluster_threshold:
        flags["clustering"] = (
            f"Dense local cluster (CC={features.clustering_coefficient:.2f}) — "
            "part of tight-knit group (fraud ring indicator)"
        )
    if features.k_core_number > kcore_threshold:
        flags["k_core"] = (
            f"Core network member (k-core={features.k_core_number:.2f}) — "
            "deeply embedded in network core (hard to isolate)"
        )

    return flags if flags else {"status": "no_centrality_risk_flags"}
