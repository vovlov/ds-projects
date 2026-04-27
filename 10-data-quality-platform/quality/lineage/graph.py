"""
Граф родословной данных / Data lineage graph.

Реализует OpenLineage-совместимую модель данных:
  - Dataset  — таблица, файл, топик Kafka, модель ML
  - Job      — процесс трансформации (пайплайн, модель, API-вызов)
  - Ребро    — направление потока данных (dataset → job, job → dataset)

Implements an OpenLineage-compatible data model without external dependencies,
suitable for CI environments. The graph supports upstream/downstream traversal
and JSON export compatible with D3.js force-directed graphs.

Sources:
  - OpenLineage spec: https://openlineage.io/docs/spec/object-model/
  - OpenLineage GitHub: https://github.com/OpenLineage/OpenLineage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    """Типы узлов в lineage-графе / Node types in the lineage graph."""

    DATASET = "dataset"
    JOB = "job"


@dataclass
class LineageNode:
    """
    Узел графа родословной / Lineage graph node.

    Соответствует Dataset или Job из спецификации OpenLineage.
    Represents either a Dataset or a Job in the OpenLineage object model.

    Attributes:
        node_id: Уникальный идентификатор (namespace + name)
        node_type: Тип узла (dataset / job)
        namespace: Пространство имён (база данных, кластер Spark, имя сервиса)
        name: Имя ресурса (таблица, задача, API-эндпоинт)
        facets: Произвольные метаданные (схема, качество, версия модели)
    """

    node_id: str
    node_type: NodeType
    namespace: str
    name: str
    facets: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def dataset(cls, namespace: str, name: str, **facets: Any) -> LineageNode:
        """Создать узел-датасет / Create a dataset node."""
        node_id = f"{namespace}/{name}"
        return cls(node_id=node_id, node_type=NodeType.DATASET, namespace=namespace, name=name, facets=facets)  # noqa: E501

    @classmethod
    def job(cls, namespace: str, name: str, **facets: Any) -> LineageNode:
        """Создать узел-задачу / Create a job node."""
        node_id = f"{namespace}/{name}"
        return cls(node_id=node_id, node_type=NodeType.JOB, namespace=namespace, name=name, facets=facets)  # noqa: E501

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в JSON-совместимый словарь."""
        return {
            "id": self.node_id,
            "type": self.node_type.value,
            "namespace": self.namespace,
            "name": self.name,
            "facets": self.facets,
        }


@dataclass
class LineageEdge:
    """
    Ребро графа родословной / Lineage graph edge.

    Направленное ребро между двумя узлами. Стандартные направления:
      - dataset → job  (датасет как вход задачи / dataset as job input)
      - job → dataset  (датасет как выход задачи / dataset as job output)

    Directed edge between two nodes. Standard directions:
      - dataset → job  (input dataset consumed by a job)
      - job → dataset  (output dataset produced by a job)
    """

    source_id: str
    target_id: str
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в JSON-совместимый словарь."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "run_id": self.run_id,
            "metadata": self.metadata,
        }


class LineageGraph:
    """
    Направленный ациклический граф родословной данных.
    Directed acyclic graph of data lineage.

    Хранит узлы (Dataset + Job) и рёбра потока данных.
    Поддерживает обход вверх/вниз по графу и экспорт в формат D3.js.

    Stores Dataset and Job nodes with directed data-flow edges.
    Supports upstream/downstream traversal and D3.js-compatible JSON export.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []

    # ------------------------------------------------------------------
    # Мутации / Mutations
    # ------------------------------------------------------------------

    def add_node(self, node: LineageNode) -> None:
        """Добавить или обновить узел / Add or update a node."""
        self._nodes[node.node_id] = node

    def add_edge(self, edge: LineageEdge) -> None:
        """Добавить ребро / Add a directed edge (duplicate-safe)."""
        key = (edge.source_id, edge.target_id)
        existing = {(e.source_id, e.target_id) for e in self._edges}
        if key not in existing:
            self._edges.append(edge)

    def add_lineage(
        self,
        job: LineageNode,
        inputs: list[LineageNode],
        outputs: list[LineageNode],
        run_id: str | None = None,
    ) -> None:
        """
        Атомарно добавить job + его входные и выходные датасеты.
        Atomically register a job together with its input and output datasets.

        Создаёт рёбра: input_dataset → job → output_dataset.
        """
        self.add_node(job)
        for inp in inputs:
            self.add_node(inp)
            self.add_edge(LineageEdge(source_id=inp.node_id, target_id=job.node_id, run_id=run_id))
        for out in outputs:
            self.add_node(out)
            self.add_edge(LineageEdge(source_id=job.node_id, target_id=out.node_id, run_id=run_id))

    # ------------------------------------------------------------------
    # Обход / Traversal
    # ------------------------------------------------------------------

    def upstream(self, node_id: str, max_depth: int = 10) -> list[str]:
        """
        Получить все узлы выше по потоку (предки) / Get all upstream ancestor node IDs.

        Обход в ширину по обратным рёбрам / BFS on reversed edges.
        """
        visited: set[str] = set()
        queue = [node_id]
        for _ in range(max_depth):
            if not queue:
                break
            current = queue.pop(0)
            for edge in self._edges:
                if edge.target_id == current and edge.source_id not in visited:
                    visited.add(edge.source_id)
                    queue.append(edge.source_id)
        return list(visited)

    def downstream(self, node_id: str, max_depth: int = 10) -> list[str]:
        """
        Получить все узлы ниже по потоку (потомки) / Get all downstream descendant node IDs.

        Обход в ширину по прямым рёбрам / BFS on forward edges.
        """
        visited: set[str] = set()
        queue = [node_id]
        for _ in range(max_depth):
            if not queue:
                break
            current = queue.pop(0)
            for edge in self._edges:
                if edge.source_id == current and edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)
        return list(visited)

    def lineage_for_dataset(self, dataset_id: str) -> dict[str, Any]:
        """
        Полная родословная конкретного датасета (вверх и вниз).
        Full lineage for a specific dataset (both upstream and downstream).
        """
        up = self.upstream(dataset_id)
        down = self.downstream(dataset_id)
        all_ids = {dataset_id} | set(up) | set(down)
        return {
            "dataset_id": dataset_id,
            "upstream": up,
            "downstream": down,
            "nodes": [self._nodes[nid].to_dict() for nid in all_ids if nid in self._nodes],
            "edges": [
                e.to_dict()
                for e in self._edges
                if e.source_id in all_ids and e.target_id in all_ids
            ],
        }

    # ------------------------------------------------------------------
    # Сериализация / Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Сериализовать граф в формат D3.js force-directed / Serialize to D3.js graph format.

        Формат совместим с D3 force simulation (nodes + links).
        Format is compatible with D3 force simulation (nodes + links).
        """
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "links": [e.to_dict() for e in self._edges],
            "stats": {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "dataset_count": sum(1 for n in self._nodes.values() if n.node_type == NodeType.DATASET),  # noqa: E501
                "job_count": sum(1 for n in self._nodes.values() if n.node_type == NodeType.JOB),
            },
        }

    @property
    def nodes(self) -> dict[str, LineageNode]:
        """Все узлы графа / All graph nodes."""
        return dict(self._nodes)

    @property
    def edges(self) -> list[LineageEdge]:
        """Все рёбра графа / All graph edges."""
        return list(self._edges)
