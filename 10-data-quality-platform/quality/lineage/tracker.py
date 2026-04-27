"""
Трекер событий родословной данных / Data lineage event tracker.

Реализует OpenLineage RunEvent модель: START → COMPLETE/FAIL/ABORT.
Каждый RunEvent описывает состояние выполнения задачи с её входами и выходами.

Implements the OpenLineage RunEvent model: START → COMPLETE/FAIL/ABORT.
Each event records the observable state of a job run and its datasets.

Sources:
  - OpenLineage spec: https://openlineage.io/docs/spec/object-model/
  - RunEvent types: https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from quality.lineage.graph import LineageGraph, LineageNode


class RunState(StrEnum):
    """Состояние выполнения задачи / Job run lifecycle state (OpenLineage)."""

    START = "START"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"
    ABORT = "ABORT"
    RUNNING = "RUNNING"


@dataclass
class LineageEvent:
    """
    Событие родословной в формате OpenLineage RunEvent.
    Lineage event in OpenLineage RunEvent format.

    Attributes:
        event_id: UUID события / Event UUID
        event_type: Тип события (START/COMPLETE/FAIL)
        job_namespace: Пространство имён задачи (сервис, проект)
        job_name: Имя задачи / Job name
        run_id: UUID запуска (один запуск = несколько событий START+COMPLETE)
        inputs: Входные датасеты / Input datasets
        outputs: Выходные датасеты / Output datasets
        event_time: ISO 8601 timestamp (UTC)
        facets: Метаданные (метрики, версии, теги) / Metadata facets
    """

    event_id: str
    event_type: RunState
    job_namespace: str
    job_name: str
    run_id: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    event_time: str
    facets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в OpenLineage-совместимый JSON."""
        return {
            "eventType": self.event_type.value,
            "eventTime": self.event_time,
            "run": {
                "runId": self.run_id,
                "facets": self.facets.get("run", {}),
            },
            "job": {
                "namespace": self.job_namespace,
                "name": self.job_name,
                "facets": self.facets.get("job", {}),
            },
            "inputs": self.inputs,
            "outputs": self.outputs,
            "_meta": {"event_id": self.event_id},
        }


class LineageTracker:
    """
    Трекер событий родословной + построитель lineage-графа.
    Lineage event tracker that builds and maintains a lineage graph.

    Хранит события в памяти (без внешних зависимостей, CI-friendly).
    Автоматически обновляет LineageGraph при каждом COMPLETE-событии.

    Stores events in-memory (no external deps, CI-friendly).
    Automatically updates the LineageGraph on each COMPLETE event.
    """

    def __init__(self) -> None:
        self._events: list[LineageEvent] = []
        self._graph = LineageGraph()

    # ------------------------------------------------------------------
    # Запись событий / Event recording
    # ------------------------------------------------------------------

    def record(
        self,
        job_namespace: str,
        job_name: str,
        event_type: RunState | str,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
        run_id: str | None = None,
        **facets: Any,
    ) -> LineageEvent:
        """
        Записать событие родословной / Record a lineage event.

        Args:
            job_namespace: Пространство имён сервиса (e.g. "churn-service")
            job_name: Имя операции (e.g. "train_model", "score_batch")
            event_type: START / COMPLETE / FAIL / ABORT
            inputs: Список входных датасетов [{"namespace": "...", "name": "..."}]
            outputs: Список выходных датасетов
            run_id: UUID запуска (авто-генерируется если None)
            **facets: Дополнительные метаданные (метрики, теги)

        Returns:
            Записанное событие / Recorded LineageEvent
        """
        if isinstance(event_type, str):
            event_type = RunState(event_type.upper())

        run_id = run_id or str(uuid.uuid4())
        event = LineageEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            job_namespace=job_namespace,
            job_name=job_name,
            run_id=run_id,
            inputs=inputs or [],
            outputs=outputs or [],
            event_time=datetime.now(tz=UTC).isoformat(),
            facets=facets,
        )
        self._events.append(event)

        # Обновляем граф только для финальных состояний (избегаем дублей)
        # Only update graph on terminal/complete states to avoid duplicate edges
        if event_type in (RunState.COMPLETE, RunState.START):
            self._update_graph(event)

        return event

    def _update_graph(self, event: LineageEvent) -> None:
        """Синхронизировать граф с событием / Sync graph from event."""
        job_node = LineageNode.job(namespace=event.job_namespace, name=event.job_name)
        input_nodes = [
            LineageNode.dataset(namespace=d.get("namespace", "default"), name=d["name"])
            for d in event.inputs
        ]
        output_nodes = [
            LineageNode.dataset(namespace=d.get("namespace", "default"), name=d["name"])
            for d in event.outputs
        ]
        self._graph.add_lineage(
            job=job_node,
            inputs=input_nodes,
            outputs=output_nodes,
            run_id=event.run_id,
        )

    # ------------------------------------------------------------------
    # Запросы / Queries
    # ------------------------------------------------------------------

    def get_graph(self) -> LineageGraph:
        """Получить текущий lineage-граф / Get the current lineage graph."""
        return self._graph

    def get_events(
        self,
        job_name: str | None = None,
        event_type: RunState | None = None,
        limit: int = 100,
    ) -> list[LineageEvent]:
        """
        Получить отфильтрованный список событий / Get filtered event list.

        Args:
            job_name: Фильтр по имени задачи / Filter by job name
            event_type: Фильтр по типу события / Filter by event type
            limit: Максимальное количество событий (последние N)
        """
        events = self._events
        if job_name:
            events = [e for e in events if e.job_name == job_name]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def get_run_history(self, run_id: str) -> list[LineageEvent]:
        """Получить все события одного запуска / Get all events for a run ID."""
        return [e for e in self._events if e.run_id == run_id]

    def summary(self) -> dict[str, Any]:
        """Краткая статистика трекера / Tracker summary statistics."""
        graph_dict = self._graph.to_dict()
        return {
            "total_events": len(self._events),
            "graph_stats": graph_dict["stats"],
            "event_type_counts": {
                state.value: sum(1 for e in self._events if e.event_type == state)
                for state in RunState
            },
        }


# Глобальный экземпляр трекера для API / Global tracker instance for API use
_tracker: LineageTracker | None = None


def get_tracker() -> LineageTracker:
    """
    Синглтон-трекер для FastAPI / Singleton tracker for FastAPI.

    Возвращает глобальный трекер, создаёт при первом вызове.
    Returns the global tracker, initializing it on first call.
    """
    global _tracker
    if _tracker is None:
        _tracker = LineageTracker()
    return _tracker


def reset_tracker() -> None:
    """Сбросить трекер (для тестов) / Reset tracker singleton (for testing)."""
    global _tracker
    _tracker = None
