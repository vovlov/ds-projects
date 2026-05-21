"""
Labeling pool manager for active learning annotation workflow.

State machine per item: unlabeled → queried → labeled.
Pool is in-memory; restart resets state (acceptable for portfolio / annotation sessions).

Usage flow:
  1. POST /active/pool/add  — annotator uploads batch of raw texts
  2. POST /active/pool/query — service returns top-N most uncertain texts
  3. POST /active/pool/label — annotator submits entity annotations per item
  4. GET  /active/pool/labeled — retrieve all labeled examples for fine-tuning
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PoolItem:
    """Один текст в пуле аннотации."""

    id: str
    text: str
    uncertainty_score: float
    strategy: str
    queried_at: str | None = None
    labeled_at: str | None = None
    annotations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class QueryBatch:
    """Результат запроса аннотации: топ-N текстов для разметки."""

    items: list[PoolItem]
    strategy: str
    batch_size: int
    unlabeled_remaining: int


@dataclass
class PoolStatus:
    """Текущий статус пула аннотации."""

    unlabeled_count: int
    queried_count: int
    labeled_count: int
    total_added: int
    strategy: str


class LabelingPool:
    """
    Менеджер пула активного обучения.

    Хранит три корзины:
      _unlabeled — новые тексты, ещё не отобранные
      _queried   — отобраны для аннотации, ожидают разметки
      _labeled   — размеченные данные (готовы для fine-tuning)

    Сортировка при query: по убыванию uncertainty_score →
    аннотатор получает самые информативные примеры первыми.
    """

    def __init__(self) -> None:
        self._unlabeled: dict[str, PoolItem] = {}
        self._queried: dict[str, PoolItem] = {}
        self._labeled: dict[str, PoolItem] = {}
        self._total_added: int = 0

    def add_texts(
        self,
        texts: list[str],
        uncertainty_scores: list[float],
        strategy: str,
    ) -> list[str]:
        """Добавить тексты в пул, вернуть их ID."""
        if len(texts) != len(uncertainty_scores):
            raise ValueError("texts и uncertainty_scores должны быть одной длины")

        ids: list[str] = []
        for text, score in zip(texts, uncertainty_scores, strict=True):
            item_id = str(uuid.uuid4())
            self._unlabeled[item_id] = PoolItem(
                id=item_id,
                text=text,
                uncertainty_score=score,
                strategy=strategy,
            )
            ids.append(item_id)

        self._total_added += len(texts)
        return ids

    def query(self, batch_size: int) -> QueryBatch:
        """
        Вернуть топ-N наиболее неопределённых текстов.
        Переводит их из unlabeled → queried.
        """
        sorted_items = sorted(
            self._unlabeled.values(),
            key=lambda x: x.uncertainty_score,
            reverse=True,
        )
        selected = sorted_items[:batch_size]

        now = datetime.now(UTC).isoformat()
        for item in selected:
            item.queried_at = now
            self._queried[item.id] = item
            del self._unlabeled[item.id]

        strategy = selected[0].strategy if selected else "unknown"
        return QueryBatch(
            items=selected,
            strategy=strategy,
            batch_size=batch_size,
            unlabeled_remaining=len(self._unlabeled),
        )

    def label(
        self,
        item_id: str,
        annotations: list[dict[str, Any]],
    ) -> PoolItem | None:
        """
        Принять разметку для item_id.
        Переводит из queried → labeled.
        Возвращает None если item_id не найден в очереди.
        """
        if item_id not in self._queried:
            return None

        item = self._queried.pop(item_id)
        item.labeled_at = datetime.now(UTC).isoformat()
        item.annotations = annotations
        self._labeled[item_id] = item
        return item

    def get_labeled(self) -> list[PoolItem]:
        """Вернуть все размеченные примеры (для fine-tuning)."""
        return list(self._labeled.values())

    def status(self, strategy: str) -> PoolStatus:
        return PoolStatus(
            unlabeled_count=len(self._unlabeled),
            queried_count=len(self._queried),
            labeled_count=len(self._labeled),
            total_added=self._total_added,
            strategy=strategy,
        )

    def reset(self) -> None:
        """Сбросить пул (для тестирования)."""
        self._unlabeled.clear()
        self._queried.clear()
        self._labeled.clear()
        self._total_added = 0
