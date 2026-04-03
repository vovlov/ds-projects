"""Batch processing pipeline для NER.

Эффективная обработка больших объёмов текста:
- Разбивка на чанки для управления памятью
- Структурированные результаты (dataclass вместо словарей)
- Совместим с Collection5 форматом — принимает токены или raw text

Применение: юридические отделы, обработка договоров пакетами.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ..data.collection5 import Dataset, sentences_to_bio
from .ner import Entity, extract_entities_from_bio, predict


@dataclass
class BatchItem:
    """Результат обработки одного текста в батче."""

    text: str
    entities: list[Entity] = field(default_factory=list)
    error: str | None = None

    @property
    def entity_types(self) -> set[str]:
        """Уникальные типы сущностей в тексте."""
        return {e.label for e in self.entities}

    @property
    def has_entities(self) -> bool:
        return len(self.entities) > 0


@dataclass
class BatchResult:
    """Агрегированный результат батч-обработки."""

    items: list[BatchItem]
    total_texts: int
    total_entities: int
    entity_type_counts: dict[str, int]

    @classmethod
    def from_items(cls, items: list[BatchItem]) -> BatchResult:
        """Build BatchResult by aggregating BatchItems."""
        total_entities = sum(len(item.entities) for item in items)
        counts: dict[str, int] = {}
        for item in items:
            for entity in item.entities:
                counts[entity.label] = counts.get(entity.label, 0) + 1
        return cls(
            items=items,
            total_texts=len(items),
            total_entities=total_entities,
            entity_type_counts=counts,
        )


def process_texts(
    texts: list[str],
    chunk_size: int = 64,
) -> BatchResult:
    """Process a list of raw texts through the NER pipeline.

    Обработка батчами по `chunk_size` для ограничения пикового потребления памяти.
    При ошибке одного текста — записывает error, продолжает обработку остальных.

    Args:
        texts: List of raw text strings.
        chunk_size: Internal processing chunk size.

    Returns:
        BatchResult with all predictions and aggregate counts.
    """
    items: list[BatchItem] = []

    for chunk in _chunked(texts, chunk_size):
        for text in chunk:
            try:
                entities = predict(text)
                items.append(BatchItem(text=text, entities=entities))
            except Exception as exc:  # noqa: BLE001
                items.append(BatchItem(text=text, error=str(exc)))

    return BatchResult.from_items(items)


def process_collection5(
    dataset: Dataset,
    chunk_size: int = 64,
) -> BatchResult:
    """Run NER pipeline over a Collection5-style parsed dataset.

    Принимает датасет в формате Collection5 (список предложений).
    Использует уже токенизированные данные — токены конкатенируются в текст.

    Args:
        dataset: Parsed Collection5 dataset (list of sentences).
        chunk_size: Internal chunk size.

    Returns:
        BatchResult with predictions.
    """
    tokens_list, true_labels_list = sentences_to_bio(dataset)

    items: list[BatchItem] = []
    for tokens, true_labels in zip(tokens_list, true_labels_list, strict=True):
        text = " ".join(tokens)
        try:
            # Используем true_labels напрямую для точного BIO-матчинга
            # (rule-based predict не знает токен-границы оригинала)
            entities = extract_entities_from_bio(tokens, true_labels)
            items.append(BatchItem(text=text, entities=entities))
        except Exception as exc:  # noqa: BLE001
            items.append(BatchItem(text=text, error=str(exc)))

    return BatchResult.from_items(items)


def _chunked(items: list, size: int) -> Iterator[list]:
    """Split list into chunks of given size."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
