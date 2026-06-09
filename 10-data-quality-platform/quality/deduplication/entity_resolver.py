"""
Entity Resolution / Record Deduplication module.

Обнаружение дубликатов записей через блокировку + попарное сравнение.
Supports blocking strategies and field-level similarity scoring.

Алгоритм:
  1. Blocking — разбиение по значениям блокирующих ключей → O(n²) → O(n·b)
  2. Pairwise comparison — Jaccard для строк, допуск для чисел, exact для категорий
  3. Weighted aggregation → record-level similarity score
  4. Threshold filtering → list[RecordPair]

Источники / Sources:
  Christen 2012 "Data Matching" (Springer) — blocking strategies, Jaccard similarity
  Köpcke & Rahm 2010 VLDB J "Frameworks for entity matching" — field-level sim aggregation
  Fellegi & Sunter 1969 JASA "A theory for record linkage" — probabilistic foundations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses / Конфигурация
# ---------------------------------------------------------------------------


@dataclass
class FieldConfig:
    """Конфигурация поля для сравнения / Field comparison configuration."""

    name: str
    weight: float = 1.0
    similarity_type: str = "jaccard"  # "jaccard" | "exact" | "numeric"
    numeric_tolerance: float = 0.1  # относительный допуск для numeric

    def __post_init__(self) -> None:
        if self.similarity_type not in {"jaccard", "exact", "numeric"}:
            raise ValueError(
                f"Unknown similarity_type '{self.similarity_type}'. "
                "Use 'jaccard', 'exact', or 'numeric'."
            )
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass
class BlockingConfig:
    """Конфигурация блокировки / Blocking strategy configuration."""

    blocking_keys: list[str]
    threshold: float = 0.8
    max_comparisons: int = 100_000

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if self.max_comparisons < 1:
            raise ValueError("max_comparisons must be >= 1")


@dataclass
class RecordPair:
    """Пара дублирующихся записей / Duplicate record pair."""

    id1: Any
    id2: Any
    similarity: float
    field_similarities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id1": self.id1,
            "id2": self.id2,
            "similarity": round(self.similarity, 4),
            "field_similarities": {k: round(v, 4) for k, v in self.field_similarities.items()},
        }


@dataclass
class DeduplicationResult:
    """Результат дедупликации / Deduplication result."""

    pairs: list[RecordPair]
    total_comparisons: int
    blocks_count: int
    threshold: float
    records_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duplicate_pairs": [p.to_dict() for p in self.pairs],
            "summary": {
                "pairs_found": len(self.pairs),
                "total_comparisons": self.total_comparisons,
                "blocks_count": self.blocks_count,
                "threshold": self.threshold,
                "records_count": self.records_count,
                "deduplication_ratio": round(len(self.pairs) / max(self.records_count, 1), 4),
            },
        }


# ---------------------------------------------------------------------------
# Функции сходства / Similarity functions
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """3-gram tokenization для устойчивости к опечаткам / typo-robust 3-grams."""
    s = re.sub(r"\s+", " ", str(text).lower().strip())
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def jaccard_similarity(a: Any, b: Any) -> float:
    """Jaccard на 3-граммах / Jaccard similarity over character 3-grams.

    Устойчив к перестановкам токенов и опечаткам в ~1-2 символа.
    Robust to token reordering and ~1-2 character typos.
    """
    sa, sb = _tokenize(str(a)), _tokenize(str(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def exact_similarity(a: Any, b: Any) -> float:
    """Точное совпадение после нормализации / Exact match after normalization."""
    return 1.0 if str(a).lower().strip() == str(b).lower().strip() else 0.0


def numeric_similarity(a: Any, b: Any, tolerance: float = 0.1) -> float:
    """Относительная близость чисел / Relative numeric closeness.

    Возвращает 1.0 если |a-b|/max(|a|,|b|,eps) <= tolerance, иначе плавно убывает.
    """
    try:
        fa, fb = float(a), float(b)
    except (ValueError, TypeError):
        return 0.0
    denom = max(abs(fa), abs(fb), 1e-9)
    rel_diff = abs(fa - fb) / denom
    if rel_diff <= tolerance:
        return 1.0
    # Плавный спад: linear decay до 0 при rel_diff = 5*tolerance
    decay_end = 5 * tolerance
    if rel_diff >= decay_end:
        return 0.0
    return 1.0 - (rel_diff - tolerance) / (decay_end - tolerance)


# ---------------------------------------------------------------------------
# Основной класс / Core class
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolver для обнаружения дублирующихся записей.

    Implements blocking + pairwise field comparison + weighted aggregation.
    Реализует блокировку + попарное сравнение полей + взвешенную агрегацию.
    """

    def _blocking_key(self, record: dict[str, Any], keys: list[str]) -> str:
        """Строит ключ блока из нескольких полей / Build block key from multiple fields."""
        parts = []
        for k in keys:
            val = str(record.get(k, "")).lower().strip()
            # Берём первые 3 символа для мягкой блокировки (prefix blocking)
            # First 3 chars for soft prefix blocking — tolerates minor typos
            parts.append(val[:3] if val else "__missing__")
        return "|".join(parts)

    def _build_blocks(
        self,
        records: list[dict[str, Any]],
        id_field: str,
        blocking_keys: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Разбиваем записи на блоки по блокирующим ключам / Partition into blocks."""
        blocks: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            key = self._blocking_key(rec, blocking_keys)
            blocks.setdefault(key, []).append(rec)
        return blocks

    def _compare_records(
        self,
        r1: dict[str, Any],
        r2: dict[str, Any],
        field_configs: list[FieldConfig],
    ) -> tuple[float, dict[str, float]]:
        """Взвешенное попарное сравнение полей / Weighted pairwise field comparison."""
        total_weight = 0.0
        weighted_sum = 0.0
        field_sims: dict[str, float] = {}

        for fc in field_configs:
            v1 = r1.get(fc.name)
            v2 = r2.get(fc.name)

            # Оба отсутствуют → не вносим вклад в score
            if v1 is None and v2 is None:
                continue
            # Один отсутствует → нулевое сходство
            if v1 is None or v2 is None:
                sim = 0.0
            elif fc.similarity_type == "jaccard":
                sim = jaccard_similarity(v1, v2)
            elif fc.similarity_type == "exact":
                sim = exact_similarity(v1, v2)
            else:  # numeric
                sim = numeric_similarity(v1, v2, fc.numeric_tolerance)

            field_sims[fc.name] = sim
            weighted_sum += fc.weight * sim
            total_weight += fc.weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        return overall, field_sims

    def resolve(
        self,
        records: list[dict[str, Any]],
        id_field: str,
        field_configs: list[FieldConfig],
        blocking_config: BlockingConfig,
    ) -> DeduplicationResult:
        """Найти дублирующиеся записи / Find duplicate records.

        Args:
            records: Список записей (словарей) / List of record dicts.
            id_field: Поле-идентификатор / Identifier field name.
            field_configs: Конфигурация сравниваемых полей / Field comparison configs.
            blocking_config: Стратегия блокировки / Blocking strategy.

        Returns:
            DeduplicationResult с парами дубликатов / with duplicate pairs.
        """
        if not records:
            return DeduplicationResult(
                pairs=[],
                total_comparisons=0,
                blocks_count=0,
                threshold=blocking_config.threshold,
                records_count=0,
            )

        blocks = self._build_blocks(records, id_field, blocking_config.blocking_keys)

        pairs: list[RecordPair] = []
        total_comparisons = 0
        comparison_limit_reached = False

        for block_records in blocks.values():
            if len(block_records) < 2:
                continue
            for i in range(len(block_records)):
                for j in range(i + 1, len(block_records)):
                    if total_comparisons >= blocking_config.max_comparisons:
                        comparison_limit_reached = True
                        break
                    total_comparisons += 1
                    r1, r2 = block_records[i], block_records[j]
                    sim, field_sims = self._compare_records(r1, r2, field_configs)
                    if sim >= blocking_config.threshold:
                        pairs.append(
                            RecordPair(
                                id1=r1.get(id_field),
                                id2=r2.get(id_field),
                                similarity=sim,
                                field_similarities=field_sims,
                            )
                        )
                if comparison_limit_reached:
                    break

        # Дедупликация пар (id1, id2) — порядок нормализован / Deduplicate pairs
        seen: set[tuple[Any, Any]] = set()
        unique_pairs: list[RecordPair] = []
        for p in pairs:
            key = (min(str(p.id1), str(p.id2)), max(str(p.id1), str(p.id2)))
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)

        return DeduplicationResult(
            pairs=unique_pairs,
            total_comparisons=total_comparisons,
            blocks_count=len(blocks),
            threshold=blocking_config.threshold,
            records_count=len(records),
        )
