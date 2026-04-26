"""
Структуры данных контрактов схем / Data contract schema structures.

Вдохновлено Confluent Schema Registry и Data Contract CLI, но без внешних
зависимостей — работает везде, включая CI без Kafka/Avro.

Inspired by Confluent Schema Registry and Data Contract CLI, but dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ColumnType(str, Enum):
    """Поддерживаемые типы столбцов / Supported column types."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class Compatibility(str, Enum):
    """
    Режим совместимости схем (Confluent-совместимая семантика).
    Schema compatibility mode (Confluent-compatible semantics).

    BACKWARD: новая схема читает данные, записанные старой.
              New schema can read data written by old schema.
    FORWARD:  старая схема читает данные, записанные новой.
              Old schema can read data written by new schema.
    FULL:     оба направления / Both directions.
    NONE:     проверка отключена / Compatibility checks disabled.
    """

    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"
    FULL = "FULL"
    NONE = "NONE"


@dataclass
class ColumnSchema:
    """Описание одного столбца / Single column descriptor."""

    name: str
    dtype: ColumnType
    nullable: bool = True
    description: str = ""
    allowed_values: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "nullable": self.nullable,
            "description": self.description,
            "allowed_values": self.allowed_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }


@dataclass
class DataSchema:
    """Полная схема датасета / Full dataset schema."""

    name: str
    columns: list[ColumnSchema]
    compatibility: Compatibility = Compatibility.BACKWARD
    description: str = ""

    def column_map(self) -> dict[str, ColumnSchema]:
        """Индекс столбцов по имени / Column index by name."""
        return {c.name: c for c in self.columns}


@dataclass
class SchemaVersion:
    """Версионированная запись схемы в реестре / Versioned schema entry in registry."""

    schema_name: str
    version: str  # семантическая версия / semantic: "1.0.0"
    schema: DataSchema
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_latest: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "version": self.version,
            "registered_at": self.registered_at,
            "is_latest": self.is_latest,
            "compatibility": self.schema.compatibility.value,
            "description": self.schema.description,
            "columns": [c.to_dict() for c in self.schema.columns],
        }
