"""
Реестр фичей — простой feature store.
Simple feature registry for managing feature definitions and values.

Хранит метаданные фичей (имя, тип, описание) и значения по entity_id.
Минимальная реализация без внешних зависимостей вроде Feast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureDefinition:
    """Определение фичи с метаданными.
    Feature definition with schema metadata."""

    name: str
    dtype: str  # "int", "float", "str", "bool"
    description: str
    entity_type: str = "user"  # "user" или "item"


class FeatureRegistry:
    """Реестр фичей: регистрация, хранение, выдача.
    Feature registry: define features, store values, retrieve by entity."""

    def __init__(self) -> None:
        # Определения фичей / Feature definitions
        self._definitions: dict[str, FeatureDefinition] = {}
        # Значения: entity_id -> {feature_name: value}
        self._store: dict[str, dict[str, Any]] = {}

    def register_feature(
        self,
        name: str,
        dtype: str,
        description: str,
        entity_type: str = "user",
    ) -> None:
        """Регистрируем новую фичу в реестре.
        Register a feature definition in the registry."""
        if name in self._definitions:
            raise ValueError(
                f"Feature '{name}' already registered / Фича '{name}' уже зарегистрирована"
            )
        self._definitions[name] = FeatureDefinition(
            name=name,
            dtype=dtype,
            description=description,
            entity_type=entity_type,
        )

    def list_features(self, entity_type: str | None = None) -> list[FeatureDefinition]:
        """Список всех зарегистрированных фичей (с опциональным фильтром).
        List all registered features, optionally filtered by entity type."""
        if entity_type is None:
            return list(self._definitions.values())
        return [fd for fd in self._definitions.values() if fd.entity_type == entity_type]

    def set_features(self, entity_id: str, features: dict[str, Any]) -> None:
        """Сохраняем значения фичей для конкретной сущности.
        Store feature values for a specific entity."""
        # Проверяем что фичи зарегистрированы
        for name in features:
            if name not in self._definitions:
                raise KeyError(
                    f"Feature '{name}' not registered / Фича '{name}' не зарегистрирована"
                )

        if entity_id not in self._store:
            self._store[entity_id] = {}
        self._store[entity_id].update(features)

    def get_features(self, entity_id: str, feature_names: list[str]) -> dict[str, Any]:
        """Получаем значения фичей для сущности.
        Retrieve feature values for a given entity. Missing features return None."""
        result: dict[str, Any] = {}
        entity_data = self._store.get(entity_id, {})

        for name in feature_names:
            if name not in self._definitions:
                raise KeyError(
                    f"Feature '{name}' not registered / Фича '{name}' не зарегистрирована"
                )
            result[name] = entity_data.get(name)

        return result

    def has_entity(self, entity_id: str) -> bool:
        """Проверяем, есть ли данные для сущности.
        Check if an entity has any stored features."""
        return entity_id in self._store

    @property
    def n_features(self) -> int:
        """Количество зарегистрированных фичей / Number of registered features."""
        return len(self._definitions)

    @property
    def n_entities(self) -> int:
        """Количество сущностей с данными / Number of entities with stored data."""
        return len(self._store)
