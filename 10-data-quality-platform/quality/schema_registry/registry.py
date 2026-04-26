"""
Реестр схем с версионированием и проверкой совместимости.
Schema registry with versioning and compatibility enforcement.

Аналог Confluent Schema Registry, но без Kafka и внешних зависимостей.
Confluent Schema Registry analogue — no Kafka, no external deps.
"""

from __future__ import annotations

from typing import Any

from .schema import ColumnType, Compatibility, DataSchema, SchemaVersion


def _bump_version(current: str, breaking: bool) -> str:
    """
    Автоматически вычислить следующую семантическую версию.
    Breaking change → major bump; non-breaking → minor bump.
    """
    major, minor, patch = (int(x) for x in current.split("."))
    if breaking:
        return f"{major + 1}.0.0"
    return f"{major}.{minor + 1}.0"


class SchemaRegistry:
    """
    In-memory реестр схем с версионированием и проверкой совместимости.

    Поддерживает:
    - Семантическое версионирование (MAJOR.MINOR.PATCH)
    - Режимы совместимости: BACKWARD / FORWARD / FULL / NONE
    - Автодетекцию breaking changes (удалённые столбцы, смена типа, nullable→NOT NULL)
    - Безопасное расширение типов: integer → float не является breaking

    Supports:
    - Semantic versioning (MAJOR.MINOR.PATCH)
    - Compatibility modes: BACKWARD / FORWARD / FULL / NONE
    - Auto-detection of breaking changes
    - Safe type widening: integer → float is non-breaking
    """

    def __init__(self) -> None:
        # schema_name → список версий в порядке регистрации
        self._versions: dict[str, list[SchemaVersion]] = {}

    def register(
        self,
        schema: DataSchema,
        version: str | None = None,
        allow_breaking: bool = False,
    ) -> SchemaVersion:
        """
        Зарегистрировать новую схему или новую версию существующей.
        Register a new schema or a new version of an existing schema.

        Если version=None — автоматический bump:
          breaking change → major, non-breaking → minor.
        allow_breaking=True обходит проверку совместимости (NONE mode override).

        Raises:
            ValueError: при нарушении совместимости без allow_breaking.
        """
        name = schema.name

        if name not in self._versions:
            sv = SchemaVersion(
                schema_name=name,
                version=version or "1.0.0",
                schema=schema,
            )
            self._versions[name] = [sv]
            return sv

        latest = self._get_latest(name)
        breaking = self._detect_breaking(latest.schema, schema)

        if breaking and not allow_breaking and schema.compatibility != Compatibility.NONE:
            raise ValueError(
                f"Breaking change в схеме '{name}' / Breaking change in schema '{name}': "
                + "; ".join(breaking)
            )

        if version is None:
            version = _bump_version(latest.version, bool(breaking))

        latest.is_latest = False
        sv = SchemaVersion(schema_name=name, version=version, schema=schema)
        self._versions[name].append(sv)
        return sv

    def get(self, name: str, version: str | None = None) -> SchemaVersion | None:
        """
        Получить версию схемы / Get a schema version.
        version=None → последняя версия / latest version.
        """
        versions = self._versions.get(name)
        if not versions:
            return None
        if version is None:
            return self._get_latest(name)
        for sv in versions:
            if sv.version == version:
                return sv
        return None

    def list_versions(self, name: str) -> list[str]:
        """Список всех версий схемы / List all registered version strings."""
        return [sv.version for sv in self._versions.get(name, [])]

    def list_schemas(self) -> list[str]:
        """Имена всех зарегистрированных схем / Names of all registered schemas."""
        return list(self._versions.keys())

    def check_compatibility(self, name: str, candidate: DataSchema) -> dict[str, Any]:
        """
        Проверить, совместим ли кандидат с последней зарегистрированной версией.
        Check if candidate schema is compatible with the latest registered version.
        """
        latest = self.get(name)
        if latest is None:
            return {
                "compatible": True,
                "reason": "no_existing_schema",
                "breaking_changes": [],
                "current_version": None,
            }

        breaking = self._detect_breaking(latest.schema, candidate)
        return {
            "compatible": not bool(breaking),
            "reason": "ok" if not breaking else "breaking_changes_detected",
            "breaking_changes": breaking,
            "current_version": latest.version,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_latest(self, name: str) -> SchemaVersion:
        return self._versions[name][-1]

    def _detect_breaking(self, old: DataSchema, new: DataSchema) -> list[str]:
        """
        Найти несовместимые изменения для BACKWARD-совместимости.
        Detect breaking changes for BACKWARD compatibility.

        BACKWARD = новая схема должна уметь читать данные, записанные старой.
        Breaking:
          - Удалённый столбец (old data has it, new schema drops it)
          - Смена типа на несовместимый (integer→float — safe widening, не breaking)
          - nullable=True → nullable=False (старые данные могут содержать null)
          - Новый NOT NULL столбец (старые данные не имеют значения)
        Non-breaking:
          - Новый nullable столбец
          - Ослабление ограничений (NOT NULL → nullable)
        """
        old_cols = old.column_map()
        new_cols = new.column_map()
        issues: list[str] = []

        for col_name in old_cols:
            if col_name not in new_cols:
                issues.append(f"Column removed: '{col_name}'")

        for col_name, new_col in new_cols.items():
            if col_name in old_cols:
                old_col = old_cols[col_name]
                if old_col.dtype != new_col.dtype:
                    # integer→float — безопасное расширение типа / safe widening
                    safe = old_col.dtype == ColumnType.INTEGER and new_col.dtype == ColumnType.FLOAT
                    if not safe:
                        issues.append(
                            f"Type changed: '{col_name}' "
                            f"{old_col.dtype.value}→{new_col.dtype.value}"
                        )
                if old_col.nullable and not new_col.nullable:
                    issues.append(f"Nullable→NOT NULL: '{col_name}'")
            else:
                # Новый столбец: NOT NULL без default — breaking для старых данных
                if not new_col.nullable:
                    issues.append(f"New required column: '{col_name}'")

        return issues


# ---------------------------------------------------------------------------
# Singleton для API / API singleton
# ---------------------------------------------------------------------------

_registry = SchemaRegistry()


def get_registry() -> SchemaRegistry:
    """Вернуть глобальный экземпляр реестра / Return global registry instance."""
    return _registry
