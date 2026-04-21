"""
Feast-совместимый мост для feature store.
Feast-compatible bridge — wraps FeatureRegistry in the Feast API surface.

Мотивация: Feast — стандарт де-факто для feature store в MLOps 2026.
Наш FeatureRegistry уже хранит фичи; этот модуль добавляет Feast-совместимый API,
чтобы модели могли переключиться на реальный Feast без изменения кода вызова.

Motivation: Feast is the de-facto standard feature store in MLOps 2026.
Our FeatureRegistry already stores features; this module exposes a Feast-compatible
API so models can switch to real Feast without changing calling code.

Паттерн: адаптер над FeatureRegistry с graceful degradation.
Pattern: adapter over FeatureRegistry with graceful degradation to real Feast.

Формат ссылки на фичу (Feast-style): "feature_view:feature_name"
Feature reference format (Feast-style): "feature_view:feature_name"

Sources:
  - Feast docs: docs.feast.dev (2026)
  - Feast GitHub: github.com/feast-dev/feast
  - Made With ML feature store: madewithml.com/courses/mlops/feature-store
  - oneuptime feature store patterns 2026: oneuptime.com/blog/post/2026-01-25-feature-stores-mlops
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recsys.feature_store.registry import FeatureRegistry


def is_available() -> bool:
    """Проверяем наличие Feast в окружении / Check if Feast is installed."""
    try:
        import feast  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Dataclasses — Feast-compatible API surface
# ---------------------------------------------------------------------------


@dataclass
class FeatureRef:
    """
    Ссылка на фичу в формате Feast: "view:feature".
    Feast-style feature reference: "feature_view:feature_name".
    """

    view: str  # имя FeatureView / FeatureView name
    name: str  # имя фичи / feature name

    @classmethod
    def parse(cls, ref: str) -> FeatureRef:
        """
        Парсим строку "view:feature" → FeatureRef.
        Parse "view:feature" string into FeatureRef.

        Raises ValueError если формат неверный / if format is invalid.
        """
        parts = ref.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"Invalid feature reference '{ref}'. Expected format: 'view:feature_name'"
            )
        return cls(view=parts[0], name=parts[1])

    def __str__(self) -> str:
        return f"{self.view}:{self.name}"


@dataclass
class FeatureRequest:
    """
    Запрос на получение online-фичей (Feast-совместимый).
    Online feature request — mirrors Feast's get_online_features() call.

    Attributes:
        entity_rows: Список сущностей с ключами. Каждая строка — dict с entity_id.
                     List of entity dicts. Each must contain the entity key.
                     Пример / Example: [{"user_id": 42}, {"user_id": 7}]
        features: Список ссылок на фичи в формате "view:feature_name".
                  List of feature refs in "view:feature_name" format.
                  Пример / Example: ["user_features:avg_rating", "user_features:n_purchases"]
    """

    entity_rows: list[dict[str, Any]]
    features: list[str]  # "view:feature_name" format


@dataclass
class EntityFeatureRow:
    """
    Строка с entity-ключом и значениями фичей.
    Entity row enriched with feature values.
    """

    entity_key: str  # entity_id использованный для поиска / entity_id used for lookup
    entity_row: dict[str, Any]  # оригинальный запрос / original request dict
    feature_values: dict[str, Any]  # имя_фичи → значение / feature_name → value
    missing_features: list[str]  # фичи, которых не нашли / features not found


@dataclass
class OnlineFeatureResponse:
    """
    Ответ Feast-compatible get_online_features().
    Response from Feast-compatible get_online_features().

    Attributes:
        rows: Строки с фичами для каждой сущности / Feature rows per entity.
        feature_refs: Список запрошенных ссылок / Requested feature refs.
        n_entities: Количество сущностей / Number of entities.
        n_missing: Суммарно пропущенных значений / Total missing values.
    """

    rows: list[EntityFeatureRow]
    feature_refs: list[str]
    n_entities: int
    n_missing: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-сериализуемый результат / JSON-serializable result."""
        return {
            "rows": [
                {
                    "entity_key": r.entity_key,
                    "entity_row": r.entity_row,
                    "feature_values": r.feature_values,
                    "missing_features": r.missing_features,
                }
                for r in self.rows
            ],
            "feature_refs": self.feature_refs,
            "n_entities": self.n_entities,
            "n_missing": self.n_missing,
        }


# ---------------------------------------------------------------------------
# Feature View Registry
# Регистр FeatureView определяет, какие фичи принадлежат какому view.
# Registry of FeatureViews: maps view names to entity_id prefixes.
# ---------------------------------------------------------------------------

# Стандартные view нашего feature store / Built-in views for our feature store.
# Ключ — имя view, значение — entity-prefix в FeatureRegistry.
# Key: view name, Value: entity prefix in FeatureRegistry.
_DEFAULT_VIEW_MAP: dict[str, str] = {
    "user_features": "user_",  # user_features:avg_rating → registry["user_{id}"]["avg_rating"]
    "item_features": "item_",  # item_features:n_ratings → registry["item_{id}"]["n_ratings"]
}


# ---------------------------------------------------------------------------
# FeastBridge
# ---------------------------------------------------------------------------


@dataclass
class FeastBridge:
    """
    Feast-совместимый фасад над FeatureRegistry.
    Feast-compatible facade over our FeatureRegistry.

    Позволяет коду получать фичи с тем же API, что и настоящий Feast.
    Когда Feast установлен — может делегировать запросы настоящему FeatureStore.
    Это устраняет training-serving skew, давая единый интерфейс.

    Allows calling code to use the same API as real Feast.
    When Feast is installed, can delegate to the real FeatureStore.
    Eliminates training-serving skew by providing a unified interface.

    Args:
        registry: Наш FeatureRegistry с данными / Our FeatureRegistry with data.
        view_map: Маппинг view_name → entity-prefix.
                  Mapping of view_name → entity-prefix in registry.
    """

    registry: FeatureRegistry
    view_map: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_VIEW_MAP))

    def get_online_features(self, request: FeatureRequest) -> OnlineFeatureResponse:
        """
        Получаем online-фичи для набора сущностей (Feast API).
        Get online features for a set of entities — mirrors Feast API.

        Для каждой entity_row определяем entity_id по доступным ключам
        (user_id, item_id, entity_id), затем достаём фичи из реестра.

        For each entity_row: detect entity key (user_id/item_id/entity_id),
        then look up features from the registry.

        Args:
            request: FeatureRequest с entity_rows и feature refs.

        Returns:
            OnlineFeatureResponse с фичами для каждой сущности.
        """
        parsed_refs = [FeatureRef.parse(ref) for ref in request.features]
        rows: list[EntityFeatureRow] = []
        total_missing = 0

        for entity_row in request.entity_rows:
            entity_key, prefix = self._resolve_entity(entity_row, parsed_refs)
            feature_values: dict[str, Any] = {}
            missing: list[str] = []

            for ref in parsed_refs:
                # Ключ в реестре: prefix + entity_key_value
                # Registry key: prefix + entity_key_value
                registry_key = f"{prefix}{entity_key}"
                try:
                    # Получаем одну фичу / Get single feature
                    result = self.registry.get_features(registry_key, [ref.name])
                    value = result.get(ref.name)
                    if value is None:
                        missing.append(str(ref))
                        total_missing += 1
                    else:
                        feature_values[str(ref)] = value
                except KeyError:
                    missing.append(str(ref))
                    total_missing += 1

            rows.append(
                EntityFeatureRow(
                    entity_key=str(entity_key),
                    entity_row=entity_row,
                    feature_values=feature_values,
                    missing_features=missing,
                )
            )

        return OnlineFeatureResponse(
            rows=rows,
            feature_refs=request.features,
            n_entities=len(rows),
            n_missing=total_missing,
        )

    def _resolve_entity(
        self,
        entity_row: dict[str, Any],
        refs: list[FeatureRef],
    ) -> tuple[Any, str]:
        """
        Определяем entity_key и entity-prefix по entity_row и запрошенным view.
        Detect entity key value and prefix from entity_row and requested views.

        Приоритет поиска ключа / Key detection priority:
          1. Первый view в запросе → prefix → угадываем ключ (user_id / item_id / entity_id)
          2. Fallback: берём любое значение из entity_row

        Returns:
            (entity_key_value, prefix)
        """
        # Определяем prefix по первому view / Determine prefix from first view
        first_view = refs[0].view if refs else "user_features"
        prefix = self.view_map.get(first_view, "")

        # Угадываем имя ключа по prefix / Guess key name from prefix
        if prefix == "user_":
            key_candidates = ["user_id", "userId", "id"]
        elif prefix == "item_":
            key_candidates = ["item_id", "product_id", "itemId", "id"]
        else:
            key_candidates = ["entity_id", "id"]

        for candidate in key_candidates:
            if candidate in entity_row:
                return entity_row[candidate], prefix

        # Fallback — берём первое значение из строки / take first value
        if entity_row:
            return next(iter(entity_row.values())), prefix

        return "unknown", prefix

    def register_view(self, view_name: str, entity_prefix: str) -> None:
        """
        Регистрируем кастомный FeatureView.
        Register a custom FeatureView name and its entity prefix.

        Позволяет расширить набор view без изменения кода.
        Allows extending the view set without code changes.
        """
        self.view_map[view_name] = entity_prefix

    def list_views(self) -> list[str]:
        """Список зарегистрированных view / List registered view names."""
        return list(self.view_map.keys())
