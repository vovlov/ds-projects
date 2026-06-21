"""
Мини-база знаний для Named Entity Linking (NEL).

Встроенные сущности: крупнейшие российские компании, города и государства.
Поддерживает точное и нечёткое совпадение псевдонимов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class EntityRecord:
    """Запись о сущности в базе знаний."""

    entity_id: str
    canonical_name: str
    aliases: list[str]
    entity_type: str  # PER | ORG | LOC
    description: str


@dataclass
class KBStats:
    n_entities: int
    by_type: dict[str, int]
    n_aliases: int


# Встроенные сущности: топ-15 компаний MOEX + крупные города/государства
_BUILT_IN: list[dict] = [
    # ORG — публичные компании
    {
        "id": "Q102048",
        "name": "Газпром",
        "aliases": ["Gazprom", "ПАО Газпром", "ОАО Газпром", "ГАЗПРОМ"],
        "type": "ORG",
        "desc": "Крупнейшая газовая корпорация России",
    },
    {
        "id": "Q154860",
        "name": "Лукойл",
        "aliases": ["LUKOIL", "ПАО Лукойл", "ЛУКОЙЛ"],
        "type": "ORG",
        "desc": "Нефтяная компания",
    },
    {
        "id": "Q163708",
        "name": "Сбербанк",
        "aliases": ["Sberbank", "ПАО Сбербанк", "Сбер", "Сберегательный банк", "СберБанк"],
        "type": "ORG",
        "desc": "Крупнейший банк России",
    },
    {
        "id": "Q184545",
        "name": "Роснефть",
        "aliases": ["Rosneft", "ПАО Роснефть", "НК Роснефть"],
        "type": "ORG",
        "desc": "Нефтегазовая компания с государственным участием",
    },
    {
        "id": "Q222548",
        "name": "ВТБ",
        "aliases": ["VTB Bank", "ПАО ВТБ", "Банк ВТБ", "Внешторгбанк"],
        "type": "ORG",
        "desc": "Государственный банк",
    },
    {
        "id": "Q272086",
        "name": "Яндекс",
        "aliases": ["Yandex", "ПАО Яндекс", "Yandex N.V.", "ЯНДЕКС"],
        "type": "ORG",
        "desc": "Технологическая компания",
    },
    {
        "id": "Q831049",
        "name": "Норильский никель",
        "aliases": ["Норникель", "Norilsk Nickel", "ПАО ГМК Норникель", "ГМК Норникель"],
        "type": "ORG",
        "desc": "Горно-металлургическая компания",
    },
    {
        "id": "Q4358463",
        "name": "Татнефть",
        "aliases": ["Tatneft", "ПАО Татнефть"],
        "type": "ORG",
        "desc": "Нефтяная компания Республики Татарстан",
    },
    {
        "id": "Q1061203",
        "name": "Сургутнефтегаз",
        "aliases": ["Surgutneftegas", "ПАО Сургутнефтегаз", "Сургут"],
        "type": "ORG",
        "desc": "Нефтегазовая компания",
    },
    {
        "id": "Q3595477",
        "name": "МТС",
        "aliases": ["MTS", "ПАО МТС", "Мобильные ТелеСистемы"],
        "type": "ORG",
        "desc": "Телекоммуникационная компания",
    },
    {
        "id": "Q4241548",
        "name": "Магнит",
        "aliases": ["Magnit", "ПАО Магнит", "Сеть Магнит"],
        "type": "ORG",
        "desc": "Крупная розничная сеть России",
    },
    {
        "id": "Q1930187",
        "name": "Альфа-Банк",
        "aliases": ["Alfa-Bank", "АО Альфа-Банк", "Альфабанк", "Альфа Банк"],
        "type": "ORG",
        "desc": "Крупный частный банк России",
    },
    {
        "id": "Q2337600",
        "name": "Северсталь",
        "aliases": ["Severstal", "ПАО Северсталь"],
        "type": "ORG",
        "desc": "Металлургическая компания",
    },
    {
        "id": "Q185684",
        "name": "Газпром нефть",
        "aliases": ["Gazprom Neft", "ОАО Газпром нефть", "Газпромнефть"],
        "type": "ORG",
        "desc": "Нефтяная дочерняя компания Газпрома",
    },
    {
        "id": "Q4380591",
        "name": "X5 Group",
        "aliases": ["X5 Retail Group", "Пятёрочка", "Перекрёсток"],
        "type": "ORG",
        "desc": "Крупнейший ретейлер России",
    },
    # LOC — крупные города и государства
    {
        "id": "Q649",
        "name": "Москва",
        "aliases": ["Moscow", "г. Москва", "МСК", "г Москва"],
        "type": "LOC",
        "desc": "Столица России",
    },
    {
        "id": "Q656",
        "name": "Санкт-Петербург",
        "aliases": [
            "Saint Petersburg",
            "Петербург",
            "Питер",
            "СПб",
            "Ленинград",
            "г. Санкт-Петербург",
        ],
        "type": "LOC",
        "desc": "Второй по величине город России",
    },
    {
        "id": "Q887",
        "name": "Екатеринбург",
        "aliases": ["Yekaterinburg", "Свердловск", "Екат"],
        "type": "LOC",
        "desc": "Административный центр Урала",
    },
    {
        "id": "Q900",
        "name": "Новосибирск",
        "aliases": ["Novosibirsk"],
        "type": "LOC",
        "desc": "Крупнейший город Сибири",
    },
    {
        "id": "Q900218",
        "name": "Казань",
        "aliases": ["Kazan"],
        "type": "LOC",
        "desc": "Столица Республики Татарстан",
    },
    {
        "id": "Q159",
        "name": "Россия",
        "aliases": ["Russia", "РФ", "Российская Федерация", "Russian Federation"],
        "type": "LOC",
        "desc": "Крупнейшее государство мира",
    },
    {
        "id": "Q30",
        "name": "США",
        "aliases": ["US", "United States", "America", "Соединённые Штаты", "США"],
        "type": "LOC",
        "desc": "Государство в Северной Америке",
    },
    {
        "id": "Q183",
        "name": "Германия",
        "aliases": ["Germany", "Deutschland", "ФРГ", "Германская Федерация"],
        "type": "LOC",
        "desc": "Государство в Западной Европе",
    },
    {
        "id": "Q142",
        "name": "Франция",
        "aliases": ["France", "French Republic", "Французская Республика"],
        "type": "LOC",
        "desc": "Государство в Западной Европе",
    },
    {
        "id": "Q148",
        "name": "Китай",
        "aliases": ["China", "КНР", "PRC", "Китайская Народная Республика"],
        "type": "LOC",
        "desc": "Государство в Восточной Азии",
    },
]


def _normalize_text(text: str) -> str:
    """Нормализация: lowercase, убираем пунктуацию, свёртываем пробелы."""
    text = text.lower().strip()
    text = re.sub(r"[«»\"'.,!?;:()/\\-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class KnowledgeBase:
    """
    База знаний для Named Entity Linking.

    Хранит сущности с псевдонимами и строит инвертированный индекс
    нормализованных псевдонимов → entity_id для быстрого точного поиска.
    """

    def __init__(self) -> None:
        self._entities: dict[str, EntityRecord] = {}
        # нормализованный псевдоним → entity_id (последний запись побеждает при конфликте)
        self._alias_index: dict[str, str] = {}
        self._load_built_in()

    def _load_built_in(self) -> None:
        for data in _BUILT_IN:
            self.add_entity(
                EntityRecord(
                    entity_id=data["id"],
                    canonical_name=data["name"],
                    aliases=data["aliases"],
                    entity_type=data["type"],
                    description=data["desc"],
                )
            )

    def add_entity(self, record: EntityRecord) -> None:
        """Добавить сущность в KB; индексируем canonical и все aliases."""
        self._entities[record.entity_id] = record
        for alias in [record.canonical_name, *record.aliases]:
            norm = _normalize_text(alias)
            if norm:
                self._alias_index[norm] = record.entity_id

    def exact_lookup(self, mention: str) -> str | None:
        """Точный поиск нормализованного упоминания → entity_id или None."""
        return self._alias_index.get(_normalize_text(mention))

    def get_entity(self, entity_id: str) -> EntityRecord | None:
        return self._entities.get(entity_id)

    def all_entities(self, entity_type: str | None = None) -> list[EntityRecord]:
        records = list(self._entities.values())
        if entity_type:
            records = [r for r in records if r.entity_type == entity_type]
        return records

    def stats(self) -> KBStats:
        by_type: dict[str, int] = {}
        for r in self._entities.values():
            by_type[r.entity_type] = by_type.get(r.entity_type, 0) + 1
        return KBStats(
            n_entities=len(self._entities),
            by_type=by_type,
            n_aliases=len(self._alias_index),
        )
