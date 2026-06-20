"""Named Entity Linking: сопоставление извлечённых сущностей с базой знаний.

Named Entity Linking (NEL) — следующий шаг после NER: каждая сущность
получает уникальный ID и каноническое имя (WikiData/Freebase-стиль).

Для юридического NER это критично: «Газпром», «ПАО Газпром», «Газпром нефть»
могут означать разные юридические лица с разными реквизитами.

Алгоритм (без внешних зависимостей, CI-совместим):
1. Фильтрация кандидатов по типу сущности (PER/ORG/LOC/COURT)
2. Скоринг: exact match → alias match → prefix match → trigram Jaccard
3. Порог confidence (дефолт 0.45) отделяет «linked» от «unlinked»
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Knowledge Base entries ────────────────────────────────────────────────────


@dataclass
class KBEntry:
    """Запись в базе знаний."""

    entity_id: str
    canonical_name: str
    entity_type: str  # PER | ORG | LOC | COURT
    aliases: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EntityLink:
    """Результат связывания сущности с записью в базе знаний."""

    entity_id: str
    canonical_name: str
    entity_type: str
    confidence: float
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "confidence": round(self.confidence, 4),
            "description": self.description,
        }


@dataclass
class LinkedEntity:
    """Именованная сущность с результатом NEL."""

    text: str
    label: str
    start: int
    end: int
    link: EntityLink | None  # None = не удалось связать (NIL entity)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "link": self.link.to_dict() if self.link else None,
        }


# ── Built-in knowledge base (legal domain, Moscow-centric) ───────────────────

_DEFAULT_KB: list[KBEntry] = [
    # ── Organizations ──────────────────────────────────────────────────────
    KBEntry(
        entity_id="ORG-001",
        canonical_name="ПАО «Газпром»",
        entity_type="ORG",
        aliases=["Газпром", "Gazprom", "ОАО Газпром"],
        description="Крупнейшая газодобывающая компания России",
    ),
    KBEntry(
        entity_id="ORG-002",
        canonical_name="ПАО «Сбербанк России»",
        entity_type="ORG",
        aliases=["Сбербанк", "Sberbank", "ОАО Сбербанк", "Сбер"],
        description="Крупнейший банк России",
    ),
    KBEntry(
        entity_id="ORG-003",
        canonical_name="ООО «Яндекс»",
        entity_type="ORG",
        aliases=["Яндекс", "Yandex", "Яндекс.Такси", "Яндекс Маркет"],
        description="Российская технологическая компания",
    ),
    KBEntry(
        entity_id="ORG-004",
        canonical_name="ПАО «Лукойл»",
        entity_type="ORG",
        aliases=["Лукойл", "LUKOIL", "Lukoil"],
        description="Вертикально интегрированная нефтяная компания",
    ),
    KBEntry(
        entity_id="ORG-005",
        canonical_name="ПАО «Ростелеком»",
        entity_type="ORG",
        aliases=["Ростелеком", "Rostelecom"],
        description="Крупнейший провайдер цифровых услуг и решений",
    ),
    KBEntry(
        entity_id="ORG-006",
        canonical_name="Apple Inc.",
        entity_type="ORG",
        aliases=["Apple", "Эпл"],
        description="Американская транснациональная технологическая компания",
    ),
    KBEntry(
        entity_id="ORG-007",
        canonical_name="Google LLC",
        entity_type="ORG",
        aliases=["Google", "Гугл", "Google Inc"],
        description="Американская транснациональная технологическая компания",
    ),
    KBEntry(
        entity_id="ORG-008",
        canonical_name="Microsoft Corporation",
        entity_type="ORG",
        aliases=["Microsoft", "Майкрософт", "MSFT"],
        description="Американская транснациональная технологическая компания",
    ),
    KBEntry(
        entity_id="ORG-009",
        canonical_name="Amazon.com, Inc.",
        entity_type="ORG",
        aliases=["Amazon", "Амазон", "AWS"],
        description="Американская транснациональная технологическая компания",
    ),
    KBEntry(
        entity_id="ORG-010",
        canonical_name="ПАО «НК Роснефть»",
        entity_type="ORG",
        aliases=["Роснефть", "Rosneft", "ОАО НК Роснефть"],
        description="Крупнейшая нефтяная компания России",
    ),
    # ── Persons ────────────────────────────────────────────────────────────
    KBEntry(
        entity_id="PER-001",
        canonical_name="Путин Владимир Владимирович",
        entity_type="PER",
        aliases=["Владимир Путин", "В.В. Путин", "Путин В.В."],
        description="Президент Российской Федерации",
    ),
    KBEntry(
        entity_id="PER-002",
        canonical_name="Медведев Дмитрий Анатольевич",
        entity_type="PER",
        aliases=["Дмитрий Медведев", "Медведев Д.А."],
        description="Заместитель Председателя Совета Безопасности РФ",
    ),
    KBEntry(
        entity_id="PER-003",
        canonical_name="Мишустин Михаил Владимирович",
        entity_type="PER",
        aliases=["Михаил Мишустин", "Мишустин М.В."],
        description="Председатель Правительства Российской Федерации",
    ),
    # ── Locations ──────────────────────────────────────────────────────────
    KBEntry(
        entity_id="LOC-001",
        canonical_name="город Москва",
        entity_type="LOC",
        aliases=["Москва", "Moscow", "г. Москва", "г.Москва"],
        description="Столица Российской Федерации",
    ),
    KBEntry(
        entity_id="LOC-002",
        canonical_name="город Санкт-Петербург",
        entity_type="LOC",
        aliases=["Санкт-Петербург", "Петербург", "Питер", "Saint Petersburg", "СПб"],
        description="Второй по величине город России",
    ),
    KBEntry(
        entity_id="LOC-003",
        canonical_name="Российская Федерация",
        entity_type="LOC",
        aliases=["Россия", "РФ", "Russia", "Российской Федерации"],
        description="Государство в Евразии",
    ),
    KBEntry(
        entity_id="LOC-004",
        canonical_name="Соединённые Штаты Америки",
        entity_type="LOC",
        aliases=["США", "US", "USA", "Америка", "United States"],
        description="Государство в Северной Америке",
    ),
    KBEntry(
        entity_id="LOC-005",
        canonical_name="Китайская Народная Республика",
        entity_type="LOC",
        aliases=["Китай", "КНР", "China", "PRC"],
        description="Государство в Восточной Азии",
    ),
    # ── Courts (Russian legal domain) ──────────────────────────────────────
    KBEntry(
        entity_id="COURT-001",
        canonical_name="Верховный Суд Российской Федерации",
        entity_type="COURT",
        aliases=["Верховный суд", "ВС РФ", "Верховный Суд РФ"],
        description="Высший судебный орган по гражданским и уголовным делам",
    ),
    KBEntry(
        entity_id="COURT-002",
        canonical_name="Конституционный Суд Российской Федерации",
        entity_type="COURT",
        aliases=["Конституционный суд", "КС РФ", "КС России"],
        description="Высший орган конституционного контроля",
    ),
    KBEntry(
        entity_id="COURT-003",
        canonical_name="Арбитражный суд города Москвы",
        entity_type="COURT",
        aliases=["АСГМ", "Арбитражный суд Москвы", "арбитражный суд г. Москвы"],
        description="Суд первой инстанции по экономическим спорам в Москве",
    ),
]


class KnowledgeBase:
    """База знаний сущностей с поиском по типу.

    Хранит KBEntry записи, строит инвертированный индекс alias → entry
    для быстрого точного сопоставления.
    """

    def __init__(self, entries: list[KBEntry] | None = None) -> None:
        self._entries = entries if entries is not None else _DEFAULT_KB
        # alias (lowercase) → entry для O(1) точного сопоставления
        self._alias_index: dict[str, KBEntry] = {}
        for entry in self._entries:
            self._alias_index[entry.canonical_name.lower()] = entry
            for alias in entry.aliases:
                self._alias_index[alias.lower()] = entry

    def get_candidates(self, entity_type: str) -> list[KBEntry]:
        """Кандидаты, отфильтрованные по типу (PER/ORG/LOC/COURT)."""
        # COURT — вспомогательный тип, ORG-запросы также ищут по COURT
        if entity_type == "ORG":
            return [e for e in self._entries if e.entity_type in ("ORG", "COURT")]
        return [e for e in self._entries if e.entity_type == entity_type]

    def lookup_exact(self, text: str) -> KBEntry | None:
        """Точный поиск по каноническому имени или псевдониму (O(1))."""
        return self._alias_index.get(text.lower())

    def add_entry(self, entry: KBEntry) -> None:
        """Динамическое добавление записи (для кастомных доменов)."""
        self._entries.append(entry)
        self._alias_index[entry.canonical_name.lower()] = entry
        for alias in entry.aliases:
            self._alias_index[alias.lower()] = entry

    def __len__(self) -> int:
        return len(self._entries)


# ── Similarity scoring ────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Привести к нижнему регистру, убрать кавычки и лишние пробелы."""
    text = text.lower()
    text = re.sub(r'[«»"\'„"]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _trigrams(text: str) -> set[str]:
    """Множество 3-граммовых символьных последовательностей."""
    t = _normalize(text)
    if len(t) < 3:
        return {t}
    return {t[i : i + 3] for i in range(len(t) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    """Коэффициент Жаккара на символьных 3-граммах ∈ [0, 1]."""
    ta, tb = _trigrams(a), _trigrams(b)
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _score_candidate(mention: str, entry: KBEntry) -> float:
    """
    Многоуровневый скоринг кандидата:
    1.0  — точное совпадение с канонич. именем или псевдонимом
    0.90 — псевдоним совпадает после нормализации
    0.75 — упоминание является началом канонич. имени (prefix)
    0.50+— trigram Jaccard × 0.8 (лексическая похожесть)
    """
    norm_mention = _normalize(mention)

    # Точное совпадение (после нормализации)
    if norm_mention == _normalize(entry.canonical_name):
        return 1.0

    # Псевдоним (нормализованный)
    for alias in entry.aliases:
        if norm_mention == _normalize(alias):
            return 0.90

    # Prefix match: "Яндекс" → "Яндекс.Такси" или наоборот
    norm_canonical = _normalize(entry.canonical_name)
    if norm_canonical.startswith(norm_mention) or norm_mention.startswith(norm_canonical):
        return 0.75
    for alias in entry.aliases:
        norm_alias = _normalize(alias)
        if norm_alias.startswith(norm_mention) or norm_mention.startswith(norm_alias):
            return 0.72

    # Trigram Jaccard для fuzzy
    best_jac = _trigram_jaccard(mention, entry.canonical_name)
    for alias in entry.aliases:
        jac = _trigram_jaccard(mention, alias)
        if jac > best_jac:
            best_jac = jac

    return best_jac * 0.80  # масштабирование fuzzy-скора


# ── Entity Linker ─────────────────────────────────────────────────────────────


class EntityLinker:
    """Связывает NER-сущности с записями в базе знаний.

    Использует многоуровневый скоринг (exact → alias → prefix → trigram)
    без внешних зависимостей. Подходит для production при небольших KB (<10K записей).
    Для больших KB рекомендуется FAISS + dense embeddings.
    """

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        self._kb = kb if kb is not None else KnowledgeBase()
        self._threshold = confidence_threshold

    def link(self, mention: str, entity_type: str) -> EntityLink | None:
        """
        Связать одну сущность с записью KB.

        Возвращает None (NIL entity) если лучший скор ниже порога —
        это означает новую, неизвестную KB сущность.
        """
        # Быстрый путь: точный поиск по индексу
        exact = self._kb.lookup_exact(mention)
        if exact is not None and exact.entity_type in (entity_type, "COURT"):
            return EntityLink(
                entity_id=exact.entity_id,
                canonical_name=exact.canonical_name,
                entity_type=exact.entity_type,
                confidence=1.0,
                description=exact.description,
            )

        # Полный скоринг по кандидатам типа
        candidates = self._kb.get_candidates(entity_type)
        if not candidates:
            return None

        best_score = 0.0
        best_entry: KBEntry | None = None
        for entry in candidates:
            score = _score_candidate(mention, entry)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score < self._threshold or best_entry is None:
            return None  # NIL

        return EntityLink(
            entity_id=best_entry.entity_id,
            canonical_name=best_entry.canonical_name,
            entity_type=best_entry.entity_type,
            confidence=round(best_score, 4),
            description=best_entry.description,
        )

    def link_entities(self, entities: list[tuple[str, str, int, int]]) -> list[LinkedEntity]:
        """
        Связать список сущностей [(text, label, start, end)] с KB.

        Возвращает LinkedEntity с link=None для NIL-сущностей.
        """
        results = []
        for text, label, start, end in entities:
            link = self.link(text, label)
            results.append(LinkedEntity(text=text, label=label, start=start, end=end, link=link))
        return results
