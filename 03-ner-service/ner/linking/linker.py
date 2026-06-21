"""
Named Entity Linker: связывает NER-упоминания с записями в базе знаний.

Алгоритм (без внешних зависимостей, CI-friendly):
1. Точное совпадение нормализованного упоминания с alias-индексом KB → score=1.0.
2. Нечёткое сопоставление: Jaccard(char-3-grams(mention), char-3-grams(alias)).
3. type_match_bonus: бонус за совпадение NER-типа с типом сущности в KB.
4. Принять кандидата, если confidence ≥ threshold.

Источники:
- Mihalcea & Csomai 2007 (ACL, entity linking через Wikipedia anchors)
- Milne & Witten 2008 (CIKM, anchor-based disambiguation)
- Shen et al. 2015 (ACM CSUR, entity linking survey)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge_base import EntityRecord, KnowledgeBase, _normalize_text


@dataclass
class LinkingConfig:
    confidence_threshold: float = 0.5
    n_candidates: int = 5
    type_match_bonus: float = 0.15
    ngram_size: int = 3


@dataclass
class EntityLinkResult:
    mention: str
    entity_type: str  # тип из NER (PER/ORG/LOC)
    entity_id: str | None  # Wikidata-style id или None если не найдено
    canonical_name: str | None
    confidence: float
    is_linked: bool
    candidates: list[dict] = field(default_factory=list)


def _char_ngrams(text: str, n: int) -> set[str]:
    """Символьные n-граммы; тексты короче n возвращают {text}."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    """Jaccard-сходство пары строк на символьных n-граммах."""
    sa = _char_ngrams(a, n)
    sb = _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class EntityLinker:
    """
    NEL: связывает упоминания (mention, entity_type) с записями в KnowledgeBase.

    Использует точное совпадение как fast-path и Jaccard n-gram для нечёткого поиска.
    Graceful degradation: работает без любых внешних зависимостей.
    """

    def __init__(
        self,
        kb: KnowledgeBase | None = None,
        config: LinkingConfig | None = None,
    ) -> None:
        self._kb = kb or KnowledgeBase()
        self._config = config or LinkingConfig()

    def _score_candidate(self, mention_norm: str, record: EntityRecord, mention_type: str) -> float:
        """max-alias Jaccard + type_match_bonus (cap at 1.0)."""
        alias_norms = [_normalize_text(a) for a in [record.canonical_name, *record.aliases] if a]
        if not alias_norms:
            return 0.0

        best_sim = max(
            _jaccard(mention_norm, an, self._config.ngram_size) for an in alias_norms if an
        )
        type_bonus = self._config.type_match_bonus if record.entity_type == mention_type else 0.0
        return min(1.0, best_sim + type_bonus)

    def link_mention(self, mention: str, entity_type: str) -> EntityLinkResult:
        """
        Связать единственное упоминание с записью KB.

        Fast path: exact alias lookup O(1).
        Slow path: scored scan всех сущностей KB.
        """
        # 1) Точное совпадение
        exact_id = self._kb.exact_lookup(mention)
        if exact_id is not None:
            record = self._kb.get_entity(exact_id)
            if record is not None:
                return EntityLinkResult(
                    mention=mention,
                    entity_type=entity_type,
                    entity_id=exact_id,
                    canonical_name=record.canonical_name,
                    confidence=1.0,
                    is_linked=True,
                    candidates=[
                        {
                            "entity_id": exact_id,
                            "canonical_name": record.canonical_name,
                            "entity_type": record.entity_type,
                            "score": 1.0,
                        }
                    ],
                )

        # 2) Нечёткое совпадение
        mention_norm = _normalize_text(mention)
        scored: list[tuple[float, EntityRecord]] = []

        for record in self._kb.all_entities():
            score = self._score_candidate(mention_norm, record, entity_type)
            if score > 0.0:
                scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._config.n_candidates]

        candidates_out = [
            {
                "entity_id": r.entity_id,
                "canonical_name": r.canonical_name,
                "entity_type": r.entity_type,
                "score": round(s, 4),
            }
            for s, r in top
        ]

        if top and top[0][0] >= self._config.confidence_threshold:
            best_score, best_record = top[0]
            return EntityLinkResult(
                mention=mention,
                entity_type=entity_type,
                entity_id=best_record.entity_id,
                canonical_name=best_record.canonical_name,
                confidence=round(best_score, 4),
                is_linked=True,
                candidates=candidates_out,
            )

        return EntityLinkResult(
            mention=mention,
            entity_type=entity_type,
            entity_id=None,
            canonical_name=None,
            confidence=round(top[0][0], 4) if top else 0.0,
            is_linked=False,
            candidates=candidates_out,
        )

    def link_entities(self, entities: list[tuple[str, str]]) -> list[EntityLinkResult]:
        """Связать список (mention, entity_type) с базой знаний."""
        return [self.link_mention(mention, etype) for mention, etype in entities]
