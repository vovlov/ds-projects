"""NER model: rule-based baseline + transformer-based (when available)."""

from __future__ import annotations

import re
from typing import NamedTuple


class Entity(NamedTuple):
    text: str
    label: str
    start: int
    end: int


def extract_entities_from_bio(tokens: list[str], labels: list[str]) -> list[Entity]:
    """Extract entities from BIO-tagged sequence."""
    entities = []
    current_entity: list[str] = []
    current_label = ""
    start_idx = 0

    for i, (token, label) in enumerate(zip(tokens, labels, strict=False)):
        if label.startswith("B-"):
            if current_entity:
                entities.append(
                    Entity(
                        text=" ".join(current_entity),
                        label=current_label,
                        start=start_idx,
                        end=i,
                    )
                )
            current_entity = [token]
            current_label = label[2:]
            start_idx = i
        elif label.startswith("I-") and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append(
                    Entity(
                        text=" ".join(current_entity),
                        label=current_label,
                        start=start_idx,
                        end=i,
                    )
                )
                current_entity = []

    if current_entity:
        entities.append(
            Entity(
                text=" ".join(current_entity),
                label=current_label,
                start=start_idx,
                end=len(tokens),
            )
        )

    return entities


# ── Rule-based NER (baseline) ───────────────────────────────────

# Simple patterns for Russian NER
_PATTERNS = {
    "PER": re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b"),
    "ORG": re.compile(
        r"\b(?:ОАО|ЗАО|ООО|ПАО|АО|ИП)\s+[«\"]?[А-ЯЁA-Z][а-яёa-zA-Z\s]+[»\"]?"
        r"|\b(?:Газпром[а-яё]*|Яндекс[а-яё]*|Сбербанк[а-яё]*|Tesla|Google|Apple|Microsoft|Amazon)\b"
    ),
    "LOC": re.compile(
        r"\b(?:Москв[а-яё]*|Санкт-Петербург[а-яё]*|Казан[а-яё]*|Новосибирск[а-яё]*"
        r"|Екатеринбург[а-яё]*|Росси[а-яё]*|США|Кита[а-яё]*|Германи[а-яё]*|Франци[а-яё]*)\b"
    ),
}


def predict_rule_based(text: str) -> list[Entity]:
    """Simple rule-based NER for Russian text."""
    entities = []
    for label, pattern in _PATTERNS.items():
        for match in pattern.finditer(text):
            entities.append(
                Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                )
            )

    # Sort by position and remove overlaps
    entities.sort(key=lambda e: e.start)
    filtered = []
    last_end = -1
    for entity in entities:
        if entity.start >= last_end:
            filtered.append(entity)
            last_end = entity.end

    return filtered


def predict(text: str) -> list[Entity]:
    """Main prediction function — uses rule-based baseline."""
    return predict_rule_based(text)
