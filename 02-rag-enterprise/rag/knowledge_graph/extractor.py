"""Entity extractor for Knowledge Graph construction.

Uses regex patterns without LLM for CI compatibility.
Production replacement: spaCy NER or Claude Haiku for higher recall.

Microsoft GraphRAG (2024) showed graph-based retrieval answers multi-hop
questions that vector search misses ("What did [person] say about [concept]?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ORG_SUFFIXES = (
    r"Inc\.|Corp\.|LLC|Ltd\.|GmbH|Company|Organization|"
    r"Association|Institute|University|Department|Group|"
    r"Systems|Technologies|Services|Platform|Labs"
)

# Each entry: pattern string. CONCEPT has capture group 1 for quoted text;
# all other patterns have no capture groups.
_PATTERNS: dict[str, str] = {
    "DATE": (
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)(?:\s+\d{1,2},?)?\s+\d{4}\b"
        r"|\b\d{4}-\d{2}-\d{2}\b"
        r"|\bQ[1-4]\s+\d{4}\b"
    ),
    # Name words each end with a space so greedy matching can't absorb the suffix.
    # Trailing lookahead instead of \b because suffixes like "Inc." end with a
    # period (non-word char), making \b fail at word boundaries after ".".
    "ORG": (r"\b(?:[A-Z][a-zA-Z&]+\s+){1,4}" + f"(?:{_ORG_SUFFIXES})" + r"(?=[^A-Za-z]|$)"),
    # Group 1 captures quoted concept; second alternative (acronym) has no group
    "CONCEPT": (
        r'"([A-Za-z][^"]{2,60})"'
        r"|\b[A-Z]{2,10}\b"
    ),
    "PERSON": (r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"),
}


@dataclass
class Entity:
    """Extracted named entity with position in source text."""

    text: str
    entity_type: str  # DATE | ORG | CONCEPT | PERSON
    chunk_id: str
    start: int = 0
    end: int = 0


def _safe_text(match: re.Match) -> str:
    """Extract entity text from match: use group 1 if captured, else group 0."""
    try:
        g1 = match.group(1)
        if g1 is not None:
            return g1.strip()
    except IndexError:
        pass
    return match.group(0).strip()


def extract_entities(text: str, chunk_id: str) -> list[Entity]:
    """Extract entities from a text chunk using regex patterns.

    Returns deduplicated list (same text+type appears once per chunk).
    """
    entities: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    for entity_type, pattern in _PATTERNS.items():
        for match in re.finditer(pattern, text):
            entity_text = _safe_text(match)
            if len(entity_text) < 2:
                continue
            key = (entity_text.lower(), entity_type)
            if key not in seen:
                seen.add(key)
                entities.append(
                    Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        chunk_id=chunk_id,
                        start=match.start(),
                        end=match.end(),
                    )
                )

    return entities
