"""NER dataset loading and preprocessing."""

from __future__ import annotations

NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(NER_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(NER_LABELS)}

# Sample annotated data for demo/testing (BIO format)
SAMPLE_DATA = [
    {
        "tokens": [
            "Владимир",
            "Путин",
            "посетил",
            "Москву",
            "и",
            "встретился",
            "с",
            "представителями",
            "Газпрома",
            ".",
        ],
        "labels": ["B-PER", "I-PER", "O", "B-LOC", "O", "O", "O", "O", "B-ORG", "O"],
    },
    {
        "tokens": ["Компания", "Яндекс", "открыла", "новый", "офис", "в", "Санкт-Петербурге", "."],
        "labels": ["O", "B-ORG", "O", "O", "O", "O", "B-LOC", "O"],
    },
    {
        "tokens": ["Илон", "Маск", "является", "CEO", "компании", "Tesla", "."],
        "labels": ["B-PER", "I-PER", "O", "O", "O", "B-ORG", "O"],
    },
    {
        "tokens": [
            "Сбербанк",
            "заключил",
            "соглашение",
            "с",
            "Правительством",
            "Московской",
            "области",
            ".",
        ],
        "labels": ["B-ORG", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "O"],
    },
    {
        "tokens": ["Мария", "Иванова", "переехала", "из", "Казани", "в", "Новосибирск", "."],
        "labels": ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "B-LOC", "O"],
    },
]


def get_sample_data() -> list[dict]:
    """Return sample NER data for demo and testing."""
    return SAMPLE_DATA


def tokens_to_text(tokens: list[str]) -> str:
    """Join tokens back into readable text."""
    return " ".join(tokens)
