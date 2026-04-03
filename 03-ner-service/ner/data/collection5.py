"""Collection5 — русский NER корпус, поддержка и утилиты.

Collection5 (Коллекция 5) — стандартный датасет для русскоязычного NER
от Гришман и др. Формат: CoNLL (token TAB label), предложения разделены пустой строкой.
Метки: PER, ORG, LOC в BIO-схеме.

Источники:
  - https://github.com/collection-5/ner
  - Sekine, S. et al. "Named Entity: Current Research and Future Directions"
  - RuBERT NER benchmarks (DeepPavlov, 2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Типы данных
# ---------------------------------------------------------------------------

Sentence = list[tuple[str, str]]  # [(token, label), ...]
Dataset = list[Sentence]


@dataclass
class NERMetrics:
    """Метрики качества NER по каждому типу сущностей и итоговые.

    Computed via seqeval (sequence-level F1) — стандарт для NER.
    """

    precision: float
    recall: float
    f1: float
    support: int
    per_entity: dict[str, dict[str, float]] = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = [
            f"Overall   P={self.precision:.3f}  R={self.recall:.3f}"
            f"  F1={self.f1:.3f}  (support={self.support})"
        ]
        for etype, m in self.per_entity.items():
            lines.append(
                f"  {etype:<6}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Образец данных Collection5 (CoNLL-формат, встроен для CI)
#
# Выбраны предложения, репрезентативные для реального корпуса:
# новостные тексты с именами, организациями, локациями.
# ---------------------------------------------------------------------------

_COLLECTION5_CONLL_SAMPLE = """\
Путин\tB-PER
посетил\tO
Берлин\tB-LOC
и\tO
встретился\tO
с\tO
канцлером\tO
Германии\tB-LOC
.\tO

Акции\tO
Газпрома\tB-ORG
упали\tO
на\tO
3\tO
%\tO
после\tO
заявлений\tO
министра\tO
финансов\tO
США\tB-LOC
.\tO

Алексей\tB-PER
Миллер\tI-PER
возглавляет\tO
ПАО\tB-ORG
Газпром\tI-ORG
с\tO
2001\tO
года\tO
.\tO

МВФ\tB-ORG
предоставил\tO
кредит\tO
Украине\tB-LOC
в\tO
размере\tO
5\tO
млрд\tO
долларов\tO
.\tO

Наталья\tB-PER
Касперская\tI-PER
основала\tO
компанию\tO
InfoWatch\tB-ORG
в\tO
Москве\tB-LOC
.\tO

Центробанк\tB-ORG
России\tB-LOC
повысил\tO
ключевую\tO
ставку\tO
до\tO
16\tO
%\tO
.\tO

Матч\tO
прошёл\tO
в\tO
Санкт-Петербурге\tB-LOC
на\tO
стадионе\tO
Газпром\tB-ORG
Арена\tI-ORG
.\tO

Сергей\tB-PER
Лавров\tI-PER
провёл\tO
переговоры\tO
с\tO
представителями\tO
ООН\tB-ORG
в\tO
Женеве\tB-LOC
.\tO
"""


# ---------------------------------------------------------------------------
# Парсер CoNLL
# ---------------------------------------------------------------------------


def parse_conll(text: str, sep: str = "\t") -> Dataset:
    """Parse CoNLL-formatted text into a list of sentences.

    Each sentence is a list of (token, label) pairs.
    Sentences are separated by blank lines.

    Args:
        text: Raw CoNLL text.
        sep: Column separator (tab by default).

    Returns:
        List of sentences, each sentence is [(token, label), ...].
    """
    sentences: Dataset = []
    current: Sentence = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            # Конец предложения — пустая строка разделяет предложения в CoNLL
            if current:
                sentences.append(current)
                current = []
        else:
            parts = line.split(sep)
            if len(parts) >= 2:
                token, label = parts[0], parts[1]
                current.append((token, label))

    if current:
        sentences.append(current)

    return sentences


def sentences_to_bio(sentences: Dataset) -> tuple[list[list[str]], list[list[str]]]:
    """Extract tokens and labels from parsed sentences.

    Returns:
        (tokens_list, labels_list) — parallel lists of lists.
    """
    tokens_list = [[tok for tok, _ in sent] for sent in sentences]
    labels_list = [[lbl for _, lbl in sent] for sent in sentences]
    return tokens_list, labels_list


# ---------------------------------------------------------------------------
# Загрузчик датасета
# ---------------------------------------------------------------------------


def load_collection5(path: Path | None = None) -> Dataset:
    """Load Collection5 dataset from CoNLL file or return built-in sample.

    Args:
        path: Path to a CoNLL file. If None, returns the built-in sample
              (safe for CI — no network required).

    Returns:
        List of sentences in [(token, label)] format.
    """
    if path is None:
        # Встроенный образец — для тестирования и демо без скачивания
        return parse_conll(_COLLECTION5_CONLL_SAMPLE)

    with open(path, encoding="utf-8") as f:
        return parse_conll(f.read())


def get_collection5_sample() -> Dataset:
    """Return the built-in Collection5-style sample (CI-safe)."""
    return parse_conll(_COLLECTION5_CONLL_SAMPLE)


# ---------------------------------------------------------------------------
# Статистика датасета
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Статистика датасета NER."""

    num_sentences: int
    num_tokens: int
    entity_counts: dict[str, int]

    def __repr__(self) -> str:
        lines = [
            f"Sentences : {self.num_sentences}",
            f"Tokens    : {self.num_tokens}",
            "Entities  :",
        ]
        for etype, cnt in sorted(self.entity_counts.items()):
            lines.append(f"  {etype:<6}: {cnt}")
        return "\n".join(lines)


def compute_dataset_stats(dataset: Dataset) -> DatasetStats:
    """Compute basic statistics for a NER dataset.

    Args:
        dataset: List of sentences (parsed CoNLL).

    Returns:
        DatasetStats with sentence/token/entity counts.
    """
    num_sentences = len(dataset)
    num_tokens = sum(len(sent) for sent in dataset)
    entity_counts: dict[str, int] = {}

    for sent in dataset:
        in_entity = False
        for _, label in sent:
            if label.startswith("B-"):
                etype = label[2:]
                entity_counts[etype] = entity_counts.get(etype, 0) + 1
                in_entity = True
            elif label.startswith("I-") and in_entity:
                pass  # продолжение сущности
            else:
                in_entity = False

    return DatasetStats(
        num_sentences=num_sentences,
        num_tokens=num_tokens,
        entity_counts=entity_counts,
    )


# ---------------------------------------------------------------------------
# Метрики качества (через seqeval)
# ---------------------------------------------------------------------------


def compute_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> NERMetrics:
    """Compute NER evaluation metrics using seqeval.

    Seqeval реализует chunk-level F1 — стандарт для CoNLL-оценки.
    Считает precision/recall/F1 на уровне сущностей (не токенов),
    что корректнее для NER задач.

    Args:
        true_labels: Ground-truth BIO labels per sentence.
        pred_labels: Predicted BIO labels per sentence.

    Returns:
        NERMetrics with overall and per-entity scores.
    """
    try:
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise ImportError("seqeval required: uv sync --extra ner") from exc

    overall_p = precision_score(true_labels, pred_labels, zero_division=0)
    overall_r = recall_score(true_labels, pred_labels, zero_division=0)
    overall_f1 = f1_score(true_labels, pred_labels, zero_division=0)

    # Число токенов-сущностей в ground truth
    support = sum(1 for sent in true_labels for lbl in sent if lbl != "O")

    # Per-entity breakdown через classification_report
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    per_entity: dict[str, dict[str, float]] = {}
    for key, val in report.items():
        # seqeval возвращает ключи вида 'PER', 'ORG', 'LOC', 'micro avg', etc.
        if isinstance(val, dict) and key not in ("micro avg", "macro avg", "weighted avg"):
            per_entity[key] = {
                "precision": float(val.get("precision", 0.0)),
                "recall": float(val.get("recall", 0.0)),
                "f1": float(val.get("f1-score", 0.0)),
            }

    return NERMetrics(
        precision=float(overall_p),
        recall=float(overall_r),
        f1=float(overall_f1),
        support=support,
        per_entity=per_entity,
    )
