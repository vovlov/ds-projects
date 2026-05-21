"""
Active learning sampling strategies for NER annotation.

Uncertainty sampling selects unlabeled examples where the model is least certain,
maximizing information gain per annotation dollar. Integrates with ConformalNERPredictor:
nonconformity_score ∈ [0,1] (0 = certain, 1 = maximally uncertain) is used as the
confidence proxy — no additional inference passes required.

References:
  Lewis & Gale 1994 (Least Confidence, Margin),
  Shannon 1948 (Entropy),
  Settles 2012 "Active Learning" synthesis lecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple


class SamplingStrategy(StrEnum):
    LEAST_CONFIDENCE = "least_confidence"
    MARGIN = "margin"
    ENTROPY = "entropy"


@dataclass
class ActiveLearningConfig:
    """Конфигурация стратегии активного обучения."""

    strategy: SamplingStrategy = SamplingStrategy.LEAST_CONFIDENCE
    batch_size: int = 10


class UncertaintyScore(NamedTuple):
    """Оценка неопределённости для одного текста."""

    text: str
    score: float  # Чем выше — тем ценнее аннотировать
    n_entities: int  # Кол-во найденных сущностей
    strategy: str


def least_confidence_score(nonconformity_scores: list[float]) -> float:
    """
    LC uncertainty = max nonconformity across entities.

    Выбираем текст, где хотя бы одна сущность вызывает максимальную неопределённость.
    Nonconformity ∈ [0,1]: 0 = полная уверенность, 1 = модель не знает.
    """
    if not nonconformity_scores:
        return 0.0
    return max(nonconformity_scores)


def margin_score(nonconformity_scores: list[float]) -> float:
    """
    Margin uncertainty = разница между худшей и лучшей уверенностью.

    Высокое значение → в тексте есть как уверенные, так и неуверенные сущности:
    точечная аннотация даст максимальный сигнал.
    """
    if not nonconformity_scores:
        return 0.0
    if len(nonconformity_scores) == 1:
        return nonconformity_scores[0]
    return max(nonconformity_scores) - min(nonconformity_scores)


def entropy_score(nonconformity_scores: list[float]) -> float:
    """
    Entropy uncertainty = среднее энтропии бинарного распределения уверенности.

    Каждую сущность трактуем как бинарную переменную (правильно/неправильно).
    Максимум при nonconformity = 0.5 (полная неопределённость).
    """
    if not nonconformity_scores:
        return 0.0

    def binary_entropy(p_error: float) -> float:
        p = max(1e-10, min(1.0 - 1e-10, p_error))
        q = 1.0 - p
        return -(p * math.log2(p) + q * math.log2(q))

    return sum(binary_entropy(s) for s in nonconformity_scores) / len(nonconformity_scores)


def score_text(
    text: str,
    nonconformity_scores: list[float],
    config: ActiveLearningConfig,
) -> UncertaintyScore:
    """Вычислить оценку неопределённости для текста по выбранной стратегии."""
    strategy = config.strategy

    if strategy == SamplingStrategy.LEAST_CONFIDENCE:
        score = least_confidence_score(nonconformity_scores)
    elif strategy == SamplingStrategy.MARGIN:
        score = margin_score(nonconformity_scores)
    else:  # ENTROPY
        score = entropy_score(nonconformity_scores)

    return UncertaintyScore(
        text=text,
        score=round(score, 6),
        n_entities=len(nonconformity_scores),
        strategy=str(strategy),
    )
