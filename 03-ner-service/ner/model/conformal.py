"""
Split Conformal Prediction для NER — калибровка уверенности классификации сущностей.

Split Conformal Prediction (Papadopoulos et al. 2002, Angelopoulos & Bates 2022):
  Статистически валидная гарантия покрытия: P(true_label ∈ C(X)) ≥ 1-α
  без предположений о распределении данных (distribution-free).

Применение: production escalation к эксперту при неуверенных предсказаниях,
  соответствие EU AI Act Article 13 (прозрачность ML-систем).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .ner import _PATTERNS, Entity

ALL_LABELS: list[str] = ["PER", "ORG", "LOC"]


@dataclass
class ConformalConfig:
    """Параметры конформного предсказателя."""

    alpha: float = 0.1
    min_calibration_samples: int = 10


@dataclass
class ConformalEntityResult:
    """Сущность NER с конформным множеством предсказаний."""

    text: str
    label: str
    start: int
    end: int
    nonconformity_score: float
    prediction_set: list[str]
    is_certain: bool
    coverage: float


@dataclass
class CalibrationResult:
    """Результат калибровки конформного предсказателя."""

    q_hat: float
    n_calibration: int
    alpha: float
    coverage_empirical: float
    score_distribution: list[float] = field(default_factory=list)


class ConformalNERPredictor:
    """
    Inductive Conformal Predictor для классификации типов именованных сущностей.

    Nonconformity score = 1 - P̂(true_label | entity_text), где P̂ аппроксимируется
    через нормализованные оценки совпадения с regex-паттернами.

    Калибровка: q_hat = quantile((1-α)(1 + 1/n)) по nonconformity-скорам на
    held-out данных (конечная поправка Venn-Abers для coverage ≥ 1-α).
    """

    def __init__(self, config: ConformalConfig | None = None) -> None:
        self.config = config or ConformalConfig()
        self.q_hat: float = 1.0
        self._calibrated: bool = False
        self._n_calibration: int = 0

    def _pattern_score(self, text: str, label: str) -> float:
        """
        Оценка совпадения текста с паттерном для label.

        Возвращает [0, 1]: 1.0 — полное совпадение, 0.6 — частичное, 0.0 — нет.
        """
        pattern = _PATTERNS.get(label)
        if pattern is None:
            return 0.0
        stripped = text.strip()
        if pattern.fullmatch(stripped):
            return 1.0
        if pattern.search(stripped):
            return 0.6
        return 0.0

    def _nonconformity_score(self, text: str, label: str) -> float:
        """
        Nonconformity score для пары (text, label): 1 - P̂(label | text).

        Низкий score → сущность хорошо соответствует label → высокая уверенность.
        """
        scores = {lb: self._pattern_score(text, lb) for lb in ALL_LABELS}
        total = sum(scores.values())

        if total == 0.0:
            return 1.0 - 1.0 / len(ALL_LABELS)

        return 1.0 - scores[label] / total

    def calibrate(
        self,
        entities: list[Entity],
        true_labels: list[str] | None = None,
    ) -> CalibrationResult:
        """
        Калибровать q_hat на held-out сущностях.

        Если true_labels не задан — считаем, что предсказания entities верны.
        """
        if true_labels is None:
            true_labels = [e.label for e in entities]

        scores = [
            self._nonconformity_score(e.text, lbl)
            for e, lbl in zip(entities, true_labels, strict=False)
        ]

        n = len(scores)
        if n < self.config.min_calibration_samples:
            self.q_hat = 1.0
        else:
            # Конечная поправка: level = (n+1)(1-α)/n, capped at 1.0
            level = min((n + 1) * (1.0 - self.config.alpha) / n, 1.0)
            sorted_scores = sorted(scores)
            idx = min(math.ceil(level * n) - 1, n - 1)
            self.q_hat = float(sorted_scores[idx])

        self._calibrated = True
        self._n_calibration = n

        covered = sum(1 for s in scores if s <= self.q_hat)
        empirical_coverage = covered / n if n > 0 else 0.0

        return CalibrationResult(
            q_hat=self.q_hat,
            n_calibration=n,
            alpha=self.config.alpha,
            coverage_empirical=empirical_coverage,
            score_distribution=scores,
        )

    def predict_set(self, entity: Entity) -> ConformalEntityResult:
        """
        Вернуть конформное множество предсказаний для сущности.

        Включает label тогда и только тогда, когда nonconformity_score ≤ q_hat.
        Гарантия: истинный label попадает в множество с вероятностью ≥ 1-α.
        """
        prediction_set = [
            lb for lb in ALL_LABELS if self._nonconformity_score(entity.text, lb) <= self.q_hat
        ]
        if not prediction_set:
            prediction_set = [entity.label]

        nc_score = self._nonconformity_score(entity.text, entity.label)

        return ConformalEntityResult(
            text=entity.text,
            label=entity.label,
            start=entity.start,
            end=entity.end,
            nonconformity_score=round(nc_score, 4),
            prediction_set=prediction_set,
            is_certain=len(prediction_set) == 1,
            coverage=round(1.0 - self.config.alpha, 4),
        )

    def predict_text(self, text: str) -> list[ConformalEntityResult]:
        """Запустить NER и добавить конформные оценки уверенности."""
        from .ner import predict

        return [self.predict_set(e) for e in predict(text)]
