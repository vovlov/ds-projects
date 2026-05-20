"""Active Learning для NER — отбор неразмеченных текстов для аннотации.

Uncertainty Sampling (Lewis & Gale 1994): модель указывает, какие примеры
требуют аннотации эксперта — информационная ценность нового примера
пропорциональна неопределённости текущей модели.

Интеграция с конформным предсказателем: nonconformity_score напрямую
служит мерой неопределённости (высокий score → модель не уверена).
Тексты без распознанных сущностей получают максимальный score=1.0 —
это cold-start: новая территория требует аннотации в первую очередь.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class SamplingStrategy(StrEnum):
    """Стратегия отбора кандидатов для аннотации."""

    UNCERTAINTY = "uncertainty"  # максимальный nonconformity score
    MARGIN = "margin"  # размер prediction_set (неоднозначность типа)
    ENTROPY = "entropy"  # энтропия паттерн-скоров по всем label
    RANDOM = "random"  # случайный baseline


@dataclass
class AnnotationCandidate:
    """Текст, отобранный для аннотации экспертом."""

    candidate_id: str
    text: str
    uncertainty_score: float  # 0 → уверен, 1 → полная неопределённость
    sampling_reason: str  # human-readable объяснение выбора
    predicted_entities: list[dict]  # текущие предсказания NER
    strategy: str
    annotated: bool = False
    annotation: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "uncertainty_score": round(self.uncertainty_score, 4),
            "sampling_reason": self.sampling_reason,
            "predicted_entities": self.predicted_entities,
            "strategy": self.strategy,
            "annotated": self.annotated,
        }


@dataclass
class ActiveLearningStats:
    """Статистика сессии активного обучения."""

    total_candidates: int
    annotated: int
    pending: int
    avg_uncertainty: float
    recalibrations: int
    strategy_used: str

    def to_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "annotated": self.annotated,
            "pending": self.pending,
            "avg_uncertainty": round(self.avg_uncertainty, 4),
            "recalibrations": self.recalibrations,
            "strategy_used": self.strategy_used,
        }


class ActiveLearner:
    """
    Активный обучающий агент для NER.

    Выбирает тексты для аннотации по мере неопределённости конформного
    предсказателя. При получении аннотаций обновляет калибровочный набор
    и пересчитывает q_hat — модель становится точнее без полного переобучения.
    """

    def __init__(
        self,
        conformal_predictor: object | None = None,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
        seed: int | None = None,
    ) -> None:
        self._conformal = conformal_predictor
        self._strategy = strategy
        self._queue: list[AnnotationCandidate] = []
        self._annotated: list[AnnotationCandidate] = []
        self._recalibrations: int = 0
        self._rng = random.Random(seed)

    # ── Uncertainty computation ──────────────────────────────────────────────

    def _uncertainty_score(self, text: str) -> tuple[float, str, list[dict]]:
        """
        Вернуть (score, reason, entities) для текста по текущей стратегии.

        Возвращает score ∈ [0, 1]: 0 — модель уверена, 1 — полная неопределённость.
        """
        if self._conformal is None:
            # Без предсказателя — случайный score (baseline тест)
            sc = self._rng.random()
            return sc, "no_predictor: random score", []

        from ..model.ner import predict

        entities = predict(text)
        if not entities:
            # Текст без сущностей — cold-start: неизвестная территория
            return 1.0, "cold_start: no entities detected, high priority for annotation", []

        predicted_dicts = [
            {"text": e.text, "label": e.label, "start": e.start, "end": e.end} for e in entities
        ]

        if self._strategy == SamplingStrategy.RANDOM:
            return self._rng.random(), "random sampling baseline", predicted_dicts

        if self._strategy == SamplingStrategy.UNCERTAINTY:
            return self._score_uncertainty(entities, predicted_dicts)

        if self._strategy == SamplingStrategy.MARGIN:
            return self._score_margin(entities, predicted_dicts)

        if self._strategy == SamplingStrategy.ENTROPY:
            return self._score_entropy(entities, predicted_dicts)

        return 0.5, f"unknown strategy: {self._strategy}", predicted_dicts

    def _score_uncertainty(
        self, entities: list, predicted_dicts: list[dict]
    ) -> tuple[float, str, list[dict]]:
        """Max nonconformity score среди всех сущностей в тексте."""
        scores = [self._conformal._nonconformity_score(e.text, e.label) for e in entities]
        max_score = max(scores)
        worst_entity = entities[scores.index(max_score)]
        reason = (
            f"uncertainty: '{worst_entity.text}' has nonconformity={max_score:.3f}, "
            f"model unsure about entity type"
        )
        return max_score, reason, predicted_dicts

    def _score_margin(
        self, entities: list, predicted_dicts: list[dict]
    ) -> tuple[float, str, list[dict]]:
        """
        Средний размер prediction_set — большой set означает неоднозначность типа.

        Нормировано на число меток: 1/n_labels → 0 (уверен), 1 → все метки равновероятны.
        """
        from ..model.conformal import ALL_LABELS

        n_labels = len(ALL_LABELS)
        set_sizes = [len(self._conformal.predict_set(e).prediction_set) for e in entities]
        avg_size = sum(set_sizes) / len(set_sizes)
        score = (avg_size - 1) / max(n_labels - 1, 1)
        worst_idx = set_sizes.index(max(set_sizes))
        reason = (
            f"margin: avg prediction_set size={avg_size:.1f}/{n_labels}, "
            f"entity '{entities[worst_idx].text}' is most ambiguous"
        )
        return min(score, 1.0), reason, predicted_dicts

    def _score_entropy(
        self, entities: list, predicted_dicts: list[dict]
    ) -> tuple[float, str, list[dict]]:
        """
        Shannon entropy паттерн-скоров по всем меткам.

        Максимальная энтропия = log(n_labels) при равномерном распределении.
        Нормировано на [0, 1].
        """
        from ..model.conformal import ALL_LABELS

        max_entropy = math.log(len(ALL_LABELS))
        entropies = []
        for e in entities:
            scores_raw = [self._conformal._pattern_score(e.text, lb) for lb in ALL_LABELS]
            total = sum(scores_raw)
            if total == 0:
                entropies.append(1.0)  # нет паттерна → максимальная неопределённость
            else:
                probs = [s / total for s in scores_raw]
                entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
                entropies.append(max(0.0, entropy / max_entropy) if max_entropy > 0 else 0.0)

        max_ent = max(entropies)
        worst_idx = entropies.index(max_ent)
        reason = (
            f"entropy: normalized entropy={max_ent:.3f} for "
            f"'{entities[worst_idx].text}', label distribution is unclear"
        )
        return max_ent, reason, predicted_dicts

    # ── Sampling ─────────────────────────────────────────────────────────────

    def select_candidates(self, texts: list[str], n: int = 5) -> list[AnnotationCandidate]:
        """
        Выбрать top-n текстов для аннотации из пула.

        Возвращает отсортированный по убыванию неопределённости список
        кандидатов и добавляет их в внутреннюю очередь.
        """
        scored: list[tuple[float, str, str, list[dict]]] = []
        for text in texts:
            score, reason, preds = self._uncertainty_score(text)
            scored.append((score, text, reason, preds))

        # Сортировка: самые неопределённые — первые
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:n]

        candidates = [
            AnnotationCandidate(
                candidate_id=str(uuid.uuid4()),
                text=text,
                uncertainty_score=score,
                sampling_reason=reason,
                predicted_entities=preds,
                strategy=str(self._strategy),
            )
            for score, text, reason, preds in selected
        ]

        # Дедупликация: не добавлять уже имеющиеся тексты
        existing_texts = {c.text for c in self._queue} | {c.text for c in self._annotated}
        new_candidates = [c for c in candidates if c.text not in existing_texts]
        self._queue.extend(new_candidates)

        return candidates

    # ── Annotation feedback ──────────────────────────────────────────────────

    def receive_annotation(self, candidate_id: str, annotation: list[dict]) -> bool:
        """
        Принять аннотацию эксперта и инициировать рекалибровку.

        annotation — список сущностей вида {"text": ..., "label": ...}.
        Возвращает True если рекалибровка выполнена успешно.
        """
        candidate = self._find_in_queue(candidate_id)
        if candidate is None:
            return False

        candidate.annotated = True
        candidate.annotation = annotation
        self._queue.remove(candidate)
        self._annotated.append(candidate)

        # Рекалибровать конформный предсказатель на новых аннотациях
        if self._conformal is not None and annotation:
            self._recalibrate_from_annotations()

        return True

    def _find_in_queue(self, candidate_id: str) -> AnnotationCandidate | None:
        for c in self._queue:
            if c.candidate_id == candidate_id:
                return c
        return None

    def _recalibrate_from_annotations(self) -> None:
        """Обновить q_hat конформного предсказателя на всех накопленных аннотациях."""
        from ..model.ner import Entity

        all_entities: list[Entity] = []
        for candidate in self._annotated:
            for ann in candidate.annotation:
                all_entities.append(
                    Entity(
                        text=ann.get("text", ""),
                        label=ann.get("label", "PER"),
                        start=ann.get("start", 0),
                        end=ann.get("end", 0),
                    )
                )

        if all_entities:
            self._conformal.calibrate(all_entities)
            self._recalibrations += 1

    # ── State & stats ─────────────────────────────────────────────────────────

    def get_stats(self) -> ActiveLearningStats:
        """Вернуть статистику сессии активного обучения."""
        all_candidates = self._queue + self._annotated
        avg_uncertainty = (
            sum(c.uncertainty_score for c in all_candidates) / len(all_candidates)
            if all_candidates
            else 0.0
        )
        return ActiveLearningStats(
            total_candidates=len(all_candidates),
            annotated=len(self._annotated),
            pending=len(self._queue),
            avg_uncertainty=avg_uncertainty,
            recalibrations=self._recalibrations,
            strategy_used=str(self._strategy),
        )

    def get_queue(self) -> list[AnnotationCandidate]:
        """Вернуть очередь ожидающих аннотации кандидатов (по убыванию неопределённости)."""
        return sorted(self._queue, key=lambda c: c.uncertainty_score, reverse=True)

    def reset(self) -> None:
        """Сбросить состояние — для тестов и перезапуска сессии."""
        self._queue.clear()
        self._annotated.clear()
        self._recalibrations = 0
