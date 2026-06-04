"""
Decoupled Confident Learning (DeCoLe) — обнаружение ошибок разметки по подгруппам.
Decoupled Confident Learning for subgroup-aware label error detection.

Стандартный Confident Learning (Northcutt et al. 2021, JAIR) строит "confident joint" —
приближение матрицы перехода шума Q[s,y]: P(Ỹ=s | Y=y) по всему датасету.
Это работает, когда шум равномерно распределён по данным.

Standard Confident Learning (Northcutt et al. 2021, JAIR) builds a "confident joint" —
an approximation of the noise transition matrix Q[s,y]: P(Ỹ=s | Y=y) over the whole dataset.
This works when noise is uniformly distributed across data.

DeCoLe (arXiv:2507.07216, 2025) декуплирует модель шума по подгруппам (g):
Q_g[s,y] = P(Ỹ=s | Y=y, G=g) — у каждой группы своя матрица шума.
Это ловит систематические ошибки аннотации, специфичные для конкретного слоя данных:
  - Аннотатор X систематически путает класс A с B только для текстов на испанском
  - Данные из источника S имеют повышенный шум для класса "мошенничество"
  - Демографическая группа underserved имеет worse label quality (систематическое неравенство)

DeCoLe (arXiv:2507.07216, 2025) decouples the noise model per subgroup (g):
Q_g[s,y] = P(Ỹ=s | Y=y, G=g) — each group has its own noise matrix.
This catches systematic annotation errors specific to a particular data slice:
  - Annotator X systematically confuses class A with B only for Spanish texts
  - Data from source S has elevated noise for class "fraud"
  - Underserved demographic groups have worse label quality (systematic inequality)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass
class LabelError:
    """Один подозрительный пример с возможной ошибкой разметки.
    One suspect example with a likely labeling error.
    """

    index: int
    given_label: int
    suggested_label: int
    confidence: float
    group: str | None
    error_type: str  # "off_diagonal" | "high_noise_group" | "confident_disagreement"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "given_label": self.given_label,
            "suggested_label": self.suggested_label,
            "confidence": round(self.confidence, 4),
            "group": self.group,
            "error_type": self.error_type,
        }


@dataclass
class NoiseMatrix:
    """Оценка матрицы перехода шума для одной подгруппы.
    Estimated label noise transition matrix for one subgroup.
    """

    group: str
    n_examples: int
    matrix: np.ndarray  # shape [K, K]: Q[noisy_label, true_label]
    noise_rate: float  # доля примеров с неправильными метками / fraction of mislabeled examples

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "n_examples": self.n_examples,
            "noise_rate": round(self.noise_rate, 4),
            "matrix": self.matrix.round(4).tolist(),
        }


@dataclass
class LabelQualityReport:
    """Полный отчёт о качестве разметки датасета.
    Full label quality audit report for a dataset.
    """

    audit_id: str
    timestamp: str
    n_examples: int
    n_classes: int
    n_groups: int
    n_errors_found: int
    error_rate: float
    errors: list[LabelError]
    noise_matrices: list[NoiseMatrix]

    def to_dict(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "n_examples": self.n_examples,
            "n_classes": self.n_classes,
            "n_groups": self.n_groups,
            "n_errors_found": self.n_errors_found,
            "error_rate": round(self.error_rate, 4),
            "errors": [e.to_dict() for e in self.errors],
            "noise_matrices": [m.to_dict() for m in self.noise_matrices],
        }


class DecoupledConfidentLearning:
    """
    DeCoLe: Decoupled Confident Learning для обнаружения ошибок разметки.
    DeCoLe: Decoupled Confident Learning for label error detection.

    Алгоритм / Algorithm:
    1. Для каждой группы g вычисляем порог t_{g,j} = mean(P̂(Y=j|X)) по примерам с ỹ=j в группе g.
    2. Пример (x_i, ỹ_i) в группе g "confident" в классе j ↔ P̂(Y=j|x_i) ≥ t_{g,j}.
    3. Confident joint C_g[s,j] = |{i ∈ g : ỹ_i=s, argmax P̂=j, P̂(j)≥t_{g,j}}|.
    4. Нормируем → матрица шума Q_g[s,j].
    5. Примеры с ỹ_i ≠ argmax P̂ и P̂(argmax) ≥ t_{g,argmax} — кандидаты на ошибки.

    Для каждой группы g compute threshold t_{g,j} = mean(P̂(Y=j|X)) for examples with ỹ=j in group g.
    Example (x_i, ỹ_i) in group g is "confident" in class j ↔ P̂(Y=j|x_i) ≥ t_{g,j}.
    Confident joint C_g[s,j] = |{i ∈ g : ỹ_i=s, argmax P̂=j, P̂(j)≥t_{g,j}}|.
    Normalize → noise matrix Q_g[s,j]. Examples with ỹ_i ≠ argmax P̂ and P̂(argmax) ≥ t_{g,argmax}
    are label error candidates.

    Ключевое отличие от CL / Key difference from standard CL:
    Порог t_{g,j} специфичен для группы g — группы с плохим качеством разметки
    не снижают порог для чистых групп. Это устраняет систематический bias глобального порога,
    который позволяет "зашумлённым" группам загрязнять оценку шума для "чистых" групп.

    The group-specific threshold t_{g,j} means high-noise groups don't lower the bar
    for clean groups — eliminating the systematic bias in global threshold estimation.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: fallback-порог если у группы нет примеров данного класса.
                Fallback threshold when a group has no examples of a given class.
        """
        self.confidence_threshold = confidence_threshold

    def _compute_thresholds(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
    ) -> np.ndarray:
        """Вычислить per-class порог = mean(P̂(y=j|x)) для x: ỹ=j."""
        n_classes = pred_probs.shape[1]
        thresholds = np.full(n_classes, self.confidence_threshold)
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                thresholds[c] = float(pred_probs[mask, c].mean())
        return thresholds

    def _estimate_noise_matrix(
        self,
        labels: np.ndarray,
        pred_probs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Вычислить confident joint и оценить матрицу шума.
        Compute confident joint and estimate the noise transition matrix.
        Returns (noise_matrix [K,K], estimated_noise_rate).
        """
        n_classes = pred_probs.shape[1]
        thresholds = self._compute_thresholds(labels, pred_probs)

        C = np.zeros((n_classes, n_classes), dtype=float)
        for local_i in range(len(labels)):
            s = labels[local_i]
            probs = pred_probs[local_i]
            # Predicted class confident in: argmax prob that exceeds its threshold
            confident_classes = np.where(probs >= thresholds)[0]
            if len(confident_classes) == 0:
                continue
            j = int(confident_classes[np.argmax(probs[confident_classes])])
            if j != s:
                C[s, j] += 1.0

        row_sums = C.sum(axis=1, keepdims=True)
        # safe divide: rows with zero sum (no off-diagonal confident examples) → 0
        with np.errstate(divide="ignore", invalid="ignore"):
            noise_matrix = np.where(row_sums > 0, C / row_sums, 0.0)
        # Off-diagonal sum = estimated noise rate
        noise_rate = float(noise_matrix.sum() - np.diag(noise_matrix).sum())
        return noise_matrix, max(0.0, noise_rate)

    def _classify_error_type(self, confidence: float, noise_rate: float) -> str:
        """Классифицировать тип ошибки по силе сигнала.
        Classify error type by signal strength.
        """
        if confidence >= 0.9:
            return "confident_disagreement"
        if noise_rate > 0.10:
            return "high_noise_group"
        return "off_diagonal"

    def find_label_errors(
        self,
        labels: list[int] | np.ndarray,
        pred_probs: list[list[float]] | np.ndarray,
        groups: list[str] | np.ndarray | None = None,
    ) -> LabelQualityReport:
        """
        Основной метод: обнаружить ошибки разметки с декуплированием по группам.
        Main method: detect label errors with group-decoupled noise modeling.

        Args:
            labels: noisy labels ỹ_i ∈ {0, ..., K-1}, shape [n].
            pred_probs: out-of-fold predicted probabilities P̂(Y=j|x_i), shape [n, K].
                        Критично: должны быть получены через cross-validation на train set,
                        а не предсказаны той же моделью, что обучалась на всех данных.
                        Critical: must be obtained via cross-validation, NOT from a model
                        trained on the full train set (would lead to false positives on the
                        examples the model already memorized).
            groups: опциональные метки подгрупп shape [n] (str).
                    Optional subgroup membership labels [n] (str).
                    None → single global noise model (standard CL behavior).

        Returns:
            LabelQualityReport отсортированный по убыванию confidence ошибки.
            LabelQualityReport sorted by descending confidence of error.
        """
        labels_arr = np.asarray(labels, dtype=int)
        probs_arr = np.asarray(pred_probs, dtype=float)
        n = len(labels_arr)
        n_classes = probs_arr.shape[1]

        if groups is None:
            groups_arr = np.array(["all"] * n, dtype=object)
        else:
            groups_arr = np.asarray(groups, dtype=object)
        unique_groups = sorted(set(groups_arr.tolist()))

        all_errors: list[LabelError] = []
        noise_matrices: list[NoiseMatrix] = []

        for group_name in unique_groups:
            mask = groups_arr == group_name
            global_indices = np.where(mask)[0]
            g_labels = labels_arr[mask]
            g_probs = probs_arr[mask]

            # Need at least 2 examples per class to estimate thresholds reliably
            if len(g_labels) < n_classes * 2:
                continue

            noise_mat, noise_rate = self._estimate_noise_matrix(g_labels, g_probs)
            noise_matrices.append(
                NoiseMatrix(
                    group=str(group_name),
                    n_examples=int(mask.sum()),
                    matrix=noise_mat,
                    noise_rate=noise_rate,
                )
            )

            thresholds = self._compute_thresholds(g_labels, g_probs)

            for local_i in range(len(g_labels)):
                s = g_labels[local_i]
                probs = g_probs[local_i]
                predicted_class = int(np.argmax(probs))

                # Error: model is confident in a DIFFERENT class than the given label
                if predicted_class != s and probs[predicted_class] >= thresholds[predicted_class]:
                    confidence = float(probs[predicted_class])
                    all_errors.append(
                        LabelError(
                            index=int(global_indices[local_i]),
                            given_label=int(s),
                            suggested_label=predicted_class,
                            confidence=confidence,
                            group=str(group_name) if groups is not None else None,
                            error_type=self._classify_error_type(confidence, noise_rate),
                        )
                    )

        all_errors.sort(key=lambda e: e.confidence, reverse=True)

        return LabelQualityReport(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            n_examples=n,
            n_classes=n_classes,
            n_groups=len(unique_groups),
            n_errors_found=len(all_errors),
            error_rate=round(len(all_errors) / max(n, 1), 4),
            errors=all_errors,
            noise_matrices=noise_matrices,
        )
