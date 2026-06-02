"""Fair ML / Bias Detection for customer churn prediction.

Fairness Analysis and Bias Detection for Production ML Models.

Raw accuracy metrics hide systematic unfairness: a model may achieve 90%
accuracy overall while being 20% less accurate for female customers or
senior citizens. EU AI Act Article 9(7) requires demonstrating "appropriate
measures" against bias for high-risk AI systems.

This module implements four complementary fairness metrics:

    Demographic Parity:   P(ŷ=1 | A=0) ≈ P(ŷ=1 | A=1)
                          Equal prediction rates regardless of protected attribute.

    Equal Opportunity:    TPR(A=0) ≈ TPR(A=1)
                          Equal recall for positive class — churners identified at
                          the same rate regardless of protected group.

    Equalized Odds:       TPR(A=0) ≈ TPR(A=1) AND FPR(A=0) ≈ FPR(A=1)
                          Both error rates equal — strongest guarantee.

    Predictive Parity:    PPV(A=0) ≈ PPV(A=1)
                          Predicted probability means the same thing for all groups.

The 80% Rule (disparate impact): ratio of positive-rate group_b / group_a < 0.8
signals illegal discrimination under US EEOC / EU GDPR.

Post-processing mitigation: optimal_thresholds() finds per-group thresholds that
equalize TPR without retraining — cheapest production fix.

Источники:
- Hardt et al. 2016 "Equality of Opportunity in Supervised Learning" (NIPS)
- Feldman et al. 2015 "Certifying and Removing Disparate Impact" (KDD)
- Chouldechova 2017 "Fair Prediction with Disparate Impact" (Big Data)
- EU AI Act Article 9(7) и 10(2)(f) — risk management & data bias requirements
- EEOC Uniform Guidelines 1978 — 80% rule (disparate impact ratio)
- BiasGuard arxiv 2501.04142 — post-processing без переобучения
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np


class FairnessSeverity(StrEnum):
    """Severity level of detected bias.

    Уровень серьёзности обнаруженного смещения.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GroupMetrics:
    """Per-group classification metrics.

    Метрики классификации для отдельной защищённой группы.
    """

    sample_size: int
    positive_rate: float  # P(ŷ=1 | group)
    tpr: float  # True Positive Rate (recall) — NaN если нет положительных истинных
    fpr: float  # False Positive Rate — NaN если нет отрицательных истинных
    ppv: float  # Positive Predictive Value (precision) — NaN если нет предсказанных 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, handling NaN for JSON serialization."""
        return {
            "sample_size": self.sample_size,
            "positive_rate": round(self.positive_rate, 4),
            "tpr": None if np.isnan(self.tpr) else round(self.tpr, 4),
            "fpr": None if np.isnan(self.fpr) else round(self.fpr, 4),
            "ppv": None if np.isnan(self.ppv) else round(self.ppv, 4),
        }


@dataclass
class FairnessMetrics:
    """Fairness metrics comparing two demographic groups.

    Метрики справедливости, сравнивающие две демографические группы.
    group_a = privileged group, group_b = unprivileged group.
    """

    demographic_parity_diff: float  # |P(ŷ=1|A) - P(ŷ=1|B)|
    equal_opportunity_diff: float  # |TPR(A) - TPR(B)|
    equalized_odds_diff: float  # max(|TPR diff|, |FPR diff|)
    predictive_parity_diff: float  # |PPV(A) - PPV(B)|
    disparate_impact_ratio: float  # min(positive_rate) / max(positive_rate), 80% rule

    def to_dict(self) -> dict[str, Any]:
        """Serialise metrics for API/MLflow logging."""
        return {
            "demographic_parity_diff": round(self.demographic_parity_diff, 4),
            "equal_opportunity_diff": round(self.equal_opportunity_diff, 4),
            "equalized_odds_diff": round(self.equalized_odds_diff, 4),
            "predictive_parity_diff": round(self.predictive_parity_diff, 4),
            "disparate_impact_ratio": round(self.disparate_impact_ratio, 4),
        }


@dataclass
class FairnessReport:
    """Complete fairness audit report for EU AI Act compliance.

    Полный отчёт по аудиту справедливости для соответствия EU AI Act.
    """

    audit_id: str
    timestamp: str
    protected_attribute: str
    group_a_label: str
    group_b_label: str
    metrics: FairnessMetrics
    group_a: GroupMetrics
    group_b: GroupMetrics
    severity: FairnessSeverity
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise full report for API response and audit log."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "protected_attribute": self.protected_attribute,
            "group_a_label": self.group_a_label,
            "group_b_label": self.group_b_label,
            "metrics": self.metrics.to_dict(),
            "group_a": self.group_a.to_dict(),
            "group_b": self.group_b.to_dict(),
            "severity": self.severity.value,
            "recommendations": self.recommendations,
        }


class BiasDetector:
    """Fairness analyzer for binary classification models.

    Анализатор справедливости для бинарных классификаторов.

    Computes group-level fairness metrics and detects discriminatory patterns.
    Supports post-processing threshold optimization to equalize TPR across groups
    without retraining the underlying model (Hardt et al. 2016).

    Usage:
        detector = BiasDetector(protected_attribute="gender")
        report = detector.analyze(y_true, y_pred, groups)
        thresholds = detector.optimal_thresholds(y_true, y_proba, groups)
    """

    # 80% rule threshold from EEOC Uniform Guidelines (1978)
    _DISPARATE_IMPACT_HIGH = 0.80
    # Moderate concern threshold
    _DISPARATE_IMPACT_MEDIUM = 0.90
    # Equal opportunity difference thresholds
    _EOD_HIGH = 0.10
    _EOD_MEDIUM = 0.05

    def __init__(self, protected_attribute: str = "protected_group") -> None:
        self.protected_attribute = protected_attribute

    def analyze(
        self,
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        groups: list[str] | np.ndarray,
        group_a_label: str | None = None,
        group_b_label: str | None = None,
    ) -> FairnessReport:
        """Compute fairness metrics for binary predictions.

        Вычислить метрики справедливости для бинарных предсказаний.

        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Model predictions (0/1)
            groups: Protected attribute values per sample
            group_a_label: Label for group A (auto-detected if None)
            group_b_label: Label for group B (auto-detected if None)

        Returns:
            FairnessReport with metrics, severity, and recommendations
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        groups_arr = np.asarray(groups)

        unique_groups = np.unique(groups_arr)
        if len(unique_groups) < 2:
            raise ValueError(f"Need at least 2 groups for fairness analysis, got: {unique_groups}")

        # Use first two unique groups; in binary protected attributes (male/female)
        # this covers the full population
        label_a = group_a_label or str(unique_groups[0])
        label_b = group_b_label or str(unique_groups[1])

        mask_a = groups_arr == unique_groups[0]
        mask_b = groups_arr == unique_groups[1]

        gm_a = self._compute_group_metrics(y_true_arr, y_pred_arr, mask_a)
        gm_b = self._compute_group_metrics(y_true_arr, y_pred_arr, mask_b)

        metrics = self._compute_fairness_metrics(gm_a, gm_b)
        severity = self._classify_severity(metrics)
        recommendations = self._build_recommendations(metrics, severity)

        return FairnessReport(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            protected_attribute=self.protected_attribute,
            group_a_label=label_a,
            group_b_label=label_b,
            metrics=metrics,
            group_a=gm_a,
            group_b=gm_b,
            severity=severity,
            recommendations=recommendations,
        )

    def analyze_proba(
        self,
        y_true: list[int] | np.ndarray,
        y_proba: list[float] | np.ndarray,
        groups: list[str] | np.ndarray,
        threshold: float = 0.5,
        group_a_label: str | None = None,
        group_b_label: str | None = None,
    ) -> FairnessReport:
        """Analyze fairness using probability scores with a fixed threshold.

        Анализ справедливости на основе вероятностей с фиксированным порогом.
        """
        y_pred = (np.asarray(y_proba) >= threshold).astype(int)
        return self.analyze(y_true, y_pred, groups, group_a_label, group_b_label)

    def optimal_thresholds(
        self,
        y_true: list[int] | np.ndarray,
        y_proba: list[float] | np.ndarray,
        groups: list[str] | np.ndarray,
        target_metric: str = "equal_opportunity",
    ) -> dict[str, float]:
        """Find per-group probability thresholds that equalize TPR across groups.

        Post-processing bias mitigation: adjust decision boundary per group
        instead of retraining (Hardt et al. 2016, BiasGuard 2025).
        Zero model accuracy cost when target_metric="equal_opportunity".

        Найти оптимальные пороги для каждой группы, выравнивающие TPR
        без переобучения модели.

        Args:
            y_true: Ground truth labels
            y_proba: Model probability scores
            groups: Protected attribute values
            target_metric: "equal_opportunity" (equalize TPR) or
                           "demographic_parity" (equalize positive rate)

        Returns:
            dict mapping group label → optimal threshold
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_proba_arr = np.asarray(y_proba, dtype=float)
        groups_arr = np.asarray(groups)

        unique_groups = np.unique(groups_arr)
        thresholds_grid = np.linspace(0.01, 0.99, 99)

        # Compute target metric at each threshold for each group
        group_metrics_by_thresh: dict[str, list[float]] = {}
        for grp in unique_groups:
            mask = groups_arr == grp
            yt = y_true_arr[mask]
            yp = y_proba_arr[mask]
            values = []
            for t in thresholds_grid:
                yhat = (yp >= t).astype(int)
                if target_metric == "equal_opportunity":
                    pos = yt == 1
                    # TPR: ratio of true positives caught
                    tpr = float(np.mean(yhat[pos])) if pos.any() else 0.0
                    values.append(tpr)
                else:  # demographic_parity
                    values.append(float(np.mean(yhat)))
            group_metrics_by_thresh[str(grp)] = values

        # Target = mean across groups (equalise to the overall mean)
        all_vals = np.array(list(group_metrics_by_thresh.values()))  # (n_groups, 99)
        target_val = float(np.mean(all_vals))

        result: dict[str, float] = {}
        for grp in unique_groups:
            vals = np.array(group_metrics_by_thresh[str(grp)])
            # Closest threshold to the target metric value
            best_idx = int(np.argmin(np.abs(vals - target_val)))
            result[str(grp)] = round(float(thresholds_grid[best_idx]), 2)

        return result

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _compute_group_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mask: np.ndarray,
    ) -> GroupMetrics:
        """Compute per-group confusion matrix derived metrics.

        Вычислить метрики на основе матрицы ошибок для одной группы.
        """
        yt = y_true[mask]
        yp = y_pred[mask]
        n = len(yt)

        positive_rate = float(np.mean(yp)) if n > 0 else 0.0

        # TPR: TP / (TP + FN) — only defined if positives exist in ground truth
        pos_mask = yt == 1
        tpr = float(np.mean(yp[pos_mask])) if pos_mask.any() else float("nan")

        # FPR: FP / (FP + TN) — only defined if negatives exist in ground truth
        neg_mask = yt == 0
        fpr = float(np.mean(yp[neg_mask])) if neg_mask.any() else float("nan")

        # PPV (precision): TP / (TP + FP) — only defined if any predicted positive
        pred_pos_mask = yp == 1
        ppv = float(np.mean(yt[pred_pos_mask])) if pred_pos_mask.any() else float("nan")

        return GroupMetrics(
            sample_size=n,
            positive_rate=positive_rate,
            tpr=tpr,
            fpr=fpr,
            ppv=ppv,
        )

    @staticmethod
    def _compute_fairness_metrics(
        gm_a: GroupMetrics,
        gm_b: GroupMetrics,
    ) -> FairnessMetrics:
        """Compute cross-group fairness gap metrics.

        Вычислить метрики разрыва справедливости между группами.
        """
        dp_diff = abs(gm_a.positive_rate - gm_b.positive_rate)

        # Equal opportunity: gap in TPR
        if not (np.isnan(gm_a.tpr) or np.isnan(gm_b.tpr)):
            eo_diff = abs(gm_a.tpr - gm_b.tpr)
        else:
            eo_diff = float("nan")

        # Equalized odds: max of |TPR diff| and |FPR diff|
        tpr_nan = np.isnan(gm_a.tpr) or np.isnan(gm_b.tpr)
        fpr_nan = np.isnan(gm_a.fpr) or np.isnan(gm_b.fpr)
        tpr_diff = float("nan") if tpr_nan else abs(gm_a.tpr - gm_b.tpr)
        fpr_diff = float("nan") if fpr_nan else abs(gm_a.fpr - gm_b.fpr)
        if not np.isnan(tpr_diff) and not np.isnan(fpr_diff):
            eqo_diff = max(tpr_diff, fpr_diff)
        elif not np.isnan(tpr_diff):
            eqo_diff = tpr_diff
        else:
            eqo_diff = fpr_diff

        # Predictive parity: gap in PPV
        if not (np.isnan(gm_a.ppv) or np.isnan(gm_b.ppv)):
            pp_diff = abs(gm_a.ppv - gm_b.ppv)
        else:
            pp_diff = float("nan")

        # Disparate impact ratio: min/max of positive rates (80% rule)
        pr_a, pr_b = gm_a.positive_rate, gm_b.positive_rate
        di_ratio = min(pr_a, pr_b) / max(pr_a, pr_b) if max(pr_a, pr_b) > 0 else 1.0

        return FairnessMetrics(
            demographic_parity_diff=dp_diff,
            equal_opportunity_diff=eo_diff,
            equalized_odds_diff=eqo_diff,
            predictive_parity_diff=pp_diff,
            disparate_impact_ratio=di_ratio,
        )

    def _classify_severity(self, metrics: FairnessMetrics) -> FairnessSeverity:
        """Classify overall bias severity.

        Классифицировать общую степень смещения.

        Rules (in priority order):
        - HIGH: DI ratio < 0.80 (EEOC threshold) OR EOD > 0.10
        - MEDIUM: DI ratio < 0.90 OR EOD > 0.05
        - LOW: otherwise
        """
        di = metrics.disparate_impact_ratio
        eod = metrics.equal_opportunity_diff

        if di < self._DISPARATE_IMPACT_HIGH:
            return FairnessSeverity.HIGH
        if not np.isnan(eod) and eod > self._EOD_HIGH:
            return FairnessSeverity.HIGH
        if di < self._DISPARATE_IMPACT_MEDIUM:
            return FairnessSeverity.MEDIUM
        if not np.isnan(eod) and eod > self._EOD_MEDIUM:
            return FairnessSeverity.MEDIUM
        return FairnessSeverity.LOW

    @staticmethod
    def _build_recommendations(
        metrics: FairnessMetrics,
        severity: FairnessSeverity,
    ) -> list[str]:
        """Generate actionable bias mitigation recommendations.

        Сгенерировать рекомендации по устранению смещения.
        """
        recs: list[str] = []

        if severity == FairnessSeverity.HIGH:
            recs.append(
                "CRITICAL: Disparate impact ratio below 80% EEOC threshold — "
                "review model before production deployment (EU AI Act Article 9)."
            )

        if metrics.disparate_impact_ratio < 0.90:
            recs.append(
                "Use /fairness/thresholds to get per-group decision thresholds "
                "that equalize positive prediction rates (post-processing, no retraining)."
            )

        eod = metrics.equal_opportunity_diff
        if not np.isnan(eod) and eod > 0.05:
            recs.append(
                f"Equal opportunity gap {eod:.1%} > 5% — high-churn customers in one "
                "group are systematically missed. Consider per-group threshold tuning."
            )

        pp_diff = metrics.predictive_parity_diff
        if not np.isnan(pp_diff) and pp_diff > 0.05:
            recs.append(
                f"Predictive parity gap {pp_diff:.1%} — predicted churn probability "
                "has different calibration across groups. Consider Platt scaling per group."
            )

        if severity == FairnessSeverity.LOW:
            recs.append(
                "Fairness metrics within acceptable bounds. "
                "Schedule quarterly re-evaluation for drift monitoring."
            )

        return recs
