"""
OWASP ML Security Top 10 Audit Engine / Аудит безопасности ML-систем.

Реализует проверки по OWASP Machine Learning Security Top 10:
https://owasp.org/www-project-machine-learning-security-top-10/

Implements checks based on the OWASP Machine Learning Security Top 10.
All checks are stateless and require no external dependencies beyond numpy/polars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class OWASPRisk(StrEnum):
    """OWASP ML Security Top 10 risks."""

    ML01 = "ML01"  # Input Manipulation Attack (adversarial)
    ML02 = "ML02"  # Data Poisoning Attack
    ML03 = "ML03"  # Model Inversion Attack
    ML04 = "ML04"  # Membership Inference Attack
    ML05 = "ML05"  # Model Theft
    ML06 = "ML06"  # AI Supply Chain Attack
    ML07 = "ML07"  # Transfer Learning Attack
    ML08 = "ML08"  # Model Skewing
    ML09 = "ML09"  # Output Integrity Attack
    ML10 = "ML10"  # Model Poisoning


class RiskSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


_SEVERITY_SCORE: dict[RiskSeverity, int] = {
    RiskSeverity.CRITICAL: 40,
    RiskSeverity.HIGH: 20,
    RiskSeverity.MEDIUM: 10,
    RiskSeverity.LOW: 5,
    RiskSeverity.INFO: 0,
}


@dataclass
class SecurityFinding:
    """Single OWASP ML security finding."""

    risk_id: OWASPRisk
    severity: RiskSeverity
    title: str
    description: str
    recommendation: str
    affected_features: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Full OWASP ML audit result / Полный отчёт аудита безопасности."""

    findings: list[SecurityFinding]
    score: int  # 0-100, higher is better (fewer issues)
    passed_checks: int
    total_checks: int
    high_risk_count: int
    risk_summary: dict[str, int]  # severity → count

    @property
    def passed(self) -> bool:
        """Audit passed if no critical or high findings."""
        return self.high_risk_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "passed_checks": self.passed_checks,
            "total_checks": self.total_checks,
            "high_risk_count": self.high_risk_count,
            "risk_summary": self.risk_summary,
            "findings": [
                {
                    "risk_id": f.risk_id.value,
                    "severity": f.severity.value,
                    "title": f.title,
                    "description": f.description,
                    "recommendation": f.recommendation,
                    "affected_features": f.affected_features,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
        }


def _iqr_outlier_ratio(values: list[float]) -> float:
    """Доля выбросов по правилу 1.5*IQR / Outlier fraction using IQR rule."""
    if len(values) < 4:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1 = sorted_vals[n // 4]
    q3 = sorted_vals[3 * n // 4]
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    fence_lo = q1 - 1.5 * iqr
    fence_hi = q3 + 1.5 * iqr
    outliers = sum(1 for v in values if v < fence_lo or v > fence_hi)
    return outliers / len(values)


def _label_entropy(labels: list[Any]) -> float:
    """
    Shannon entropy нормализованная по log2(num_classes).
    Shannon entropy normalised to [0, 1]; 1.0 = perfectly balanced.
    """
    if not labels:
        return 1.0
    counts: dict[Any, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    total = len(labels)
    num_classes = len(counts)
    if num_classes <= 1:
        return 0.0
    import math

    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(num_classes)
    return entropy / max_entropy if max_entropy > 0 else 1.0


class OWASPMLAudit:
    """
    Движок аудита безопасности OWASP ML Top 10.

    OWASP ML Top 10 security audit engine.
    Runs stateless checks on a dataset (represented as column→values dict)
    and optional model metadata to produce a prioritised finding list.
    """

    # Thresholds tuned to common production distributions
    _OUTLIER_THRESHOLD = 0.15  # >15% outliers → potential adversarial injection
    _IMBALANCE_THRESHOLD = 0.3  # label entropy < 0.3 → high class imbalance
    _HIGH_CARDINALITY_RATIO = 0.95  # unique / total > 95% in a label col → data leak risk
    _MISSING_THRESHOLD = 0.3  # >30% missing → data completeness risk

    def check_ml01_input_manipulation(
        self, columns: dict[str, list[float]]
    ) -> SecurityFinding | None:
        """
        ML01: Проверка на манипуляцию входными данными (adversarial inputs).
        Flags numeric columns with unusually high outlier ratios that may
        indicate adversarial perturbation or injection of crafted samples.
        """
        flagged: list[str] = []
        evidence: dict[str, float] = {}
        for col, values in columns.items():
            ratio = _iqr_outlier_ratio(values)
            if ratio > self._OUTLIER_THRESHOLD:
                flagged.append(col)
                evidence[col] = round(ratio, 3)

        if not flagged:
            return None

        return SecurityFinding(
            risk_id=OWASPRisk.ML01,
            severity=RiskSeverity.HIGH,
            title="Potential adversarial inputs detected",
            description=(
                f"Columns {flagged} contain >{self._OUTLIER_THRESHOLD * 100:.0f}% statistical "
                "outliers. Adversarial examples often exploit boundary regions with extreme values."
            ),
            recommendation=(
                "Apply input validation bounds, Winsorization or anomaly pre-filter "
                "before model inference. Consider adversarial training (PGD, FGSM)."
            ),
            affected_features=flagged,
            evidence={"outlier_ratios": evidence},
        )

    def check_ml02_data_poisoning(self, label_column: list[Any] | None) -> SecurityFinding | None:
        """
        ML02: Проверка на отравление данных (data poisoning).
        Severely imbalanced labels can indicate targeted label-flipping attacks
        where an adversary poisons a minority class to degrade model performance.
        """
        if not label_column:
            return None

        entropy = _label_entropy(label_column)
        if entropy >= self._IMBALANCE_THRESHOLD:
            return None

        counts: dict[Any, int] = {}
        for lbl in label_column:
            counts[lbl] = counts.get(lbl, 0) + 1
        minority_frac = min(counts.values()) / len(label_column)

        return SecurityFinding(
            risk_id=OWASPRisk.ML02,
            severity=RiskSeverity.HIGH,
            title="Severe class imbalance — possible label poisoning",
            description=(
                f"Label entropy = {entropy:.3f} (threshold {self._IMBALANCE_THRESHOLD}). "
                f"Minority class fraction = {minority_frac:.3f}. "
                "Label-flipping attacks target minority classes to degrade recall."
            ),
            recommendation=(
                "Audit label provenance. Apply SMOTE/class-weighting. "
                "Monitor per-class metrics in production to detect gradual poisoning."
            ),
            affected_features=["label"],
            evidence={"label_entropy": round(entropy, 3), "class_counts": counts},
        )

    def check_ml03_model_inversion(self, output_fields: list[str]) -> SecurityFinding | None:
        """
        ML03: Проверка на инверсию модели (model inversion / data reconstruction).
        Detects if the API exposes intermediate representations or raw logits
        that enable reconstruction of training data.
        """
        risky_fields = [
            f
            for f in output_fields
            if any(
                kw in f.lower()
                for kw in (
                    "logit",
                    "embedding",
                    "hidden",
                    "raw_score",
                    "feature_vector",
                    "internal",
                )
            )
        ]
        if not risky_fields:
            return None

        return SecurityFinding(
            risk_id=OWASPRisk.ML03,
            severity=RiskSeverity.MEDIUM,
            title="Model outputs expose internal representations",
            description=(
                f"Output fields {risky_fields} may expose logits or embeddings "
                "that enable model inversion attacks to reconstruct training samples."
            ),
            recommendation=(
                "Return only final predictions and confidence scores. "
                "Apply output perturbation (Rényi DP) if embeddings must be shared."
            ),
            affected_features=risky_fields,
            evidence={"risky_output_fields": risky_fields},
        )

    def check_ml04_membership_inference(
        self, columns: dict[str, list[Any]]
    ) -> SecurityFinding | None:
        """
        ML04: Проверка на атаку вывода принадлежности (membership inference).
        High-cardinality columns (near-unique IDs) in training data increase
        the risk that an adversary can determine if a record was used for training.
        """
        high_cardinality: list[str] = []
        evidence: dict[str, float] = {}
        for col, values in columns.items():
            if not values:
                continue
            unique_ratio = len(set(str(v) for v in values)) / len(values)
            if unique_ratio > self._HIGH_CARDINALITY_RATIO:
                high_cardinality.append(col)
                evidence[col] = round(unique_ratio, 3)

        if not high_cardinality:
            return None

        return SecurityFinding(
            risk_id=OWASPRisk.ML04,
            severity=RiskSeverity.MEDIUM,
            title="High-cardinality columns increase membership inference risk",
            description=(
                f"Columns {high_cardinality} have >95% unique values. "
                "Including quasi-identifiers in training sets enables shadow-model attacks."
            ),
            recommendation=(
                "Remove or hash high-cardinality identifiers before training. "
                "Apply k-anonymity or differential privacy to training datasets."
            ),
            affected_features=high_cardinality,
            evidence={"unique_ratios": evidence},
        )

    def check_ml05_model_theft(
        self, has_rate_limiting: bool, exposes_probabilities: bool
    ) -> SecurityFinding | None:
        """
        ML05: Проверка на кражу модели (model extraction / model theft).
        APIs without rate limiting and full probability vectors are
        most vulnerable to model extraction via query-based cloning.
        """
        if has_rate_limiting:
            return None

        severity = RiskSeverity.HIGH if exposes_probabilities else RiskSeverity.MEDIUM
        return SecurityFinding(
            risk_id=OWASPRisk.ML05,
            severity=severity,
            title="API lacks rate limiting — model theft risk",
            description=(
                "No rate limiting detected. "
                + (
                    "Full probability vectors exposed, enabling label-only + confidence extraction."
                    if exposes_probabilities
                    else "Prediction endpoints are unrestricted."
                )
            ),
            recommendation=(
                "Implement per-IP/per-token rate limiting (e.g. 100 req/min). "
                "Return top-1 label only, or add output rounding (2 decimal places). "
                "Consider query budgets tracked in Redis."
            ),
            affected_features=[],
            evidence={
                "has_rate_limiting": has_rate_limiting,
                "exposes_probabilities": exposes_probabilities,
            },
        )

    def check_ml08_model_skewing(self, columns: dict[str, list[Any]]) -> SecurityFinding | None:
        """
        ML08: Проверка на скручивание модели (model skewing via missing data).
        High missing-value rates in inference data may indicate deliberate
        omission attacks designed to push samples into misclassified regions.
        """
        high_missing: list[str] = []
        evidence: dict[str, float] = {}
        for col, values in columns.items():
            if not values:
                continue
            missing_rate = sum(1 for v in values if v is None or str(v).strip() == "") / len(values)
            if missing_rate > self._MISSING_THRESHOLD:
                high_missing.append(col)
                evidence[col] = round(missing_rate, 3)

        if not high_missing:
            return None

        return SecurityFinding(
            risk_id=OWASPRisk.ML08,
            severity=RiskSeverity.LOW,
            title="High missing-value rate — potential model skewing",
            description=(
                f"Columns {high_missing} have >{self._MISSING_THRESHOLD * 100:.0f}% missing values. "
                "Adversaries may deliberately omit features to manipulate model outputs."
            ),
            recommendation=(
                "Enforce NOT NULL constraints on critical features at inference time. "
                "Use robust imputation with monitored fallback values."
            ),
            affected_features=high_missing,
            evidence={"missing_rates": evidence},
        )

    def check_ml09_output_integrity(self, output_fields: list[str]) -> SecurityFinding | None:
        """
        ML09: Проверка целостности выходных данных (output integrity).
        Checks whether the model output schema includes a signature or
        integrity token to prevent tampering in transit.
        """
        has_integrity = any(
            kw in " ".join(output_fields).lower()
            for kw in ("signature", "hash", "token", "hmac", "digest")
        )
        if has_integrity:
            return None

        return SecurityFinding(
            risk_id=OWASPRisk.ML09,
            severity=RiskSeverity.LOW,
            title="Model outputs lack integrity signature",
            description=(
                "No output signature/HMAC field detected. "
                "Predictions in transit can be tampered by MITM or compromised middleware."
            ),
            recommendation=(
                "Sign model outputs with HMAC-SHA256 using a service secret. "
                "Include prediction_id + timestamp in the signed payload."
            ),
            affected_features=[],
            evidence={"output_fields": output_fields},
        )

    def run_audit(
        self,
        *,
        numeric_columns: dict[str, list[float]] | None = None,
        all_columns: dict[str, list[Any]] | None = None,
        label_column: list[Any] | None = None,
        output_fields: list[str] | None = None,
        has_rate_limiting: bool = False,
        exposes_probabilities: bool = True,
    ) -> AuditReport:
        """
        Запустить полный аудит OWASP ML Top 10.

        Run a full OWASP ML Top 10 audit.
        Pass only the data you have — checks auto-skip if inputs are None/empty.
        """
        numeric_columns = numeric_columns or {}
        all_columns = all_columns or {}
        output_fields = output_fields or []

        findings: list[SecurityFinding] = []
        total_checks = 0

        checks = [
            (self.check_ml01_input_manipulation, (numeric_columns,), {}),
            (self.check_ml02_data_poisoning, (label_column,), {}),
            (self.check_ml03_model_inversion, (output_fields,), {}),
            (self.check_ml04_membership_inference, (all_columns,), {}),
            (
                self.check_ml05_model_theft,
                (),
                {
                    "has_rate_limiting": has_rate_limiting,
                    "exposes_probabilities": exposes_probabilities,
                },
            ),
            (self.check_ml08_model_skewing, (all_columns,), {}),
            (self.check_ml09_output_integrity, (output_fields,), {}),
        ]

        for check_fn, args, kwargs in checks:
            total_checks += 1
            result = check_fn(*args, **kwargs)  # type: ignore[call-arg]
            if result is not None:
                findings.append(result)

        # Score = 100 − penalty, clipped to [0, 100]
        penalty = sum(_SEVERITY_SCORE[f.severity] for f in findings)
        score = max(0, 100 - penalty)

        risk_summary: dict[str, int] = {s.value: 0 for s in RiskSeverity}
        for f in findings:
            risk_summary[f.severity.value] += 1

        high_risk_count = (
            risk_summary[RiskSeverity.CRITICAL.value] + risk_summary[RiskSeverity.HIGH.value]
        )

        return AuditReport(
            findings=findings,
            score=score,
            passed_checks=total_checks - len(findings),
            total_checks=total_checks,
            high_risk_count=high_risk_count,
            risk_summary=risk_summary,
        )
