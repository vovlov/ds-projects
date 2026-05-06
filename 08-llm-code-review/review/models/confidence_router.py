"""Confidence-based routing for automated code review decisions.

Implements Human-in-the-Loop (HITL) routing pattern where review results are
automatically triaged into one of three paths based on risk score and confidence:

- AUTO_APPROVE  : no significant issues, safe to merge without human review
- HUMAN_REVIEW  : mixed signals or medium confidence, requires human decision
- AUTO_REJECT   : critical/security vulnerabilities with high confidence, block merge

Inspired by:
- LLM Code Review Evaluation, 2025 (arxiv 2505.20206): hybrid HITL achieves
  68%+ accuracy while reducing manual burden by ~50%
- OWASP ML Security Top 10: always escalate critical security findings
- Ericsson Production Code Review 2025: confidence thresholds for routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RoutingDecision(str, Enum):
    """Routing decision for a reviewed PR/diff."""

    AUTO_APPROVE = "auto_approve"
    HUMAN_REVIEW = "human_review"
    AUTO_REJECT = "auto_reject"


# Severity weights: controls how much each level contributes to risk score.
# Critical is weighted 10× suggestion to reflect OWASP escalation policy.
_SEVERITY_WEIGHT: dict[str, float] = {
    "critical": 10.0,
    "major": 4.0,
    "minor": 1.0,
    "suggestion": 0.3,
}

# Categories that trigger immediate escalation regardless of confidence.
_ESCALATION_CATEGORIES = frozenset({"security"})


@dataclass
class RoutingConfig:
    """Thresholds that control routing decisions.

    Attributes:
        auto_approve_max_risk:  risk score at or below which AUTO_APPROVE fires.
        auto_reject_min_risk:   risk score at or above which AUTO_REJECT fires.
        critical_escalate:      if True, any 'critical' severity auto-rejects.
        security_escalate:      if True, any 'security' category auto-rejects.
    """

    auto_approve_max_risk: float = 0.5
    auto_reject_min_risk: float = 8.0
    critical_escalate: bool = True
    security_escalate: bool = True


@dataclass
class RoutingResult:
    """Result of routing a set of review comments.

    Attributes:
        decision:           final routing decision.
        risk_score:         numeric risk score computed from comments.
        confidence:         routing confidence in [0.0, 1.0].
        reason:             human-readable explanation of the decision.
        critical_findings:  list of critical/security comments that drove rejection.
        comment_count:      total number of review comments processed.
    """

    decision: RoutingDecision
    risk_score: float
    confidence: float
    reason: str
    critical_findings: list[dict] = field(default_factory=list)
    comment_count: int = 0


def compute_risk_score(comments: list[dict]) -> float:
    """Compute aggregate risk score from a list of review comments.

    Score = sum of severity weights for all comments, capped at 100.0.
    Empty comment list → 0.0 (clean diff).

    Args:
        comments: list of dicts with at least a 'severity' key.

    Returns:
        Risk score in [0.0, 100.0].
    """
    if not comments:
        return 0.0

    total = 0.0
    for c in comments:
        severity = str(c.get("severity", "suggestion")).lower()
        total += _SEVERITY_WEIGHT.get(severity, 0.3)

    return min(total, 100.0)


def _collect_critical_findings(comments: list[dict], config: RoutingConfig) -> list[dict]:
    """Return comments that warrant immediate escalation."""
    findings = []
    for c in comments:
        severity = str(c.get("severity", "")).lower()
        category = str(c.get("category", "")).lower()

        is_critical_severity = config.critical_escalate and severity == "critical"
        is_security_category = config.security_escalate and category in _ESCALATION_CATEGORIES

        if is_critical_severity or is_security_category:
            findings.append(c)

    return findings


def _routing_confidence(risk_score: float, config: RoutingConfig) -> float:
    """Estimate confidence of the routing decision in [0.0, 1.0].

    Confidence is highest near the extremes (clearly safe or clearly risky)
    and lowest in the middle zone where human judgement is most useful.
    Uses a piecewise linear function to avoid arbitrary sigmoid tuning.
    """
    low = config.auto_approve_max_risk
    high = config.auto_reject_min_risk

    if risk_score <= low:
        # Clear safe zone: confidence scales 0.5 → 1.0 as score → 0
        return 0.5 + 0.5 * (1.0 - risk_score / max(low, 1e-9))
    if risk_score >= high:
        # Clear danger zone: confidence scales 0.5 → 1.0 as score → ∞
        excess = risk_score - high
        return min(1.0, 0.5 + 0.5 * excess / max(high, 1.0))

    # Middle zone: confidence inversely proportional to distance from midpoint
    mid = (low + high) / 2.0
    half_width = (high - low) / 2.0
    distance_from_mid = abs(risk_score - mid)
    # At midpoint → 0.1 confidence; at edges of zone → approaches 0.5
    return 0.1 + 0.4 * (distance_from_mid / max(half_width, 1e-9))


def route_review(
    comments: list[dict],
    config: Optional[RoutingConfig] = None,
) -> RoutingResult:
    """Route review results to AUTO_APPROVE / HUMAN_REVIEW / AUTO_REJECT.

    Decision logic (in priority order):
    1. Any critical or security finding → AUTO_REJECT (if escalation enabled)
    2. risk_score >= auto_reject_min_risk → AUTO_REJECT
    3. risk_score <= auto_approve_max_risk → AUTO_APPROVE
    4. Otherwise → HUMAN_REVIEW

    Args:
        comments: list of review comment dicts (severity, category, comment, line).
        config:   routing thresholds; uses defaults if None.

    Returns:
        RoutingResult with decision, risk_score, confidence, and reason.
    """
    if config is None:
        config = RoutingConfig()

    risk_score = compute_risk_score(comments)
    critical_findings = _collect_critical_findings(comments, config)
    confidence = _routing_confidence(risk_score, config)

    # Priority 1: escalation findings always block the merge
    if critical_findings:
        severities = [f.get("severity", "?") for f in critical_findings]
        categories = [f.get("category", "?") for f in critical_findings]
        reason = (
            f"Escalation triggered: {len(critical_findings)} critical finding(s) "
            f"(severity={severities}, category={categories}). "
            "Immediate human review required per OWASP ML Security policy."
        )
        return RoutingResult(
            decision=RoutingDecision.AUTO_REJECT,
            risk_score=risk_score,
            confidence=min(1.0, confidence + 0.3),  # boost confidence for clear escalations
            reason=reason,
            critical_findings=critical_findings,
            comment_count=len(comments),
        )

    # Priority 2: high aggregate risk
    if risk_score >= config.auto_reject_min_risk:
        reason = (
            f"High aggregate risk score ({risk_score:.1f} ≥ {config.auto_reject_min_risk}). "
            f"Found {len(comments)} issue(s). Blocking merge, human review required."
        )
        return RoutingResult(
            decision=RoutingDecision.AUTO_REJECT,
            risk_score=risk_score,
            confidence=confidence,
            reason=reason,
            critical_findings=[],
            comment_count=len(comments),
        )

    # Priority 3: low risk, auto-approve
    if risk_score <= config.auto_approve_max_risk:
        if not comments:
            reason = "No review comments found. Diff appears clean — auto-approving."
        else:
            reason = (
                f"Low risk score ({risk_score:.1f} ≤ {config.auto_approve_max_risk}). "
                f"Only minor/suggestion issues found ({len(comments)} comment(s))."
            )
        return RoutingResult(
            decision=RoutingDecision.AUTO_APPROVE,
            risk_score=risk_score,
            confidence=confidence,
            reason=reason,
            critical_findings=[],
            comment_count=len(comments),
        )

    # Priority 4: ambiguous zone → human review
    reason = (
        f"Risk score ({risk_score:.1f}) is in the ambiguous zone "
        f"[{config.auto_approve_max_risk}, {config.auto_reject_min_risk}]. "
        f"Found {len(comments)} issue(s). Routing to human reviewer."
    )
    return RoutingResult(
        decision=RoutingDecision.HUMAN_REVIEW,
        risk_score=risk_score,
        confidence=confidence,
        reason=reason,
        critical_findings=[],
        comment_count=len(comments),
    )
