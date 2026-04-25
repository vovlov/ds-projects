"""
Risk scoring: объединяет сигналы от churn, fraud, anomaly в единый risk profile.

Бизнес-логика для telecom/fintech оператора:
- fraud (55%): немедленные финансовые потери — наибольший вес
- churn (30%): потери revenue за 6-12 месяцев
- anomaly (15%): системный индикатор (SRE-сигнал)

Action ladder:
  block      → fraud is_fraud=True  (немедленная блокировка транзакции)
  review     → high combined risk ≥ 0.70 или fraud risk_level=high
  intervene  → churn is_high_risk=True (retention campaign)
  monitor    → anomaly detected (SRE alert)
  ok         → all signals low
"""

from __future__ import annotations

from .models import AnomalyResult, ChurnResult, FraudResult, RiskProfile

# Пороги для action ladder (откалиброваны по P&L impact в telecom)
_COMBINED_REVIEW_THRESHOLD = 0.70
_COMBINED_MONITOR_THRESHOLD = 0.30

# Веса для combined score
_W_FRAUD = 0.55
_W_CHURN = 0.30
_W_ANOMALY = 0.15


def compute_risk(
    churn: ChurnResult,
    fraud: FraudResult,
    anomaly: AnomalyResult,
) -> RiskProfile:
    """Compute unified risk profile from three model outputs.

    Использует иерархический action ladder: более тяжёлые риски имеют приоритет.
    Fraud блокирует всегда, независимо от churn/anomaly сигналов.

    Args:
        churn: Output from ChurnPredictor.
        fraud: Output from FraudPredictor.
        anomaly: Output from AnomalyPredictor.

    Returns:
        RiskProfile with combined score, action, and reasons.
    """
    # Anomaly score нормализован к [0, 1] через threshold (3σ = полный балл)
    anomaly_norm = min(1.0, anomaly.max_score / 3.0) if anomaly.max_score > 0 else 0.0

    combined = (
        _W_FRAUD * fraud.fraud_probability
        + _W_CHURN * churn.churn_probability
        + _W_ANOMALY * anomaly_norm
    )

    reasons: list[str] = []
    action: str

    if fraud.is_fraud:
        # Fraud → немедленная блокировка (compliance требование)
        action = "block"
        reasons.append(f"fraud_detected p={fraud.fraud_probability:.3f}")
    elif fraud.risk_level == "high" or combined >= _COMBINED_REVIEW_THRESHOLD:
        action = "review"
        reasons.append(f"combined_risk={combined:.3f}")
        if fraud.risk_level == "high":
            reasons.append(f"fraud_high_risk p={fraud.fraud_probability:.3f}")
    elif churn.is_high_risk:
        action = "intervene"
        reasons.append(f"churn_risk p={churn.churn_probability:.3f}")
        if anomaly.is_anomaly:
            reasons.append(f"anomaly_metrics={','.join(anomaly.affected_metrics)}")
    elif anomaly.is_anomaly:
        action = "monitor"
        reasons.append(f"anomaly_detected metrics={','.join(anomaly.affected_metrics)}")
    else:
        action = "ok"

    return RiskProfile(
        combined_score=round(combined, 4),
        action=action,
        reasons=reasons,
    )
