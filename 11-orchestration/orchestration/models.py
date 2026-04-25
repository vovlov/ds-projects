"""
Dataclasses для multi-model orchestration pipeline.

Unified event model объединяет входы для churn (Project 01),
fraud (Project 04) и anomaly (Project 05) в единый API.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class CustomerData:
    """Customer features for churn prediction (Project 01).

    Subset of Telco Customer Churn dataset features.
    """

    customer_id: str
    tenure: int  # months
    monthly_charges: float
    total_charges: float
    contract: str  # "Month-to-month" | "One year" | "Two year"
    internet_service: str  # "Fiber optic" | "DSL" | "No"
    payment_method: str = "Electronic check"


@dataclass
class TransactionData:
    """Transaction features for fraud detection (Project 04).

    Compatible with Elliptic Bitcoin dataset feature schema.
    """

    avg_amount: float  # средняя сумма транзакции
    n_transactions: int  # количество транзакций
    account_age_days: float  # возраст аккаунта в днях


@dataclass
class MetricSnapshot:
    """System metrics time series for anomaly detection (Project 05).

    Each list contains the most recent readings (min 10 points recommended).
    """

    cpu: list[float]  # CPU usage %
    latency: list[float]  # request latency ms
    requests: list[float]  # requests per second


@dataclass
class PipelineEvent:
    """Unified event для multi-model orchestration pipeline.

    Один event объединяет данные для всех трёх downstream-моделей.
    Используется в telecom/fintech: transaction event с контекстом клиента и системы.
    """

    customer: CustomerData
    transaction: TransactionData
    metrics: MetricSnapshot
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class ChurnResult:
    """Output of churn prediction stage."""

    churn_probability: float
    is_high_risk: bool


@dataclass
class FraudResult:
    """Output of fraud detection stage."""

    fraud_probability: float
    is_fraud: bool
    risk_level: str  # "low" | "medium" | "high"


@dataclass
class AnomalyResult:
    """Output of anomaly detection stage."""

    is_anomaly: bool
    max_score: float  # максимальный z-score по всем метрикам
    affected_metrics: list[str]  # список метрик с аномалиями


@dataclass
class RiskProfile:
    """Combined risk assessment from all three models.

    Weighted sum: fraud (55%) + churn (30%) + anomaly (15%).
    Веса отражают business impact: fraud = немедленные потери,
    churn = revenue за 6-12 мес, anomaly = системный сигнал.
    """

    combined_score: float  # 0.0–1.0
    action: str  # "block" | "review" | "intervene" | "monitor" | "ok"
    reasons: list[str]


@dataclass
class PipelineResult:
    """Полный результат orchestration pipeline."""

    event_id: str
    timestamp: str
    churn: ChurnResult
    fraud: FraudResult
    anomaly: AnomalyResult
    risk: RiskProfile
    processing_ms: float
