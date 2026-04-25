"""
FastAPI orchestration service: unified multi-model risk scoring.

Endpoint POST /orchestrate принимает PipelineEvent в JSON,
прогоняет через три модели и возвращает PipelineResult.

Используется как единая точка входа для:
- Telecom CRM (churn + fraud screen)
- Fintech anti-fraud gate
- SRE observability (anomaly context в риск-репорте)
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from ..models import (
    CustomerData,
    MetricSnapshot,
    PipelineEvent,
    TransactionData,
)
from ..pipeline import OrchestrationPipeline

app = FastAPI(
    title="Multi-Model Orchestration API",
    description=(
        "Unified risk scoring pipeline: "
        "Churn (Project 01) → Fraud (Project 04) → Anomaly (Project 05)"
    ),
    version="1.0.0",
)

# Singleton pipeline instance — predictors инициализируются один раз при старте
_pipeline = OrchestrationPipeline()


# ──────────────────────────────────────────
# Pydantic request/response models
# ──────────────────────────────────────────


class CustomerRequest(BaseModel):
    customer_id: str = Field(..., examples=["C-12345"])
    tenure: int = Field(..., ge=0, examples=[6])
    monthly_charges: float = Field(..., ge=0, examples=[85.5])
    total_charges: float = Field(..., ge=0, examples=[513.0])
    contract: str = Field(..., examples=["Month-to-month"])
    internet_service: str = Field(..., examples=["Fiber optic"])
    payment_method: str = Field(default="Electronic check")


class TransactionRequest(BaseModel):
    avg_amount: float = Field(..., ge=0, examples=[2500.0])
    n_transactions: int = Field(..., ge=0, examples=[18])
    account_age_days: float = Field(..., ge=0, examples=[25.0])


class MetricRequest(BaseModel):
    cpu: list[float] = Field(..., min_length=1, examples=[[20.0] * 15 + [95.0]])
    latency: list[float] = Field(..., min_length=1, examples=[[50.0] * 15 + [50.0]])
    requests: list[float] = Field(..., min_length=1, examples=[[100.0] * 15 + [100.0]])


class OrchestrationRequest(BaseModel):
    customer: CustomerRequest
    transaction: TransactionRequest
    metrics: MetricRequest


class ChurnResponse(BaseModel):
    churn_probability: float
    is_high_risk: bool


class FraudResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    max_score: float
    affected_metrics: list[str]


class RiskResponse(BaseModel):
    combined_score: float
    action: str
    reasons: list[str]


class OrchestrationResponse(BaseModel):
    event_id: str
    timestamp: str
    churn: ChurnResponse
    fraud: FraudResponse
    anomaly: AnomalyResponse
    risk: RiskResponse
    processing_ms: float


# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    """Service health check."""
    return {"status": "healthy", "version": "1.0.0", "models": ["churn", "fraud", "anomaly"]}


@app.post("/orchestrate", response_model=OrchestrationResponse)
def orchestrate(request: OrchestrationRequest) -> OrchestrationResponse:
    """Run multi-model orchestration pipeline for a customer event.

    Последовательно запускает churn → fraud → anomaly predictors
    и возвращает unified risk profile с action recommendation.
    """
    event = PipelineEvent(
        customer=CustomerData(
            customer_id=request.customer.customer_id,
            tenure=request.customer.tenure,
            monthly_charges=request.customer.monthly_charges,
            total_charges=request.customer.total_charges,
            contract=request.customer.contract,
            internet_service=request.customer.internet_service,
            payment_method=request.customer.payment_method,
        ),
        transaction=TransactionData(
            avg_amount=request.transaction.avg_amount,
            n_transactions=request.transaction.n_transactions,
            account_age_days=request.transaction.account_age_days,
        ),
        metrics=MetricSnapshot(
            cpu=request.metrics.cpu,
            latency=request.metrics.latency,
            requests=request.metrics.requests,
        ),
    )

    result = _pipeline.run(event)

    return OrchestrationResponse(
        event_id=result.event_id,
        timestamp=result.timestamp,
        churn=ChurnResponse(
            churn_probability=result.churn.churn_probability,
            is_high_risk=result.churn.is_high_risk,
        ),
        fraud=FraudResponse(
            fraud_probability=result.fraud.fraud_probability,
            is_fraud=result.fraud.is_fraud,
            risk_level=result.fraud.risk_level,
        ),
        anomaly=AnomalyResponse(
            is_anomaly=result.anomaly.is_anomaly,
            max_score=result.anomaly.max_score,
            affected_metrics=result.anomaly.affected_metrics,
        ),
        risk=RiskResponse(
            combined_score=result.risk.combined_score,
            action=result.risk.action,
            reasons=result.risk.reasons,
        ),
        processing_ms=result.processing_ms,
    )


@app.post("/orchestrate/batch")
def orchestrate_batch(requests: list[OrchestrationRequest]) -> list[OrchestrationResponse]:
    """Process a batch of customer events through the pipeline."""
    events = [
        PipelineEvent(
            customer=CustomerData(
                customer_id=req.customer.customer_id,
                tenure=req.customer.tenure,
                monthly_charges=req.customer.monthly_charges,
                total_charges=req.customer.total_charges,
                contract=req.customer.contract,
                internet_service=req.customer.internet_service,
                payment_method=req.customer.payment_method,
            ),
            transaction=TransactionData(
                avg_amount=req.transaction.avg_amount,
                n_transactions=req.transaction.n_transactions,
                account_age_days=req.transaction.account_age_days,
            ),
            metrics=MetricSnapshot(
                cpu=req.metrics.cpu,
                latency=req.metrics.latency,
                requests=req.metrics.requests,
            ),
        )
        for req in requests
    ]

    results = _pipeline.run_batch(events)

    return [
        OrchestrationResponse(
            event_id=r.event_id,
            timestamp=r.timestamp,
            churn=ChurnResponse(
                churn_probability=r.churn.churn_probability,
                is_high_risk=r.churn.is_high_risk,
            ),
            fraud=FraudResponse(
                fraud_probability=r.fraud.fraud_probability,
                is_fraud=r.fraud.is_fraud,
                risk_level=r.fraud.risk_level,
            ),
            anomaly=AnomalyResponse(
                is_anomaly=r.anomaly.is_anomaly,
                max_score=r.anomaly.max_score,
                affected_metrics=r.anomaly.affected_metrics,
            ),
            risk=RiskResponse(
                combined_score=r.risk.combined_score,
                action=r.risk.action,
                reasons=r.risk.reasons,
            ),
            processing_ms=r.processing_ms,
        )
        for r in results
    ]
