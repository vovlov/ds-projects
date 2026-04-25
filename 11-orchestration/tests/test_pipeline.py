"""Tests for OrchestrationPipeline and risk computation."""

from __future__ import annotations

from orchestration.models import (
    AnomalyResult,
    ChurnResult,
    CustomerData,
    FraudResult,
    MetricSnapshot,
    PipelineEvent,
    TransactionData,
)
from orchestration.pipeline import OrchestrationPipeline
from orchestration.risk import compute_risk

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────


def make_event(
    tenure: int = 12,
    monthly_charges: float = 70.0,
    contract: str = "Month-to-month",
    internet: str = "Fiber optic",
    avg_amount: float = 200.0,
    n_txn: int = 5,
    account_age: float = 180.0,
    cpu: list[float] | None = None,
) -> PipelineEvent:
    return PipelineEvent(
        customer=CustomerData(
            customer_id="C-TEST",
            tenure=tenure,
            monthly_charges=monthly_charges,
            total_charges=tenure * monthly_charges,
            contract=contract,
            internet_service=internet,
        ),
        transaction=TransactionData(
            avg_amount=avg_amount,
            n_transactions=n_txn,
            account_age_days=account_age,
        ),
        metrics=MetricSnapshot(
            cpu=cpu if cpu is not None else [20.0] * 20,
            latency=[50.0] * 20,
            requests=[100.0] * 20,
        ),
    )


# ──────────────────────────────────────────
# TestRiskScorer
# ──────────────────────────────────────────


class TestRiskScorer:
    def test_fraud_action_block(self) -> None:
        churn = ChurnResult(0.3, False)
        fraud = FraudResult(0.8, True, "high")
        anomaly = AnomalyResult(False, 0.0, [])
        risk = compute_risk(churn, fraud, anomaly)
        assert risk.action == "block"

    def test_high_combined_score_review(self) -> None:
        churn = ChurnResult(0.8, True)
        fraud = FraudResult(0.75, True, "high")
        anomaly = AnomalyResult(False, 0.0, [])
        risk = compute_risk(churn, fraud, anomaly)
        # fraud.is_fraud=True → block takes priority
        assert risk.action == "block"

    def test_churn_only_intervene(self) -> None:
        churn = ChurnResult(0.8, True)
        fraud = FraudResult(0.1, False, "low")
        anomaly = AnomalyResult(False, 0.0, [])
        risk = compute_risk(churn, fraud, anomaly)
        assert risk.action == "intervene"

    def test_anomaly_only_monitor(self) -> None:
        churn = ChurnResult(0.1, False)
        fraud = FraudResult(0.1, False, "low")
        anomaly = AnomalyResult(True, 4.5, ["cpu"])
        risk = compute_risk(churn, fraud, anomaly)
        assert risk.action == "monitor"

    def test_all_clear_ok(self) -> None:
        churn = ChurnResult(0.1, False)
        fraud = FraudResult(0.1, False, "low")
        anomaly = AnomalyResult(False, 0.0, [])
        risk = compute_risk(churn, fraud, anomaly)
        assert risk.action == "ok"

    def test_combined_score_in_range(self) -> None:
        churn = ChurnResult(0.5, True)
        fraud = FraudResult(0.5, False, "medium")
        anomaly = AnomalyResult(False, 1.5, [])
        risk = compute_risk(churn, fraud, anomaly)
        assert 0.0 <= risk.combined_score <= 1.0

    def test_fraud_takes_priority_over_churn(self) -> None:
        churn = ChurnResult(0.9, True)
        fraud = FraudResult(0.9, True, "high")
        anomaly = AnomalyResult(True, 5.0, ["cpu", "latency"])
        risk = compute_risk(churn, fraud, anomaly)
        assert risk.action == "block"

    def test_reasons_nonempty_when_action_not_ok(self) -> None:
        churn = ChurnResult(0.8, True)
        fraud = FraudResult(0.1, False, "low")
        anomaly = AnomalyResult(False, 0.0, [])
        risk = compute_risk(churn, fraud, anomaly)
        assert len(risk.reasons) > 0

    def test_ok_has_no_reasons(self) -> None:
        churn = ChurnResult(0.1, False)
        fraud = FraudResult(0.1, False, "low")
        anomaly = AnomalyResult(False, 0.0, [])
        risk = compute_risk(churn, fraud, anomaly)
        assert risk.action == "ok"
        assert risk.reasons == []


# ──────────────────────────────────────────
# TestOrchestrationPipeline
# ──────────────────────────────────────────


class TestOrchestrationPipeline:
    def setup_method(self) -> None:
        self.pipeline = OrchestrationPipeline()

    def test_run_returns_result(self) -> None:
        event = make_event()
        result = self.pipeline.run(event)
        assert result is not None
        assert result.event_id == event.event_id

    def test_processing_ms_positive(self) -> None:
        event = make_event()
        result = self.pipeline.run(event)
        assert result.processing_ms >= 0.0

    def test_result_has_all_fields(self) -> None:
        event = make_event()
        result = self.pipeline.run(event)
        assert hasattr(result, "churn")
        assert hasattr(result, "fraud")
        assert hasattr(result, "anomaly")
        assert hasattr(result, "risk")
        assert hasattr(result, "timestamp")

    def test_fraud_event_triggers_block(self) -> None:
        event = make_event(
            avg_amount=12000.0,
            n_txn=40,
            account_age=3.0,
        )
        result = self.pipeline.run(event)
        # Very suspicious transaction → block or review
        assert result.risk.action in ("block", "review")

    def test_loyal_customer_low_risk(self) -> None:
        event = make_event(
            tenure=60,
            monthly_charges=30.0,
            contract="Two year",
            internet="DSL",
            avg_amount=100.0,
            n_txn=3,
            account_age=730.0,
        )
        result = self.pipeline.run(event)
        assert result.risk.combined_score < 0.50

    def test_batch_run_same_as_single(self) -> None:
        events = [make_event(tenure=i) for i in range(1, 6)]
        batch_results = self.pipeline.run_batch(events)
        single_results = [self.pipeline.run(e) for e in events]
        # event_ids should match (same events)
        for b, s in zip(batch_results, single_results, strict=True):
            assert b.event_id == s.event_id

    def test_batch_preserves_order(self) -> None:
        events = [make_event(avg_amount=float(i * 1000)) for i in range(1, 6)]
        results = self.pipeline.run_batch(events)
        assert len(results) == len(events)
        for event, result in zip(events, results, strict=True):
            assert result.event_id == event.event_id

    def test_anomaly_event_monitor_or_review(self) -> None:
        event = make_event(
            tenure=60,
            contract="Two year",
            avg_amount=100.0,
            account_age=730.0,
            cpu=[20.0] * 15 + [90.0, 95.0, 92.0, 88.0, 91.0],
        )
        result = self.pipeline.run(event)
        assert result.anomaly.is_anomaly is True
        assert result.risk.action in ("monitor", "intervene", "review", "block")

    def test_event_id_preserved_in_result(self) -> None:
        event = make_event()
        result = self.pipeline.run(event)
        assert result.event_id == event.event_id

    def test_pipeline_with_custom_predictors(self) -> None:
        from orchestration.predictors.anomaly import AnomalyPredictor
        from orchestration.predictors.churn import ChurnPredictor
        from orchestration.predictors.fraud import FraudPredictor

        pipeline = OrchestrationPipeline(
            churn_predictor=ChurnPredictor(),
            fraud_predictor=FraudPredictor(),
            anomaly_predictor=AnomalyPredictor(threshold_sigma=2.0),
        )
        event = make_event()
        result = pipeline.run(event)
        assert result is not None
