"""
Multi-model orchestration pipeline: Churn → Fraud → Anomaly.

Сценарий: telecom/fintech оператор обрабатывает customer event.
Каждый event проходит через три модели последовательно.
Результат — unified risk profile для business-решения.

Архитектура:
  PipelineEvent → [ChurnPredictor] → [FraudPredictor] → [AnomalyPredictor]
                → compute_risk() → PipelineResult

Pipeline выполняет три predictor последовательно (не параллельно),
так как audit trail требует чёткого порядка обработки для EU AI Act compliance.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from .models import PipelineEvent, PipelineResult
from .predictors.anomaly import AnomalyPredictor
from .predictors.churn import ChurnPredictor
from .predictors.fraud import FraudPredictor
from .risk import compute_risk


class OrchestrationPipeline:
    """Multi-model orchestration pipeline.

    Принимает PipelineEvent, запускает три модели последовательно,
    возвращает PipelineResult с unified risk profile.

    Dependency injection: predictors передаются в конструктор,
    что позволяет подменять их mock-объектами в тестах.
    """

    def __init__(
        self,
        churn_predictor: ChurnPredictor | None = None,
        fraud_predictor: FraudPredictor | None = None,
        anomaly_predictor: AnomalyPredictor | None = None,
    ) -> None:
        self.churn = churn_predictor or ChurnPredictor()
        self.fraud = fraud_predictor or FraudPredictor()
        self.anomaly = anomaly_predictor or AnomalyPredictor()

    def run(self, event: PipelineEvent) -> PipelineResult:
        """Process a customer event through all three models.

        Args:
            event: Unified pipeline event with customer, transaction, and metric data.

        Returns:
            PipelineResult with individual model outputs and combined risk profile.
        """
        t0 = time.perf_counter()

        churn_result = self.churn.predict(event.customer)
        fraud_result = self.fraud.predict(event.transaction)
        anomaly_result = self.anomaly.predict(event.metrics)
        risk_profile = compute_risk(churn_result, fraud_result, anomaly_result)

        processing_ms = round((time.perf_counter() - t0) * 1000, 2)

        return PipelineResult(
            event_id=event.event_id,
            timestamp=datetime.now(UTC).isoformat(),
            churn=churn_result,
            fraud=fraud_result,
            anomaly=anomaly_result,
            risk=risk_profile,
            processing_ms=processing_ms,
        )

    def run_batch(self, events: list[PipelineEvent]) -> list[PipelineResult]:
        """Process a batch of events through the pipeline.

        Args:
            events: List of pipeline events to process.

        Returns:
            List of PipelineResult in same order as input events.
        """
        return [self.run(event) for event in events]
