"""Tests for individual predictors: ChurnPredictor, FraudPredictor, AnomalyPredictor."""

from __future__ import annotations

import pytest
from orchestration.models import CustomerData, MetricSnapshot, TransactionData
from orchestration.predictors.anomaly import AnomalyPredictor
from orchestration.predictors.churn import ChurnPredictor
from orchestration.predictors.fraud import FraudPredictor

# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


@pytest.fixture
def loyal_customer() -> CustomerData:
    return CustomerData(
        customer_id="C-001",
        tenure=48,
        monthly_charges=35.0,
        total_charges=1680.0,
        contract="Two year",
        internet_service="DSL",
    )


@pytest.fixture
def churn_risk_customer() -> CustomerData:
    return CustomerData(
        customer_id="C-002",
        tenure=2,
        monthly_charges=99.0,
        total_charges=198.0,
        contract="Month-to-month",
        internet_service="Fiber optic",
    )


@pytest.fixture
def normal_transaction() -> TransactionData:
    return TransactionData(avg_amount=150.0, n_transactions=5, account_age_days=365.0)


@pytest.fixture
def suspicious_transaction() -> TransactionData:
    return TransactionData(avg_amount=8000.0, n_transactions=25, account_age_days=10.0)


@pytest.fixture
def normal_metrics() -> MetricSnapshot:
    return MetricSnapshot(
        cpu=[20.0] * 20,
        latency=[50.0] * 20,
        requests=[100.0] * 20,
    )


@pytest.fixture
def anomalous_metrics() -> MetricSnapshot:
    return MetricSnapshot(
        cpu=[20.0] * 15 + [90.0, 92.0, 88.0, 91.0, 95.0],
        latency=[50.0] * 15 + [50.0, 51.0, 52.0, 50.0, 53.0],
        requests=[100.0] * 15 + [100.0, 101.0, 99.0, 100.0, 102.0],
    )


# ──────────────────────────────────────────
# TestChurnPredictor
# ──────────────────────────────────────────


class TestChurnPredictor:
    def setup_method(self) -> None:
        self.predictor = ChurnPredictor()

    def test_loyal_customer_low_risk(self, loyal_customer: CustomerData) -> None:
        result = self.predictor.predict(loyal_customer)
        assert result.churn_probability < 0.50
        assert result.is_high_risk is False

    def test_churn_risk_customer_high_risk(self, churn_risk_customer: CustomerData) -> None:
        result = self.predictor.predict(churn_risk_customer)
        assert result.churn_probability >= 0.50
        assert result.is_high_risk is True

    def test_probability_in_valid_range(self, loyal_customer: CustomerData) -> None:
        result = self.predictor.predict(loyal_customer)
        assert 0.0 <= result.churn_probability <= 1.0

    def test_month_to_month_higher_than_two_year(self) -> None:
        mtm = CustomerData("C-A", 12, 70.0, 840.0, "Month-to-month", "DSL")
        two_yr = CustomerData("C-B", 12, 70.0, 840.0, "Two year", "DSL")
        p_mtm = self.predictor.predict(mtm).churn_probability
        p_two = self.predictor.predict(two_yr).churn_probability
        assert p_mtm > p_two

    def test_fiber_optic_higher_than_dsl(self) -> None:
        fiber = CustomerData("C-A", 12, 70.0, 840.0, "Month-to-month", "Fiber optic")
        dsl = CustomerData("C-B", 12, 70.0, 840.0, "Month-to-month", "DSL")
        p_fiber = self.predictor.predict(fiber).churn_probability
        p_dsl = self.predictor.predict(dsl).churn_probability
        assert p_fiber > p_dsl

    def test_longer_tenure_lower_risk(self) -> None:
        new_cust = CustomerData("C-A", 1, 70.0, 70.0, "Month-to-month", "Fiber optic")
        old_cust = CustomerData("C-B", 60, 70.0, 4200.0, "Month-to-month", "Fiber optic")
        p_new = self.predictor.predict(new_cust).churn_probability
        p_old = self.predictor.predict(old_cust).churn_probability
        assert p_new > p_old

    def test_high_charges_increase_risk(self) -> None:
        low = CustomerData("C-A", 12, 20.0, 240.0, "One year", "DSL")
        high = CustomerData("C-B", 12, 110.0, 1320.0, "One year", "DSL")
        p_low = self.predictor.predict(low).churn_probability
        p_high = self.predictor.predict(high).churn_probability
        assert p_high > p_low

    def test_zero_tenure(self) -> None:
        cust = CustomerData("C-A", 0, 70.0, 0.0, "Month-to-month", "Fiber optic")
        result = self.predictor.predict(cust)
        assert 0.0 <= result.churn_probability <= 1.0

    def test_unknown_contract_handled(self) -> None:
        cust = CustomerData("C-A", 12, 70.0, 840.0, "Unknown", "DSL")
        result = self.predictor.predict(cust)
        assert 0.0 <= result.churn_probability <= 1.0


# ──────────────────────────────────────────
# TestFraudPredictor
# ──────────────────────────────────────────


class TestFraudPredictor:
    def setup_method(self) -> None:
        self.predictor = FraudPredictor()

    def test_normal_transaction_low_risk(self, normal_transaction: TransactionData) -> None:
        result = self.predictor.predict(normal_transaction)
        assert result.fraud_probability < 0.50
        assert result.is_fraud is False
        assert result.risk_level == "low"

    def test_suspicious_transaction_high_risk(
        self, suspicious_transaction: TransactionData
    ) -> None:
        result = self.predictor.predict(suspicious_transaction)
        assert result.fraud_probability >= 0.50
        assert result.is_fraud is True

    def test_probability_in_valid_range(self, normal_transaction: TransactionData) -> None:
        result = self.predictor.predict(normal_transaction)
        assert 0.0 <= result.fraud_probability <= 1.0

    def test_risk_levels_consistent(self) -> None:
        txn_low = TransactionData(100.0, 3, 365.0)
        txn_high = TransactionData(10000.0, 30, 5.0)
        low = self.predictor.predict(txn_low)
        high = self.predictor.predict(txn_high)
        assert low.risk_level in ("low", "medium")
        assert high.risk_level == "high"

    def test_new_account_higher_fraud_risk(self) -> None:
        old = TransactionData(500.0, 10, 365.0)
        new = TransactionData(500.0, 10, 5.0)
        p_new = self.predictor.predict(new).fraud_probability
        p_old = self.predictor.predict(old).fraud_probability
        assert p_new > p_old

    def test_large_amount_higher_fraud_risk(self) -> None:
        small = TransactionData(50.0, 5, 180.0)
        large = TransactionData(5000.0, 5, 180.0)
        p_large = self.predictor.predict(large).fraud_probability
        p_small = self.predictor.predict(small).fraud_probability
        assert p_large > p_small

    def test_many_transactions_higher_risk(self) -> None:
        few = TransactionData(200.0, 2, 180.0)
        many = TransactionData(200.0, 30, 180.0)
        p_many = self.predictor.predict(many).fraud_probability
        p_few = self.predictor.predict(few).fraud_probability
        assert p_many > p_few

    def test_zero_account_age(self) -> None:
        txn = TransactionData(200.0, 5, 0.0)
        result = self.predictor.predict(txn)
        assert 0.0 <= result.fraud_probability <= 1.0

    def test_risk_level_matches_probability(self) -> None:
        for amt, n_txn, age in [(50.0, 2, 500.0), (3000.0, 20, 15.0), (10000.0, 40, 3.0)]:
            txn = TransactionData(amt, n_txn, age)
            result = self.predictor.predict(txn)
            if result.fraud_probability >= 0.70:
                assert result.risk_level == "high"
            elif result.fraud_probability >= 0.50:
                assert result.risk_level == "medium"
            else:
                assert result.risk_level == "low"


# ──────────────────────────────────────────
# TestAnomalyPredictor
# ──────────────────────────────────────────


class TestAnomalyPredictor:
    def setup_method(self) -> None:
        self.predictor = AnomalyPredictor(window_size=15, threshold_sigma=3.0)

    def test_stable_metrics_no_anomaly(self, normal_metrics: MetricSnapshot) -> None:
        result = self.predictor.predict(normal_metrics)
        assert result.is_anomaly is False
        assert result.max_score == 0.0

    def test_cpu_spike_detected(self, anomalous_metrics: MetricSnapshot) -> None:
        result = self.predictor.predict(anomalous_metrics)
        assert result.is_anomaly is True
        assert "cpu" in result.affected_metrics

    def test_max_score_non_negative(self, normal_metrics: MetricSnapshot) -> None:
        result = self.predictor.predict(normal_metrics)
        assert result.max_score >= 0.0

    def test_affected_metrics_subset_of_known(self, anomalous_metrics: MetricSnapshot) -> None:
        result = self.predictor.predict(anomalous_metrics)
        assert set(result.affected_metrics).issubset({"cpu", "latency", "requests"})

    def test_too_short_series_no_anomaly(self) -> None:
        metrics = MetricSnapshot(cpu=[99.0] * 3, latency=[999.0] * 3, requests=[0.0] * 3)
        result = self.predictor.predict(metrics)
        assert result.is_anomaly is False

    def test_latency_spike_detected(self) -> None:
        metrics = MetricSnapshot(
            cpu=[20.0] * 20,
            latency=[50.0] * 15 + [800.0, 850.0, 900.0, 750.0, 820.0],
            requests=[100.0] * 20,
        )
        result = self.predictor.predict(metrics)
        assert result.is_anomaly is True
        assert "latency" in result.affected_metrics

    def test_empty_metrics_handled(self) -> None:
        metrics = MetricSnapshot(cpu=[], latency=[], requests=[])
        result = self.predictor.predict(metrics)
        assert result.is_anomaly is False

    def test_higher_sigma_threshold_fewer_anomalies(self) -> None:
        low_thresh = AnomalyPredictor(threshold_sigma=2.0)
        high_thresh = AnomalyPredictor(threshold_sigma=5.0)
        metrics = MetricSnapshot(
            cpu=[20.0] * 15 + [60.0, 62.0, 58.0, 65.0, 61.0],
            latency=[50.0] * 20,
            requests=[100.0] * 20,
        )
        low_result = low_thresh.predict(metrics)
        high_result = high_thresh.predict(metrics)
        # Более мягкий порог даёт больше аномалий (или равно)
        assert low_result.max_score >= high_result.max_score or not high_result.is_anomaly
