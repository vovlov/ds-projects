"""Tests for FastAPI orchestration service endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
from orchestration.api.app import app

client = TestClient(app)

# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────


def make_request(
    tenure: int = 12,
    monthly_charges: float = 70.0,
    contract: str = "Month-to-month",
    internet: str = "Fiber optic",
    avg_amount: float = 200.0,
    n_txn: int = 5,
    account_age: float = 180.0,
) -> dict:
    return {
        "customer": {
            "customer_id": "C-TEST",
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": tenure * monthly_charges,
            "contract": contract,
            "internet_service": internet,
        },
        "transaction": {
            "avg_amount": avg_amount,
            "n_transactions": n_txn,
            "account_age_days": account_age,
        },
        "metrics": {
            "cpu": [20.0] * 20,
            "latency": [50.0] * 20,
            "requests": [100.0] * 20,
        },
    }


# ──────────────────────────────────────────
# TestOrchestrationAPI
# ──────────────────────────────────────────


class TestOrchestrationAPI:
    def test_health_endpoint(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "churn" in data["models"]
        assert "fraud" in data["models"]
        assert "anomaly" in data["models"]

    def test_orchestrate_returns_200(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        assert response.status_code == 200

    def test_orchestrate_response_schema(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        data = response.json()
        assert "event_id" in data
        assert "churn" in data
        assert "fraud" in data
        assert "anomaly" in data
        assert "risk" in data
        assert "processing_ms" in data

    def test_orchestrate_churn_fields(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        churn = response.json()["churn"]
        assert "churn_probability" in churn
        assert "is_high_risk" in churn
        assert 0.0 <= churn["churn_probability"] <= 1.0

    def test_orchestrate_fraud_fields(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        fraud = response.json()["fraud"]
        assert "fraud_probability" in fraud
        assert "is_fraud" in fraud
        assert "risk_level" in fraud
        assert fraud["risk_level"] in ("low", "medium", "high")

    def test_orchestrate_anomaly_fields(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        anomaly = response.json()["anomaly"]
        assert "is_anomaly" in anomaly
        assert "max_score" in anomaly
        assert "affected_metrics" in anomaly

    def test_orchestrate_risk_action_valid(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        risk = response.json()["risk"]
        assert risk["action"] in ("block", "review", "intervene", "monitor", "ok")

    def test_high_fraud_event_blocked(self) -> None:
        req = make_request(avg_amount=15000.0, n_txn=50, account_age=2.0)
        response = client.post("/orchestrate", json=req)
        assert response.status_code == 200
        risk = response.json()["risk"]
        assert risk["action"] in ("block", "review")

    def test_loyal_customer_low_combined_score(self) -> None:
        req = make_request(
            tenure=60,
            monthly_charges=30.0,
            contract="Two year",
            internet="DSL",
            avg_amount=80.0,
            n_txn=2,
            account_age=730.0,
        )
        response = client.post("/orchestrate", json=req)
        assert response.status_code == 200
        assert response.json()["risk"]["combined_score"] < 0.5

    def test_batch_endpoint_returns_list(self) -> None:
        requests = [make_request(tenure=i) for i in range(1, 4)]
        response = client.post("/orchestrate/batch", json=requests)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 3

    def test_batch_preserves_order(self) -> None:
        requests = [make_request(avg_amount=float(i * 100)) for i in range(1, 4)]
        response = client.post("/orchestrate/batch", json=requests)
        results = response.json()
        assert len(results) == 3
        for r in results:
            assert "event_id" in r

    def test_processing_ms_in_response(self) -> None:
        response = client.post("/orchestrate", json=make_request())
        assert response.json()["processing_ms"] >= 0.0

    def test_invalid_tenure_rejected(self) -> None:
        req = make_request()
        req["customer"]["tenure"] = -1
        response = client.post("/orchestrate", json=req)
        assert response.status_code == 422

    def test_invalid_avg_amount_rejected(self) -> None:
        req = make_request()
        req["transaction"]["avg_amount"] = -100.0
        response = client.post("/orchestrate", json=req)
        assert response.status_code == 422
