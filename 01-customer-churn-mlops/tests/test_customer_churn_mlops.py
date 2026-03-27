"""Tests for customer churn pipeline."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    load_raw,
    prepare_dataset,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw.csv"
SKIP_REASON = "Dataset not available"


@pytest.fixture
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip(SKIP_REASON)
    return load_raw(DATA_PATH)


@pytest.fixture
def full_df():
    if not DATA_PATH.exists():
        pytest.skip(SKIP_REASON)
    return prepare_dataset(DATA_PATH)


class TestDataLoading:
    def test_load_raw_shape(self, raw_df):
        assert raw_df.shape[0] == 7043
        assert raw_df.shape[1] == 21

    def test_load_raw_no_nulls_in_target(self, raw_df):
        assert raw_df[TARGET].null_count() == 0

    def test_target_is_binary(self, raw_df):
        unique_vals = raw_df[TARGET].unique().sort().to_list()
        assert unique_vals == [0, 1]

    def test_total_charges_is_float(self, raw_df):
        assert raw_df["TotalCharges"].dtype == pl.Float64

    def test_no_empty_total_charges(self, raw_df):
        assert raw_df["TotalCharges"].null_count() == 0


class TestFeatureEngineering:
    def test_add_features_creates_columns(self, full_df):
        expected = ["AvgMonthlySpend", "ExpectedTotalCharges", "TenureGroup", "NumServices"]
        for col in expected:
            assert col in full_df.columns

    def test_tenure_group_values(self, full_df):
        unique = full_df["TenureGroup"].unique().sort().to_list()
        assert set(unique) == {"long", "mid", "new"}

    def test_num_services_range(self, full_df):
        assert full_df["NumServices"].min() >= 0
        assert full_df["NumServices"].max() <= 6

    def test_avg_monthly_spend_non_negative(self, full_df):
        assert (full_df["AvgMonthlySpend"] >= 0).all()


class TestDataQuality:
    def test_categorical_features_exist(self, raw_df):
        for col in CATEGORICAL_FEATURES:
            assert col in raw_df.columns

    def test_numerical_features_exist(self, raw_df):
        for col in NUMERICAL_FEATURES:
            assert col in raw_df.columns

    def test_churn_rate_reasonable(self, raw_df):
        churn_rate = raw_df[TARGET].mean()
        assert 0.1 < churn_rate < 0.5

    def test_tenure_non_negative(self, raw_df):
        assert (raw_df["tenure"] >= 0).all()

    def test_monthly_charges_positive(self, raw_df):
        assert (raw_df["MonthlyCharges"] > 0).all()


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_predict_validation(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.post("/predict", json={"gender": "Male"})
        assert resp.status_code == 422

    def test_predict_with_model(self):
        """Integration test: predict with real trained model."""
        from pathlib import Path

        model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.pkl"
        if not model_path.exists():
            pytest.skip("Model artifact not available — run train.py first")

        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/predict",
            json={
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 70.35,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_level"] in ("low", "medium", "high")
