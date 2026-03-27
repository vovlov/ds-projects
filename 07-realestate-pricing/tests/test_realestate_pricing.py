"""Tests for real estate pricing pipeline."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import (
    CATEGORICAL_FEATURES,
    CURRENT_YEAR,
    NUMERICAL_FEATURES,
    generate_dataset,
    load_dataset,
)


@pytest.fixture
def raw_df() -> pl.DataFrame:
    return generate_dataset(n_rows=200, seed=42)


@pytest.fixture
def full_df() -> pl.DataFrame:
    return load_dataset(n_rows=200, seed=42)


class TestData:
    """Проверяем, что сгенерированные данные выглядят реалистично."""

    def test_shape(self, raw_df: pl.DataFrame) -> None:
        assert raw_df.shape[0] == 200
        assert raw_df.shape[1] == 9  # price + 8 raw features

    def test_no_nulls(self, raw_df: pl.DataFrame) -> None:
        for col in raw_df.columns:
            assert raw_df[col].null_count() == 0, f"Null values in {col}"

    def test_price_range(self, raw_df: pl.DataFrame) -> None:
        """Цены в реалистичном для Москвы диапазоне."""
        assert raw_df["price"].min() >= 3_000_000
        assert raw_df["price"].max() <= 30_000_000

    def test_sqft_range(self, raw_df: pl.DataFrame) -> None:
        assert raw_df["sqft"].min() >= 25
        assert raw_df["sqft"].max() <= 200

    def test_bedrooms_positive(self, raw_df: pl.DataFrame) -> None:
        assert (raw_df["bedrooms"] >= 1).all()

    def test_year_built_range(self, raw_df: pl.DataFrame) -> None:
        assert raw_df["year_built"].min() >= 1935
        assert raw_df["year_built"].max() <= 2025

    def test_neighborhood_types(self, raw_df: pl.DataFrame) -> None:
        assert raw_df["neighborhood"].dtype == pl.String

    def test_condition_types(self, raw_df: pl.DataFrame) -> None:
        assert raw_df["condition"].dtype == pl.String


class TestFeatureEngineering:
    """Проверяем, что инженерные признаки вычисляются корректно."""

    def test_age_computed(self, full_df: pl.DataFrame) -> None:
        assert "age" in full_df.columns
        # age = current_year - year_built
        expected = CURRENT_YEAR - full_df["year_built"]
        assert (full_df["age"] == expected).all()

    def test_price_per_sqft(self, full_df: pl.DataFrame) -> None:
        assert "price_per_sqft" in full_df.columns
        assert (full_df["price_per_sqft"] > 0).all()

    def test_has_garage_values(self, full_df: pl.DataFrame) -> None:
        assert "has_garage" in full_df.columns
        unique = set(full_df["has_garage"].unique().to_list())
        assert unique <= {"yes", "no"}

    def test_full_df_has_all_features(self, full_df: pl.DataFrame) -> None:
        for col in NUMERICAL_FEATURES:
            assert col in full_df.columns, f"Missing numerical feature: {col}"
        for col in CATEGORICAL_FEATURES:
            assert col in full_df.columns, f"Missing categorical feature: {col}"


class TestModel:
    """Проверяем, что модель обучается и выдаёт адекватные метрики."""

    def test_catboost_training(self, full_df: pl.DataFrame) -> None:
        """Быстрый тест: 3 trial, чтобы не ждать долго в CI."""
        from sklearn.model_selection import train_test_split
        from src.models.train import train_catboost

        indices = list(range(len(full_df)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        results = train_catboost(full_df[train_idx], full_df[test_idx], n_trials=3)

        assert results["r2"] > 0, "R2 should be positive — model better than mean"
        assert results["rmse"] > 0
        assert results["mae"] > 0
        assert "feature_importances" in results
        assert "model" in results


class TestAPI:
    """Проверяем API-эндпоинты."""

    def test_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_estimate_validation(self) -> None:
        """Неполный запрос должен вернуть 422."""
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.post("/estimate", json={"sqft": 65})
        assert resp.status_code == 422

    def test_estimate_with_model(self) -> None:
        """Integration test: оценка с обученной моделью."""
        model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.pkl"
        if not model_path.exists():
            pytest.skip("Model artifact not available — run train.py first")

        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/estimate",
            json={
                "sqft": 65,
                "bedrooms": 2,
                "bathrooms": 1,
                "year_built": 2015,
                "lot_size": 0,
                "garage": 1,
                "neighborhood": "Хамовники",
                "condition": "хорошее",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["estimated_price"] > 0
        assert data["confidence_low"] < data["estimated_price"]
        assert data["confidence_high"] > data["estimated_price"]
        assert len(data["top_factors"]) > 0
