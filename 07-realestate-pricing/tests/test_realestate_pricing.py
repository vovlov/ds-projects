"""Tests for real estate pricing pipeline."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricing.data.load import (
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
        from pricing.models.train import train_catboost
        from sklearn.model_selection import train_test_split

        indices = list(range(len(full_df)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        results = train_catboost(full_df[train_idx], full_df[test_idx], n_trials=3)

        assert results["r2"] > 0, "R2 should be positive — model better than mean"
        assert results["rmse"] > 0
        assert results["mae"] > 0
        assert "feature_importances" in results
        assert "model" in results


class TestExplain:
    """Проверяем explain_prediction — SHAP waterfall для конкретного предсказания."""

    def test_explain_prediction_catboost(self, full_df: pl.DataFrame) -> None:
        """SHAP через CatBoost built-in должен возвращать корректную структуру."""
        from catboost import CatBoostRegressor
        from pricing.models.explain import explain_prediction
        from pricing.models.train import MODEL_FEATURES

        # Обучаем минимальную модель (50 деревьев) — чтобы не ждать долго
        cat_cols = ["neighborhood", "condition", "has_garage"]
        cat_indices = [i for i, f in enumerate(MODEL_FEATURES) if f in cat_cols]

        X = full_df.select(MODEL_FEATURES).to_pandas()
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")
        X_vals = X.values
        y = full_df["price"].to_numpy().astype(float)

        model = CatBoostRegressor(
            iterations=50, verbose=0, random_seed=42, cat_features=cat_indices
        )
        model.fit(X_vals, y)

        # Берём первый объект
        sample = X_vals[0]
        result = explain_prediction(model, sample, MODEL_FEATURES)

        assert "contributions" in result
        assert "bias" in result
        assert "prediction" in result
        assert len(result["contributions"]) == len(MODEL_FEATURES)
        # prediction должна быть близка к bias + sum(contributions)
        total = result["bias"] + sum(result["contributions"].values())
        assert abs(total - result["prediction"]) < 10  # < 10 руб расхождение

    def test_explain_contributions_sum_to_prediction(self, full_df: pl.DataFrame) -> None:
        """Сумма SHAP + bias = prediction (ключевое свойство SHAP)."""
        from catboost import CatBoostRegressor
        from pricing.models.explain import explain_prediction
        from pricing.models.train import MODEL_FEATURES

        cat_cols = ["neighborhood", "condition", "has_garage"]
        cat_indices = [i for i, f in enumerate(MODEL_FEATURES) if f in cat_cols]

        X = full_df.select(MODEL_FEATURES).to_pandas()
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")
        X_vals = X.values
        y = full_df["price"].to_numpy().astype(float)

        model = CatBoostRegressor(
            iterations=50, verbose=0, random_seed=42, cat_features=cat_indices
        )
        model.fit(X_vals, y)

        sample = X_vals[5]
        result = explain_prediction(model, sample, MODEL_FEATURES)
        # SHAP additivity: f(x) ≈ bias + Σ contributions
        reconstructed = result["bias"] + sum(result["contributions"].values())
        actual_pred = float(model.predict(sample.reshape(1, -1))[0])
        # Должны совпадать с точностью до округления (round(..., 0))
        assert abs(reconstructed - actual_pred) < 100  # < 100 руб — погрешность округления


class TestAPI:
    """Проверяем API-эндпоинты."""

    def test_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient
        from pricing.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_estimate_validation(self) -> None:
        """Неполный запрос должен вернуть 422."""
        from fastapi.testclient import TestClient
        from pricing.api.app import app

        client = TestClient(app)
        resp = client.post("/estimate", json={"sqft": 65})
        assert resp.status_code == 422

    def test_estimate_with_model(self) -> None:
        """Integration test: оценка с обученной моделью."""
        model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.pkl"
        if not model_path.exists():
            pytest.skip("Model artifact not available — run train.py first")

        from fastapi.testclient import TestClient
        from pricing.api.app import app

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

    def test_estimate_shap_waterfall_structure(self) -> None:
        """SHAP waterfall в ответе должен иметь правильную структуру."""
        model_path = Path(__file__).resolve().parents[1] / "artifacts" / "model.pkl"
        if not model_path.exists():
            pytest.skip("Model artifact not available — run train.py first")

        from fastapi.testclient import TestClient
        from pricing.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/estimate",
            json={
                "sqft": 80,
                "bedrooms": 3,
                "bathrooms": 1,
                "year_built": 2010,
                "lot_size": 0,
                "garage": 0,
                "neighborhood": "Марьино",
                "condition": "хорошее",
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # shap_waterfall может быть None если модель не CatBoost, но структура должна быть верной
        if data["shap_waterfall"] is not None:
            wf = data["shap_waterfall"]
            assert "base_value" in wf
            assert "contributions" in wf
            assert "prediction" in wf
            assert len(wf["contributions"]) > 0

            # Каждый contribution должен иметь нужные поля
            for c in wf["contributions"]:
                assert "feature" in c
                assert "value" in c
                assert "contribution" in c
                assert c["direction"] in ("positive", "negative")

            # top_factors должны быть из SHAP (иметь поле contribution)
            assert len(data["top_factors"]) > 0
            assert "contribution" in data["top_factors"][0]
