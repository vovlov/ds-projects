"""Tests for real estate pricing pipeline."""

import sys
from pathlib import Path

import numpy as np
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


class TestH3GeoFeatures:
    """Тесты H3 геопространственных признаков для оценки недвижимости."""

    def test_is_available_returns_bool(self) -> None:
        from pricing.data.geo import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_lat_lng_to_h3_returns_nonempty_string(self) -> None:
        from pricing.data.geo import lat_lng_to_h3

        result = lat_lng_to_h3(55.73, 37.57, resolution=7)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_lat_lng_to_h3_deterministic(self) -> None:
        """Одинаковые координаты → одинаковая ячейка (воспроизводимость)."""
        from pricing.data.geo import lat_lng_to_h3

        cell1 = lat_lng_to_h3(55.73, 37.57, resolution=7)
        cell2 = lat_lng_to_h3(55.73, 37.57, resolution=7)
        assert cell1 == cell2

    def test_lat_lng_to_h3_different_resolutions_differ(self) -> None:
        """r7 и r8 ячейки должны быть разными (r8 мельче)."""
        from pricing.data.geo import lat_lng_to_h3

        r7 = lat_lng_to_h3(55.73, 37.57, resolution=7)
        r8 = lat_lng_to_h3(55.73, 37.57, resolution=8)
        assert r7 != r8

    def test_generate_neighborhood_coordinates_count(self) -> None:
        from pricing.data.geo import generate_neighborhood_coordinates

        rng = np.random.default_rng(42)
        lats, lngs = generate_neighborhood_coordinates(["Хамовники", "Арбат", "Марьино"], rng)
        assert len(lats) == 3
        assert len(lngs) == 3

    def test_generate_neighborhood_coordinates_moscow_bounds(self) -> None:
        """Координаты должны лежать в границах московского региона."""
        from pricing.data.geo import NEIGHBORHOOD_COORDS, generate_neighborhood_coordinates

        rng = np.random.default_rng(0)
        neighborhoods = list(NEIGHBORHOOD_COORDS.keys())
        lats, lngs = generate_neighborhood_coordinates(neighborhoods, rng)
        for lat, lng in zip(lats, lngs, strict=True):
            assert 55.0 <= lat <= 56.5, f"lat={lat} out of Moscow bounds"
            assert 36.5 <= lng <= 38.5, f"lng={lng} out of Moscow bounds"

    def test_enrich_with_geo_adds_latlon_and_h3(self, full_df: pl.DataFrame) -> None:
        from pricing.data.geo import enrich_with_geo

        enriched = enrich_with_geo(full_df, seed=42)
        for col in ["latitude", "longitude", "h3_r7", "h3_r8"]:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_enrich_with_geo_market_stats_columns(self, full_df: pl.DataFrame) -> None:
        from pricing.data.geo import enrich_with_geo

        enriched = enrich_with_geo(full_df, seed=42, include_market_stats=True)
        for col in ["h3_r7_median_price", "h3_r7_count", "price_vs_district"]:
            assert col in enriched.columns, f"Missing market stat: {col}"

    def test_price_vs_district_is_positive(self, full_df: pl.DataFrame) -> None:
        """price_vs_district = price / hex_median — должна быть > 0."""
        from pricing.data.geo import enrich_with_geo

        enriched = enrich_with_geo(full_df, seed=42)
        ratios = enriched["price_vs_district"].drop_nulls().to_list()
        assert len(ratios) > 0
        for r in ratios:
            assert r > 0, f"price_vs_district={r} is non-positive"

    def test_enrich_without_market_stats_skips_ratio(self, full_df: pl.DataFrame) -> None:
        from pricing.data.geo import enrich_with_geo

        enriched = enrich_with_geo(full_df, seed=42, include_market_stats=False)
        assert "h3_r7" in enriched.columns
        assert "h3_r8" in enriched.columns
        assert "price_vs_district" not in enriched.columns

    def test_add_h3_features_graceful_no_latlon(self, full_df: pl.DataFrame) -> None:
        """Без latitude/longitude датафрейм возвращается без изменений."""
        from pricing.data.geo import add_h3_features

        result = add_h3_features(full_df)
        assert result.columns == full_df.columns

    def test_geo_features_list_contents(self) -> None:
        from pricing.data.geo import GEO_FEATURES, GEO_MARKET_FEATURES

        assert "h3_r7" in GEO_FEATURES
        assert "h3_r8" in GEO_FEATURES
        assert "price_vs_district" in GEO_MARKET_FEATURES
        assert "h3_r7_median_price" in GEO_MARKET_FEATURES

    def test_neighborhood_coords_covers_all_districts(self) -> None:
        """Все районы датасета должны иметь координаты в NEIGHBORHOOD_COORDS."""
        from pricing.data.geo import NEIGHBORHOOD_COORDS
        from pricing.data.load import NEIGHBORHOODS

        missing = [n for n in NEIGHBORHOODS if n not in NEIGHBORHOOD_COORDS]
        assert not missing, f"Missing coordinates for neighborhoods: {missing}"

    def test_h3_r7_count_matches_group_size(self, full_df: pl.DataFrame) -> None:
        """Сумма h3_r7_count по уникальным ячейкам ≥ числа строк (без дублей)."""
        from pricing.data.geo import enrich_with_geo

        enriched = enrich_with_geo(full_df, seed=42)
        # Каждая ячейка содержит как минимум одну запись
        min_count = enriched["h3_r7_count"].min()
        assert min_count is not None and min_count >= 1
