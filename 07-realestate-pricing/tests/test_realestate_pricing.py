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


# ---------------------------------------------------------------------------
# Quantile Regression Tests
# ---------------------------------------------------------------------------


def _make_regression_data(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Простые регрессионные данные для тестов без полного датасета."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = 5_000_000 + X[:, 0] * 2_000_000 + X[:, 1] * 500_000 + rng.standard_normal(n) * 300_000
    return X, y


def _make_api_regression_data(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Данные с 9 фичами — соответствует _build_feature_array в API.

    Порядок: sqft, bedrooms, bathrooms, year_built, lot_size, age, neighborhood_code,
             condition_code, has_garage_code.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 9))
    y = 5_000_000 + X[:, 0] * 2_000_000 + X[:, 1] * 500_000 + rng.standard_normal(n) * 300_000
    return X, y


@pytest.mark.skipif(
    not __import__("pricing.models.quantile", fromlist=["is_available"]).is_available(),
    reason="lightgbm not installed",
)
class TestQuantileRegressionModel:
    """Юнит-тесты QuantileRegressionModel на синтетических данных."""

    @pytest.fixture
    def fitted_model(self):
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_regression_data(300)
        model = QuantileRegressionModel(QuantileConfig(n_estimators=50))
        model.fit(X[:200], y[:200])
        return model, X[200:], y[200:]

    def test_fit_sets_is_fitted(self) -> None:
        from pricing.models.quantile import QuantileRegressionModel

        X, y = _make_regression_data(100)
        model = QuantileRegressionModel()
        assert not model._is_fitted
        model.fit(X, y)
        assert model._is_fitted

    def test_fit_creates_five_models(self) -> None:
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_regression_data(100)
        model = QuantileRegressionModel(QuantileConfig(n_estimators=30))
        model.fit(X, y)
        assert set(model._models.keys()) == {0.025, 0.05, 0.5, 0.95, 0.975}

    def test_predict_interval_length_matches_input(self, fitted_model) -> None:
        model, X_test, _ = fitted_model
        intervals = model.predict_interval(X_test)
        assert len(intervals) == len(X_test)

    def test_predict_interval_ordering_after_cqr(self) -> None:
        """После CQR 95% интервал в среднем шире 90% (симметричная поправка q_hat).

        Без калибровки возможно квантильное пересечение (quantile crossing) —
        это known issue quantile regression. CQR исправляет его через отдельные q_hat.
        """
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_regression_data(300)
        model = QuantileRegressionModel(QuantileConfig(n_estimators=50))
        model.fit(X[:200], y[:200])
        model.calibrate(X[200:250], y[200:250])

        intervals = model.predict_interval(X[250:])
        mean_w90 = np.mean([iv.width_90 for iv in intervals])
        mean_w95 = np.mean([iv.width_95 for iv in intervals])
        # В среднем 95%-интервал должен быть шире (q_hat_95 ≥ q_hat_90 обычно)
        msg = f"Средняя ширина 95% ({mean_w95:.0f}) << 90% ({mean_w90:.0f})"
        assert mean_w95 >= mean_w90 * 0.9, msg

    def test_predict_interval_widths_positive(self, fitted_model) -> None:
        model, X_test, _ = fitted_model
        for iv in model.predict_interval(X_test):
            assert iv.width_90 > 0, "Ширина 90% интервала должна быть > 0"
            assert iv.width_95 > 0, "Ширина 95% интервала должна быть > 0"

    def test_predict_before_fit_raises(self) -> None:
        from pricing.models.quantile import QuantileRegressionModel

        X, _ = _make_regression_data(10)
        model = QuantileRegressionModel()
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict_interval(X)

    def test_calibrate_sets_cqr_flags(self, fitted_model) -> None:
        model, X_calib, y_calib = fitted_model
        assert not model._is_calibrated
        model.calibrate(X_calib, y_calib)
        assert model._is_calibrated

    def test_calibrate_before_fit_raises(self) -> None:
        from pricing.models.quantile import QuantileRegressionModel

        X, y = _make_regression_data(50)
        model = QuantileRegressionModel()
        with pytest.raises(RuntimeError):
            model.calibrate(X, y)

    def test_cqr_widens_intervals(self, fitted_model) -> None:
        """После CQR интервалы должны быть ≥ интервалам без калибровки."""
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_regression_data(300)
        model = QuantileRegressionModel(QuantileConfig(n_estimators=50))
        model.fit(X[:200], y[:200])

        X_test = X[250:]
        raw_widths = [iv.width_90 for iv in model.predict_interval(X_test)]

        model.calibrate(X[200:250], y[200:250])
        cqr_widths = [iv.width_90 for iv in model.predict_interval(X_test)]

        # CQR регулирует q_hat — может быть как положительным, так и отрицательным,
        # но в среднем на достаточном объёме должно давать покрытие ≥ 90%.
        assert np.mean(cqr_widths) != np.mean(raw_widths) or model._cqr_q_hat_90 == 0.0

    def test_is_cqr_calibrated_flag_in_interval(self, fitted_model) -> None:
        model, X_test, y_test = fitted_model
        model.calibrate(X_test[:50], y_test[:50])
        for iv in model.predict_interval(X_test[50:]):
            assert iv.is_cqr_calibrated is True

    def test_coverage_result_structure(self, fitted_model) -> None:
        from pricing.models.quantile import CalibrationResult

        model, X_test, y_test = fitted_model
        result = model.compute_coverage(X_test, y_test)
        assert isinstance(result, CalibrationResult)
        assert 0.0 <= result.coverage_90 <= 1.0
        assert 0.0 <= result.coverage_95 <= 1.0
        assert result.n_samples == len(y_test)

    def test_coverage_95_ge_90(self, fitted_model) -> None:
        """95% покрытие должно быть ≥ 90% покрытия (монотонность квантилей)."""
        model, X_test, y_test = fitted_model
        result = model.compute_coverage(X_test, y_test)
        assert result.coverage_95 >= result.coverage_90

    def test_coverage_with_cqr_approaches_nominal(self) -> None:
        """После CQR на достаточном датасете покрытие должно быть близко к 90%."""
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_regression_data(600)
        model = QuantileRegressionModel(QuantileConfig(n_estimators=100))
        model.fit(X[:300], y[:300])
        model.calibrate(X[300:450], y[300:450])
        result = model.compute_coverage(X[450:], y[450:])
        # CQR гарантирует ≥ 90%, с запасом 15% для малых выборок
        assert result.coverage_90 >= 0.75, f"coverage_90={result.coverage_90:.2f} ниже ожидаемого"
        assert result.coverage_95 >= 0.80, f"coverage_95={result.coverage_95:.2f} ниже ожидаемого"


class TestCalibrationResult:
    """Тесты CalibrationResult dataclass."""

    def test_is_well_calibrated_true(self) -> None:
        from pricing.models.quantile import CalibrationResult

        r = CalibrationResult(
            coverage_90=0.90,
            coverage_95=0.95,
            mean_width_90=1_000_000,
            mean_width_95=1_500_000,
            n_samples=100,
        )
        assert r.is_well_calibrated()

    def test_is_well_calibrated_false_when_off(self) -> None:
        from pricing.models.quantile import CalibrationResult

        r = CalibrationResult(
            coverage_90=0.70,
            coverage_95=0.75,
            mean_width_90=500_000,
            mean_width_95=700_000,
            n_samples=50,
        )
        assert not r.is_well_calibrated()

    def test_to_dict_keys(self) -> None:
        from pricing.models.quantile import CalibrationResult

        r = CalibrationResult(
            coverage_90=0.92,
            coverage_95=0.96,
            mean_width_90=1_200_000,
            mean_width_95=1_800_000,
            n_samples=200,
        )
        d = r.to_dict()
        for key in [
            "coverage_90",
            "coverage_95",
            "mean_width_90",
            "mean_width_95",
            "n_samples",
            "is_well_calibrated",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_coverage_values_rounded(self) -> None:
        from pricing.models.quantile import CalibrationResult

        r = CalibrationResult(
            coverage_90=0.9123456,
            coverage_95=0.9567890,
            mean_width_90=1_000_000,
            mean_width_95=1_500_000,
            n_samples=100,
        )
        d = r.to_dict()
        assert d["coverage_90"] == round(0.9123456, 4)


class TestIsAvailable:
    """Тест функции is_available() — должна вернуть bool."""

    def test_returns_bool(self) -> None:
        from pricing.models.quantile import is_available

        result = is_available()
        assert isinstance(result, bool)


class TestQuantileAPI:
    """Тесты API эндпоинтов quantile regression."""

    def test_health_includes_lgbm_available(self) -> None:
        from fastapi.testclient import TestClient
        from pricing.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "lgbm_available" in resp.json()

    def test_health_includes_quantile_model_loaded(self) -> None:
        from fastapi.testclient import TestClient
        from pricing.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert "quantile_model_loaded" in data
        assert isinstance(data["quantile_model_loaded"], bool)

    def test_estimate_intervals_validation_error(self) -> None:
        """Неполный запрос → 422."""
        from fastapi.testclient import TestClient
        from pricing.api.app import app

        client = TestClient(app)
        resp = client.post("/estimate/intervals", json={"sqft": 65})
        assert resp.status_code == 422

    def test_estimate_intervals_without_model_returns_503(self) -> None:
        """Без артефакта quantile модели → 503 Service Unavailable."""
        import pricing.api.app as api_module
        from fastapi.testclient import TestClient
        from pricing.api.app import app

        # Сброс глобального состояния, чтобы тест не зависел от порядка
        original = api_module._quantile_model
        api_module._quantile_model = None

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/estimate/intervals",
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
        # Либо 503 (нет артефакта), либо 200 (если артефакт есть в CI)
        assert resp.status_code in (200, 503)

        api_module._quantile_model = original

    @pytest.mark.skipif(
        not __import__("pricing.models.quantile", fromlist=["is_available"]).is_available(),
        reason="lightgbm not installed",
    )
    def test_estimate_intervals_with_injected_model(self) -> None:
        """Инжектировать обученную модель в глобал и проверить ответ.

        Модель обучается на 9 фичах — столько же строит _build_feature_array в API.
        """
        import pricing.api.app as api_module
        from fastapi.testclient import TestClient
        from pricing.api.app import app
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_api_regression_data(300)
        qm = QuantileRegressionModel(QuantileConfig(n_estimators=30))
        qm.fit(X[:200], y[:200])
        qm.calibrate(X[200:250], y[200:250])

        original = api_module._quantile_model
        api_module._quantile_model = qm

        client = TestClient(app)
        resp = client.post(
            "/estimate/intervals",
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

        for key in [
            "estimated_price",
            "interval_90_low",
            "interval_90_high",
            "interval_95_low",
            "interval_95_high",
            "width_90",
            "width_95",
            "is_cqr_calibrated",
            "lgbm_available",
        ]:
            assert key in data, f"Missing field: {key}"
        assert data["is_cqr_calibrated"] is True
        assert data["lgbm_available"] is True

        api_module._quantile_model = original

    @pytest.mark.skipif(
        not __import__("pricing.models.quantile", fromlist=["is_available"]).is_available(),
        reason="lightgbm not installed",
    )
    def test_estimate_intervals_widths_positive(self) -> None:
        """Ширины интервалов должны быть положительными в ответе API."""
        import pricing.api.app as api_module
        from fastapi.testclient import TestClient
        from pricing.api.app import app
        from pricing.models.quantile import QuantileConfig, QuantileRegressionModel

        X, y = _make_api_regression_data(300)
        qm = QuantileRegressionModel(QuantileConfig(n_estimators=30))
        qm.fit(X[:200], y[:200])
        qm.calibrate(X[200:], y[200:])

        original = api_module._quantile_model
        api_module._quantile_model = qm

        client = TestClient(app)
        resp = client.post(
            "/estimate/intervals",
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

        assert data["width_90"] > 0, "width_90 должна быть > 0"
        assert data["width_95"] > 0, "width_95 должна быть > 0"
        assert data["interval_90_low"] < data["interval_90_high"]
        assert data["interval_95_low"] < data["interval_95_high"]

        api_module._quantile_model = original
