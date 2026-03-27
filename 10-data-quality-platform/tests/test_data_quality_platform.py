"""
Тесты для Data Quality Platform / Tests for the data quality platform.

Покрываем: профилирование, expectations, дрифт, API health.
Covers: profiling, expectations, drift detection, API health.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.data.profiler import detect_distribution_type, profile_column, profile_dataframe
from src.quality.drift import detect_drift, ks_test, psi
from src.quality.expectations import (
    expect_column_exists,
    expect_not_null,
    expect_unique,
    expect_values_in_range,
    expect_values_in_set,
    run_suite,
)

# ---------------------------------------------------------------------------
# Фикстуры / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    """Простой DataFrame для тестов / Simple test DataFrame."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "score": [88.5, 92.3, 76.1, 95.0, 81.4],
        }
    )


@pytest.fixture()
def df_with_nulls() -> pl.DataFrame:
    """DataFrame с пропусками / DataFrame with nulls."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, None, 30.0, None, 50.0],
            "category": ["a", "b", None, "a", "b"],
        }
    )


@pytest.fixture()
def df_with_duplicates() -> pl.DataFrame:
    """DataFrame с дубликатами / DataFrame with duplicate values."""
    return pl.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "name": ["Alice", "Bob", "Bob", "Diana", "Eve"],
        }
    )


# ---------------------------------------------------------------------------
# Тесты профилирования / Profiler tests
# ---------------------------------------------------------------------------


class TestProfiler:
    """Тесты для модуля профилирования / Profiler tests."""

    def test_profile_numeric_column(self, sample_df: pl.DataFrame) -> None:
        """Числовой столбец — есть mean, std, min, max."""
        result = profile_column(sample_df["age"])
        assert result["name"] == "age"
        assert result["count"] == 5
        assert result["null_count"] == 0
        assert result["mean"] == 35.0
        assert result["min"] == 25.0
        assert result["max"] == 45.0
        assert "std" in result

    def test_profile_string_column(self, sample_df: pl.DataFrame) -> None:
        """Строковый столбец — есть top_values."""
        result = profile_column(sample_df["name"])
        assert result["dtype"] == "String"
        assert result["unique_count"] == 5
        assert "top_values" in result

    def test_profile_column_with_nulls(
        self,
        df_with_nulls: pl.DataFrame,
    ) -> None:
        """Столбец с пропусками — корректный null_count."""
        result = profile_column(df_with_nulls["value"])
        assert result["null_count"] == 2
        assert result["null_pct"] == 40.0

    def test_profile_dataframe_overview(
        self,
        sample_df: pl.DataFrame,
    ) -> None:
        """Профиль DataFrame — есть overview и все столбцы."""
        report = profile_dataframe(sample_df)
        assert report["overview"]["row_count"] == 5
        assert report["overview"]["column_count"] == 4
        assert len(report["columns"]) == 4

    def test_detect_normal_distribution(self) -> None:
        """Нормальное распределение определяется корректно."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 1000)
        series = pl.Series("test", values)
        result = detect_distribution_type(series)
        assert result == "normal"

    def test_detect_categorical(self) -> None:
        """Мало уникальных — категориальный."""
        series = pl.Series("cat", [1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        result = detect_distribution_type(series)
        assert result == "categorical"


# ---------------------------------------------------------------------------
# Тесты expectations / Expectations tests
# ---------------------------------------------------------------------------


class TestExpectations:
    """Тесты для модуля проверок качества / Expectations tests."""

    def test_not_null_pass(self, sample_df: pl.DataFrame) -> None:
        """Нет пропусков — проверка проходит."""
        result = expect_not_null(sample_df, "id")
        assert result["passed"] is True

    def test_not_null_fail(self, df_with_nulls: pl.DataFrame) -> None:
        """Есть пропуски — проверка не проходит."""
        result = expect_not_null(df_with_nulls, "value")
        assert result["passed"] is False
        assert result["details"]["null_count"] == 2

    def test_unique_pass(self, sample_df: pl.DataFrame) -> None:
        """Все уникальные — OK."""
        result = expect_unique(sample_df, "id")
        assert result["passed"] is True

    def test_unique_fail(self, df_with_duplicates: pl.DataFrame) -> None:
        """Есть дубликаты — FAIL."""
        result = expect_unique(df_with_duplicates, "id")
        assert result["passed"] is False
        assert result["details"]["duplicate_count"] > 0

    def test_range_pass(self, sample_df: pl.DataFrame) -> None:
        """Все значения в диапазоне — OK."""
        result = expect_values_in_range(sample_df, "age", 0, 100)
        assert result["passed"] is True

    def test_range_fail(self, sample_df: pl.DataFrame) -> None:
        """Есть значения вне диапазона — FAIL."""
        result = expect_values_in_range(sample_df, "age", 30, 50)
        assert result["passed"] is False
        assert result["details"]["out_of_range_count"] > 0

    def test_column_exists_pass(self, sample_df: pl.DataFrame) -> None:
        """Столбец существует."""
        result = expect_column_exists(sample_df, "id")
        assert result["passed"] is True

    def test_column_exists_fail(self, sample_df: pl.DataFrame) -> None:
        """Столбец не существует."""
        result = expect_column_exists(sample_df, "nonexistent")
        assert result["passed"] is False

    def test_values_in_set_pass(self, sample_df: pl.DataFrame) -> None:
        """Все значения в допустимом множестве."""
        result = expect_values_in_set(
            sample_df,
            "name",
            ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        )
        assert result["passed"] is True

    def test_values_in_set_fail(self, sample_df: pl.DataFrame) -> None:
        """Есть значения вне множества."""
        result = expect_values_in_set(
            sample_df,
            "name",
            ["Alice", "Bob"],
        )
        assert result["passed"] is False

    def test_run_suite(self, sample_df: pl.DataFrame) -> None:
        """Запуск набора проверок из конфига."""
        config = {
            "suite_name": "test_suite",
            "expectations": [
                {"check": "expect_not_null", "column": "id"},
                {"check": "expect_unique", "column": "id"},
                {
                    "check": "expect_values_in_range",
                    "column": "age",
                    "kwargs": {"min_value": 0, "max_value": 150},
                },
                {"check": "expect_column_exists", "column": "name"},
            ],
        }
        results = run_suite(sample_df, config)
        assert len(results) == 4
        assert all(r["passed"] for r in results)

    def test_missing_column_handled(self, sample_df: pl.DataFrame) -> None:
        """Отсутствующий столбец обрабатывается без ошибки."""
        result = expect_not_null(sample_df, "missing_col")
        assert result["passed"] is False
        assert "error" in result["details"]


# ---------------------------------------------------------------------------
# Тесты дрифта / Drift tests
# ---------------------------------------------------------------------------


class TestDrift:
    """Тесты для модуля детекции дрифта / Drift detection tests."""

    def test_psi_identical(self) -> None:
        """PSI одинаковых данных близок к нулю."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        result = psi(data, data, bins=10)
        assert result < 0.01

    def test_psi_detects_shift(self) -> None:
        """PSI обнаруживает сдвиг среднего."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)  # сдвиг на 3 сигмы
        result = psi(ref, cur, bins=10)
        assert result > 0.25  # значительный дрифт

    def test_ks_identical(self) -> None:
        """KS-тест одинаковых данных — p-value > 0.05."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        result = ks_test(data, data)
        assert result["p_value"] > 0.05

    def test_ks_detects_drift(self) -> None:
        """KS-тест обнаруживает реальный дрифт."""
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(2, 1, 500)
        result = ks_test(ref, cur)
        assert result["p_value"] < 0.05

    def test_detect_drift_no_drift(self) -> None:
        """detect_drift на одинаковых данных — дрифта нет."""
        rng = np.random.default_rng(42)
        df = pl.DataFrame(
            {
                "a": rng.normal(0, 1, 500).tolist(),
                "b": rng.normal(10, 2, 500).tolist(),
            }
        )
        report = detect_drift(df, df)
        assert report["drift_detected"] is False

    def test_detect_drift_with_drift(self) -> None:
        """detect_drift ловит дрифт, когда он есть."""
        rng = np.random.default_rng(42)
        ref_df = pl.DataFrame(
            {
                "value": rng.normal(0, 1, 500).tolist(),
            }
        )
        cur_df = pl.DataFrame(
            {
                "value": rng.normal(5, 1, 500).tolist(),
            }
        )
        report = detect_drift(ref_df, cur_df)
        assert report["drift_detected"] is True
        assert report["columns_with_drift"] == 1


# ---------------------------------------------------------------------------
# Тесты API / API tests
# ---------------------------------------------------------------------------


class TestAPI:
    """Тесты для FastAPI-эндпоинтов / API endpoint tests."""

    def test_health(self) -> None:
        """Health endpoint возвращает 200."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_profile_endpoint(self, tmp_path: pl.DataFrame) -> None:
        """POST /profile возвращает профиль."""
        # Создаём временный CSV
        csv_path = tmp_path / "test.csv"
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df.write_csv(str(csv_path))

        client = TestClient(app)
        with open(csv_path, "rb") as f:
            response = client.post("/profile", files={"file": f})

        assert response.status_code == 200
        data = response.json()
        assert "overview" in data
        assert "columns" in data

    def test_validate_endpoint(self, tmp_path: pl.DataFrame) -> None:
        """POST /validate возвращает результаты проверок."""
        csv_path = tmp_path / "test.csv"
        df = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        df.write_csv(str(csv_path))

        suite_yaml = "suite_name: test\nexpectations:\n  - check: expect_not_null\n    column: id\n"

        client = TestClient(app)
        with open(csv_path, "rb") as f:
            response = client.post(
                "/validate",
                files={"file": f},
                data={"suite": suite_yaml},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] == 1
