"""
Тесты корреляционного анализа / Tests for correlation analysis.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from quality.analytics.correlation import (
    CorrelationMatrix,
    _cramers_v,
    _flag,
    correlation_report,
    cramers_v_matrix,
    pearson_matrix,
    spearman_matrix,
)
from quality.api.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def numeric_df() -> pl.DataFrame:
    """DataFrame с числовыми столбцами разной корреляции."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "x": x,
            "y_perfect": x * 2.0 + 1.0,  # r ≈ 1.0 — leakage
            "y_strong": x * 0.97 + rng.normal(0, 0.1, n),  # r ≈ 0.97 — strong
            "y_weak": rng.normal(0, 1, n),  # r ≈ 0 — ok
        }
    )


@pytest.fixture
def categorical_df() -> pl.DataFrame:
    """DataFrame с категориальными столбцами."""
    n = 200
    rng = np.random.default_rng(0)
    a = rng.choice(["cat", "dog", "bird"], n)
    # b полностью зависит от a — Cramér's V ≈ 1.0
    mapping = {"cat": "meow", "dog": "woof", "bird": "tweet"}
    b = np.array([mapping[v] for v in a])
    c = rng.choice(["red", "blue", "green"], n)  # независим
    return pl.DataFrame({"animal": a.tolist(), "sound": b.tolist(), "color": c.tolist()})


@pytest.fixture
def mixed_df(numeric_df: pl.DataFrame, categorical_df: pl.DataFrame) -> pl.DataFrame:
    """Смешанный DataFrame."""
    return pl.concat([numeric_df, categorical_df], how="horizontal")


@pytest.fixture
def csv_bytes(numeric_df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    numeric_df.write_csv(buf)
    return buf.getvalue()


@pytest.fixture
def cat_csv_bytes(categorical_df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    categorical_df.write_csv(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TestFlagHelper
# ---------------------------------------------------------------------------


class TestFlagHelper:
    def test_flag_leakage(self) -> None:
        assert _flag(0.999) == "leakage"

    def test_flag_leakage_negative(self) -> None:
        assert _flag(-0.999) == "leakage"

    def test_flag_strong(self) -> None:
        assert _flag(0.96) == "strong"

    def test_flag_strong_negative(self) -> None:
        assert _flag(-0.96) == "strong"

    def test_flag_ok(self) -> None:
        assert _flag(0.5) == "ok"

    def test_flag_zero(self) -> None:
        assert _flag(0.0) == "ok"

    def test_flag_boundary_leakage(self) -> None:
        assert _flag(0.99) == "leakage"

    def test_flag_boundary_strong(self) -> None:
        assert _flag(0.95) == "strong"


# ---------------------------------------------------------------------------
# TestCramersV
# ---------------------------------------------------------------------------


class TestCramersV:
    def test_perfect_association(self) -> None:
        """Идеально зависимые переменные → V = 1.0."""
        cont = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        v = _cramers_v(cont)
        assert v == pytest.approx(1.0, abs=0.01)

    def test_no_association(self) -> None:
        """Равномерное распределение → V близко к 0."""
        cont = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
        v = _cramers_v(cont)
        assert v < 0.05

    def test_empty_table_returns_zero(self) -> None:
        cont = np.zeros((3, 3), dtype=int)
        assert _cramers_v(cont) == 0.0

    def test_returns_finite(self) -> None:
        rng = np.random.default_rng(7)
        cont = rng.integers(1, 20, size=(4, 4))
        v = _cramers_v(cont)
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# TestPearsonMatrix
# ---------------------------------------------------------------------------


class TestPearsonMatrix:
    def test_returns_correlation_matrix(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        assert isinstance(cm, CorrelationMatrix)
        assert cm.method == "pearson"

    def test_columns_are_numeric(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        assert set(cm.columns) == {"x", "y_perfect", "y_strong", "y_weak"}

    def test_diagonal_is_one(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        n = len(cm.columns)
        for i in range(n):
            assert cm.matrix[i][i] == pytest.approx(1.0)

    def test_perfect_correlation_flagged(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        leakage = [p for p in cm.suspicious_pairs if p.flag == "leakage"]
        assert len(leakage) >= 1
        names = {(p.col_a, p.col_b) for p in leakage}
        assert any("y_perfect" in n for pair in names for n in pair)

    def test_weak_correlation_not_flagged(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        for p in cm.suspicious_pairs:
            cols = {p.col_a, p.col_b}
            if "x" in cols and "y_weak" in cols:
                pytest.fail(f"Weak pair x/y_weak incorrectly flagged: {p}")

    def test_symmetric_matrix(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        n = len(cm.columns)
        for i in range(n):
            for j in range(n):
                assert cm.matrix[i][j] == cm.matrix[j][i]

    def test_n_total_pairs(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        n = len(cm.columns)
        assert cm.n_total_pairs == n * (n - 1) // 2

    def test_n_suspicious_matches_list(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        assert cm.n_suspicious == len(cm.suspicious_pairs)

    def test_to_dict_structure(self, numeric_df: pl.DataFrame) -> None:
        d = pearson_matrix(numeric_df).to_dict()
        assert "method" in d
        assert "columns" in d
        assert "matrix" in d
        assert "suspicious_pairs" in d
        assert "n_total_pairs" in d

    def test_suspicious_pairs_sorted_desc(self, numeric_df: pl.DataFrame) -> None:
        cm = pearson_matrix(numeric_df)
        coeffs = [abs(p.coefficient) for p in cm.suspicious_pairs]
        assert coeffs == sorted(coeffs, reverse=True)

    def test_empty_df_no_numeric(self) -> None:
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        cm = pearson_matrix(df)
        assert cm.columns == []
        assert cm.n_total_pairs == 0


# ---------------------------------------------------------------------------
# TestSpearmanMatrix
# ---------------------------------------------------------------------------


class TestSpearmanMatrix:
    def test_method_label(self, numeric_df: pl.DataFrame) -> None:
        cm = spearman_matrix(numeric_df)
        assert cm.method == "spearman"

    def test_perfect_rank_correlation(self, numeric_df: pl.DataFrame) -> None:
        """y_perfect = 2x+1 → Spearman r = 1.0 → leakage."""
        cm = spearman_matrix(numeric_df)
        leakage = [p for p in cm.suspicious_pairs if p.flag == "leakage"]
        assert len(leakage) >= 1

    def test_p_value_present(self, numeric_df: pl.DataFrame) -> None:
        cm = spearman_matrix(numeric_df)
        for p in cm.suspicious_pairs:
            assert p.p_value is not None
            assert 0.0 <= p.p_value <= 1.0

    def test_non_linear_monotone(self) -> None:
        """Pearson < Spearman для монотонной нелинейной зависимости."""
        rng = np.random.default_rng(1)
        n = 300
        x = rng.uniform(0.1, 5, n)
        y = np.log(x) + rng.normal(0, 0.05, n)
        df = pl.DataFrame({"x": x, "y": y})
        p_cm = pearson_matrix(df)
        s_cm = spearman_matrix(df)
        # Spearman ловит монотонную связь log(x) лучше (ближе к 1.0)
        s_r = abs(s_cm.matrix[0][1] or 0)
        p_r = abs(p_cm.matrix[0][1] or 0)
        assert s_r >= p_r - 0.05  # Spearman >= Pearson для log-зависимости


# ---------------------------------------------------------------------------
# TestCramersVMatrix
# ---------------------------------------------------------------------------


class TestCramersVMatrix:
    def test_method_label(self, categorical_df: pl.DataFrame) -> None:
        cm = cramers_v_matrix(categorical_df)
        assert cm.method == "cramers_v"

    def test_perfect_categorical(self, categorical_df: pl.DataFrame) -> None:
        """animal/sound полностью зависимы → V ≈ 1.0 → leakage."""
        cm = cramers_v_matrix(categorical_df)
        leakage = [p for p in cm.suspicious_pairs if p.flag == "leakage"]
        assert len(leakage) >= 1
        assert any(
            {"p.col_a", "p.col_b"} == {"animal", "sound"}
            or {p.col_a, p.col_b} == {"animal", "sound"}
            for p in leakage
        )

    def test_independent_not_flagged(self, categorical_df: pl.DataFrame) -> None:
        """color не зависит от animal → не флагируется."""
        cm = cramers_v_matrix(categorical_df)
        for p in cm.suspicious_pairs:
            cols = {p.col_a, p.col_b}
            if "color" in cols:
                pytest.fail(f"Independent column 'color' incorrectly flagged: {p}")

    def test_p_value_is_none(self, categorical_df: pl.DataFrame) -> None:
        """Cramér's V не имеет аналитического p-value."""
        cm = cramers_v_matrix(categorical_df)
        for p in cm.suspicious_pairs:
            assert p.p_value is None

    def test_includes_low_cardinality_numeric(self) -> None:
        """Числовые столбцы с <=10 уникальными значениями включаются."""
        df = pl.DataFrame(
            {"cat_int": [0, 1, 0, 1, 0] * 20, "cat_str": ["a", "b", "a", "b", "a"] * 20}
        )
        cm = cramers_v_matrix(df)
        assert "cat_int" in cm.columns


# ---------------------------------------------------------------------------
# TestCorrelationReport
# ---------------------------------------------------------------------------


class TestCorrelationReport:
    def test_all_methods_present(self, numeric_df: pl.DataFrame) -> None:
        report = correlation_report(numeric_df)
        assert "pearson" in report
        assert "spearman" in report
        assert "cramers_v" in report

    def test_summary_present(self, numeric_df: pl.DataFrame) -> None:
        report = correlation_report(numeric_df)
        assert "summary" in report
        s = report["summary"]
        assert "n_suspicious_total" in s
        assert "leakage_risk" in s
        assert "top_suspicious_pairs" in s

    def test_leakage_detected(self, numeric_df: pl.DataFrame) -> None:
        report = correlation_report(numeric_df)
        assert report["summary"]["leakage_risk"] is True

    def test_single_method(self, numeric_df: pl.DataFrame) -> None:
        report = correlation_report(numeric_df, methods=["pearson"])
        assert "pearson" in report
        assert "spearman" not in report
        assert report["summary"]["methods_run"] == ["pearson"]

    def test_unknown_method_raises(self, numeric_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            correlation_report(numeric_df, methods=["unknown"])

    def test_top_suspicious_max_10(self, numeric_df: pl.DataFrame) -> None:
        report = correlation_report(numeric_df)
        assert len(report["summary"]["top_suspicious_pairs"]) <= 10

    def test_suspicious_sorted_desc(self, numeric_df: pl.DataFrame) -> None:
        report = correlation_report(numeric_df)
        top = report["summary"]["top_suspicious_pairs"]
        coeffs = [abs(p["coefficient"]) for p in top]
        assert coeffs == sorted(coeffs, reverse=True)

    def test_mixed_df(self, mixed_df: pl.DataFrame) -> None:
        report = correlation_report(mixed_df)
        # Должны быть и числовые подозрительные, и категориальные
        assert report["pearson"]["n_suspicious"] >= 1
        assert report["cramers_v"]["n_suspicious"] >= 1


# ---------------------------------------------------------------------------
# TestCorrelationAPIEndpoints
# ---------------------------------------------------------------------------


class TestCorrelationAPIEndpoints:
    def test_correlation_endpoint_200(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={"methods": "pearson,spearman"},
        )
        assert resp.status_code == 200

    def test_correlation_response_structure(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={"methods": "pearson"},
        )
        body = resp.json()
        assert "pearson" in body
        assert "summary" in body
        assert "matrix" in body["pearson"]
        assert "suspicious_pairs" in body["pearson"]

    def test_all_methods_default(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        body = resp.json()
        assert "pearson" in body
        assert "spearman" in body
        assert "cramers_v" in body

    def test_invalid_method_422(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={"methods": "invalid_method"},
        )
        assert resp.status_code == 422

    def test_leakage_detected_in_response(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            data={"methods": "pearson"},
        )
        body = resp.json()
        assert body["summary"]["leakage_risk"] is True

    def test_suspicious_endpoint_200(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation/suspicious",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200

    def test_suspicious_response_structure(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation/suspicious",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        body = resp.json()
        assert "suspicious_pairs" in body
        assert "n_suspicious_total" in body
        assert "leakage_risk" in body

    def test_suspicious_pairs_have_required_fields(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation/suspicious",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        pairs = resp.json()["suspicious_pairs"]
        assert len(pairs) > 0
        for p in pairs:
            assert "col_a" in p
            assert "col_b" in p
            assert "coefficient" in p
            assert "flag" in p

    def test_categorical_cramers_v_endpoint(self, cat_csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation",
            files={"file": ("data.csv", cat_csv_bytes, "text/csv")},
            data={"methods": "cramers_v"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["cramers_v"]["n_suspicious"] >= 1

    def test_n_suspicious_total_nonnegative(self, csv_bytes: bytes) -> None:
        resp = client.post(
            "/analytics/correlation/suspicious",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        assert resp.json()["n_suspicious_total"] >= 0
