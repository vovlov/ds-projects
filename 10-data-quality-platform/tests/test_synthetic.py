"""
Тесты для модуля генерации синтетических данных.
Tests for the synthetic data generation module.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.synthetic.generator import (
    ColumnStats,
    SyntheticConfig,
    SyntheticDataGenerator,
    SyntheticResult,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# TestSyntheticConfig / TestColumnStats / TestSyntheticResult
# ---------------------------------------------------------------------------


class TestSyntheticConfigDefaults:
    def test_defaults(self) -> None:
        cfg = SyntheticConfig()
        assert cfg.n_samples == 100
        assert cfg.epsilon is None
        assert cfg.seed == 42
        assert cfg.categorical_threshold == 10

    def test_custom(self) -> None:
        cfg = SyntheticConfig(n_samples=50, epsilon=1.0, seed=7, categorical_threshold=5)
        assert cfg.n_samples == 50
        assert cfg.epsilon == 1.0
        assert cfg.seed == 7
        assert cfg.categorical_threshold == 5


class TestColumnStats:
    def test_continuous_to_dict(self) -> None:
        s = ColumnStats(
            name="age", col_type="continuous", mean=30.0, std=5.0, min_val=18.0, max_val=65.0
        )
        d = s.to_dict()
        assert d["name"] == "age"
        assert d["col_type"] == "continuous"
        assert d["mean"] == 30.0
        assert d["n_categories"] is None

    def test_categorical_to_dict(self) -> None:
        s = ColumnStats(
            name="gender",
            col_type="categorical",
            categories=["F", "M"],
            probabilities=[0.5, 0.5],
        )
        d = s.to_dict()
        assert d["n_categories"] == 2
        assert d["categories"] == ["F", "M"]
        assert d["mean"] is None


class TestSyntheticResult:
    def test_to_dict_keys(self) -> None:
        stats = ColumnStats(
            name="x", col_type="continuous", mean=0.0, std=1.0, min_val=-3.0, max_val=3.0
        )
        r = SyntheticResult(
            data={"x": [1.0, 2.0]},
            n_samples=2,
            column_stats=[stats],
            privacy_budget=None,
            fidelity_score=0.95,
        )
        d = r.to_dict()
        assert "n_samples" in d
        assert "columns" in d
        assert "fidelity_score" in d
        assert "privacy_budget" in d
        assert d["fidelity_score"] == 0.95


# ---------------------------------------------------------------------------
# TestSyntheticDataGeneratorFit
# ---------------------------------------------------------------------------


class TestSyntheticDataGeneratorFit:
    def test_not_fitted_initially(self) -> None:
        gen = SyntheticDataGenerator()
        assert not gen.is_fitted

    def test_fitted_after_fit(self) -> None:
        gen = SyntheticDataGenerator()
        gen.fit({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]})
        assert gen.is_fitted

    def test_fit_detects_continuous(self) -> None:
        data = {"price": list(range(20))}
        gen = SyntheticDataGenerator(SyntheticConfig(categorical_threshold=10))
        gen.fit(data)
        stats = gen._column_stats[0]
        assert stats.col_type == "continuous"

    def test_fit_detects_categorical_low_cardinality(self) -> None:
        data = {"color": ["red", "blue", "green", "red", "blue"]}
        gen = SyntheticDataGenerator()
        gen.fit(data)
        stats = gen._column_stats[0]
        assert stats.col_type == "categorical"
        assert set(stats.categories) == {"red", "blue", "green"}  # type: ignore[arg-type]

    def test_fit_categorical_probabilities_sum_to_one(self) -> None:
        data = {"tier": ["A", "B", "C", "A", "B", "A"]}
        gen = SyntheticDataGenerator()
        gen.fit(data)
        stats = gen._column_stats[0]
        assert abs(sum(stats.probabilities) - 1.0) < 1e-9  # type: ignore[arg-type]

    def test_fit_mixed_columns(self) -> None:
        data = {
            "age": list(range(20, 60)),
            "status": ["active", "inactive"] * 20,
        }
        gen = SyntheticDataGenerator()
        gen.fit(data)
        types = {s.name: s.col_type for s in gen._column_stats}
        assert types["age"] == "continuous"
        assert types["status"] == "categorical"

    def test_fit_builds_cholesky_for_two_continuous(self) -> None:
        data = {
            "x": [float(i) for i in range(30)],
            "y": [float(i) * 2 + 1 for i in range(30)],
        }
        gen = SyntheticDataGenerator()
        gen.fit(data)
        assert gen._corr_cholesky is not None

    def test_fit_no_cholesky_for_single_continuous(self) -> None:
        data = {"x": [float(i) for i in range(30)]}
        gen = SyntheticDataGenerator()
        gen.fit(data)
        assert gen._corr_cholesky is None


# ---------------------------------------------------------------------------
# TestSyntheticDataGeneratorGenerate
# ---------------------------------------------------------------------------


class TestSyntheticDataGeneratorGenerate:
    def test_generate_before_fit_raises(self) -> None:
        gen = SyntheticDataGenerator()
        with pytest.raises(RuntimeError):
            gen.generate()

    def test_generate_correct_n_samples(self) -> None:
        data = {"v": list(range(50))}
        gen = SyntheticDataGenerator(SyntheticConfig(n_samples=20))
        result = gen.fit_generate(data, n_samples=20)
        assert result.n_samples == 20
        assert len(result.data["v"]) == 20

    def test_generate_custom_n_samples_override(self) -> None:
        data = {"v": list(range(50))}
        gen = SyntheticDataGenerator()
        result = gen.fit_generate(data, n_samples=77)
        assert len(result.data["v"]) == 77

    def test_generate_continuous_within_range(self) -> None:
        vals = list(range(10, 100))
        gen = SyntheticDataGenerator()
        result = gen.fit_generate({"x": vals}, n_samples=500)
        synth = result.data["x"]
        assert all(10 <= v <= 99 for v in synth)

    def test_generate_categorical_only_known_values(self) -> None:
        data = {"tier": ["gold", "silver", "bronze"] * 20}
        gen = SyntheticDataGenerator()
        result = gen.fit_generate(data, n_samples=200)
        synth = result.data["tier"]
        assert all(v in {"gold", "silver", "bronze"} for v in synth)

    def test_generate_preserves_mean_approx(self) -> None:
        import numpy as np

        vals = list(range(1000))
        gen = SyntheticDataGenerator(SyntheticConfig(seed=0))
        result = gen.fit_generate({"x": vals}, n_samples=1000)
        synth_mean = float(np.mean(result.data["x"]))
        # Допуск ±50 (≈ 10% от range=1000)
        assert abs(synth_mean - 499.5) < 50

    def test_generate_fidelity_score_in_range(self) -> None:
        vals = list(range(50))
        gen = SyntheticDataGenerator()
        result = gen.fit_generate({"x": vals}, n_samples=200)
        assert 0.0 <= result.fidelity_score <= 1.0

    def test_generate_high_fidelity_without_dp(self) -> None:
        vals = list(range(1000))
        gen = SyntheticDataGenerator(SyntheticConfig(seed=42))
        result = gen.fit_generate({"x": vals}, n_samples=1000)
        assert result.fidelity_score > 0.8

    def test_generate_dp_privacy_budget_recorded(self) -> None:
        data = {"x": list(range(100))}
        gen = SyntheticDataGenerator(SyntheticConfig(epsilon=1.0))
        result = gen.fit_generate(data)
        assert result.privacy_budget == 1.0

    def test_generate_no_dp_privacy_budget_none(self) -> None:
        data = {"x": list(range(100))}
        gen = SyntheticDataGenerator()
        result = gen.fit_generate(data)
        assert result.privacy_budget is None

    def test_generate_correlated_columns(self) -> None:
        import numpy as np

        # x и y сильно коррелированы — синтетические тоже должны коррелировать
        n = 200
        x_vals = list(range(n))
        y_vals = [v * 3 + 10 for v in x_vals]
        gen = SyntheticDataGenerator(SyntheticConfig(seed=1))
        result = gen.fit_generate({"x": x_vals, "y": y_vals}, n_samples=300)
        corr = float(np.corrcoef(result.data["x"], result.data["y"])[0, 1])
        assert corr > 0.5  # должна сохраниться положительная корреляция

    def test_generate_column_stats_in_result(self) -> None:
        data = {"a": list(range(30)), "b": ["X", "Y"] * 15}
        gen = SyntheticDataGenerator()
        result = gen.fit_generate(data)
        names = [s.name for s in result.column_stats]
        assert "a" in names
        assert "b" in names

    def test_fit_generate_convenience(self) -> None:
        data = {"v": list(range(50))}
        gen = SyntheticDataGenerator()
        result = gen.fit_generate(data, n_samples=10)
        assert len(result.data["v"]) == 10


# ---------------------------------------------------------------------------
# TestSyntheticAPIEndpoints
# ---------------------------------------------------------------------------


class TestSyntheticAPIEndpoints:
    def test_generate_200(self) -> None:
        payload = {
            "data": {"age": list(range(20, 60)), "salary": [i * 1000 for i in range(20, 60)]},
            "n_samples": 50,
        }
        resp = client.post("/synthetic/generate", json=payload)
        assert resp.status_code == 200

    def test_generate_response_structure(self) -> None:
        payload = {
            "data": {"x": list(range(30))},
            "n_samples": 10,
        }
        resp = client.post("/synthetic/generate", json=payload)
        body = resp.json()
        assert "data" in body
        assert "n_samples" in body
        assert "fidelity_score" in body
        assert "column_stats" in body

    def test_generate_correct_n_samples(self) -> None:
        payload = {
            "data": {"v": list(range(50))},
            "n_samples": 25,
        }
        resp = client.post("/synthetic/generate", json=payload)
        body = resp.json()
        assert body["n_samples"] == 25
        assert len(body["data"]["v"]) == 25

    def test_generate_with_categorical(self) -> None:
        payload = {
            "data": {"status": ["active", "inactive", "pending"] * 10},
            "n_samples": 30,
        }
        resp = client.post("/synthetic/generate", json=payload)
        body = resp.json()
        assert resp.status_code == 200
        assert set(body["data"]["status"]).issubset({"active", "inactive", "pending"})

    def test_generate_with_dp(self) -> None:
        payload = {
            "data": {"x": list(range(100))},
            "n_samples": 50,
            "epsilon": 2.0,
        }
        resp = client.post("/synthetic/generate", json=payload)
        body = resp.json()
        assert resp.status_code == 200
        assert body["privacy_budget"] == 2.0

    def test_generate_empty_data_422(self) -> None:
        payload = {"data": {}, "n_samples": 10}
        resp = client.post("/synthetic/generate", json=payload)
        assert resp.status_code == 422

    def test_generate_all_columns_empty_422(self) -> None:
        payload = {"data": {"x": []}, "n_samples": 10}
        resp = client.post("/synthetic/generate", json=payload)
        assert resp.status_code == 422

    def test_info_200(self) -> None:
        resp = client.get("/synthetic/info")
        assert resp.status_code == 200

    def test_info_structure(self) -> None:
        resp = client.get("/synthetic/info")
        body = resp.json()
        assert "algorithm" in body
        assert "differential_privacy" in body
        assert "compliance" in body
        assert "references" in body

    def test_info_compliance_keys(self) -> None:
        resp = client.get("/synthetic/info")
        compliance = resp.json()["compliance"]
        assert "GDPR_Article_5" in compliance
        assert "EU_AI_Act_Article_10" in compliance

    def test_generate_fidelity_score_valid_range(self) -> None:
        payload = {
            "data": {"age": list(range(18, 70)), "income": [i * 500 for i in range(18, 70)]},
            "n_samples": 100,
        }
        resp = client.post("/synthetic/generate", json=payload)
        body = resp.json()
        score = body["fidelity_score"]
        assert 0.0 <= score <= 1.0
