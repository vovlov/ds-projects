"""
Тесты для Data Quality Platform / Tests for the data quality platform.

Покрываем: профилирование, expectations, дрифт, API health.
Covers: profiling, expectations, drift detection, API health.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.data.profiler import detect_distribution_type, profile_column, profile_dataframe
from quality.quality.drift import detect_drift, ks_test, psi
from quality.quality.expectations import (
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


# ---------------------------------------------------------------------------
# Тесты системы алертинга / Alerting system tests
# ---------------------------------------------------------------------------


class TestDriftAlert:
    """Тесты для DriftAlert dataclass / DriftAlert dataclass tests."""

    def test_create_alert_from_stable_report(self) -> None:
        """Отчёт без дрейфа → severity=ok, features_drifted пустой."""
        from quality.alerts.alerting import create_alert_from_report

        report = {
            "drift_detected": False,
            "columns_checked": 2,
            "columns_with_drift": 0,
            "details": [
                {
                    "column": "age",
                    "psi": 0.05,
                    "psi_alert": "ok",
                    "drift_detected": False,
                },
            ],
        }
        alert = create_alert_from_report(report)
        assert alert.severity == "ok"
        assert alert.features_drifted == []
        assert alert.max_psi == 0.05
        assert alert.columns_checked == 2
        assert alert.columns_with_drift == 0

    def test_create_alert_warning_severity(self) -> None:
        """PSI=0.15 → severity=warning."""
        from quality.alerts.alerting import create_alert_from_report

        report = {
            "drift_detected": True,
            "columns_checked": 1,
            "columns_with_drift": 1,
            "details": [
                {
                    "column": "MonthlyCharges",
                    "psi": 0.15,
                    "psi_alert": "moderate",
                    "drift_detected": True,
                },
            ],
        }
        alert = create_alert_from_report(report, source="test-pipeline")
        assert alert.severity == "warning"
        assert "MonthlyCharges" in alert.features_drifted
        assert alert.max_psi == 0.15
        assert alert.source == "test-pipeline"

    def test_create_alert_critical_severity(self) -> None:
        """PSI=0.4 → severity=critical."""
        from quality.alerts.alerting import create_alert_from_report

        report = {
            "drift_detected": True,
            "columns_checked": 1,
            "columns_with_drift": 1,
            "details": [
                {
                    "column": "tenure",
                    "psi": 0.40,
                    "psi_alert": "critical",
                    "drift_detected": True,
                },
            ],
        }
        alert = create_alert_from_report(report)
        assert alert.severity == "critical"
        assert alert.max_psi == 0.40

    def test_alert_to_dict_serializable(self) -> None:
        """to_dict() возвращает JSON-сериализуемый словарь."""
        import json

        from quality.alerts.alerting import create_alert_from_report

        report = {
            "drift_detected": False,
            "columns_checked": 1,
            "columns_with_drift": 0,
            "details": [{"column": "x", "psi": 0.01, "drift_detected": False}],
        }
        alert = create_alert_from_report(report)
        d = alert.to_dict()
        # Убедимся, что сериализуется без ошибок
        assert json.dumps(d)
        assert "severity" in d
        assert "timestamp" in d

    def test_alert_is_actionable_threshold(self) -> None:
        """is_actionable корректно работает с порогами."""
        from quality.alerts.alerting import DriftAlert

        ok_alert = DriftAlert(
            severity="ok",
            features_drifted=[],
            max_psi=0.05,
            columns_checked=1,
            columns_with_drift=0,
            timestamp="2026-04-13T10:00:00+00:00",
        )
        warn_alert = DriftAlert(
            severity="warning",
            features_drifted=["x"],
            max_psi=0.15,
            columns_checked=1,
            columns_with_drift=1,
            timestamp="2026-04-13T10:00:00+00:00",
        )
        crit_alert = DriftAlert(
            severity="critical",
            features_drifted=["x"],
            max_psi=0.35,
            columns_checked=1,
            columns_with_drift=1,
            timestamp="2026-04-13T10:00:00+00:00",
        )

        assert not ok_alert.is_actionable("warning")
        assert warn_alert.is_actionable("warning")
        assert not warn_alert.is_actionable("critical")
        assert crit_alert.is_actionable("warning")
        assert crit_alert.is_actionable("critical")


class TestLogAlertChannel:
    """Тесты для LogAlertChannel / LogAlertChannel tests."""

    def test_send_returns_true(self) -> None:
        """send() всегда возвращает True (лог недоступным не бывает)."""
        from quality.alerts.alerting import DriftAlert, LogAlertChannel

        channel = LogAlertChannel()
        alert = DriftAlert(
            severity="warning",
            features_drifted=["score"],
            max_psi=0.18,
            columns_checked=2,
            columns_with_drift=1,
            timestamp="2026-04-13T10:00:00+00:00",
        )
        result = channel.send(alert)
        assert result is True

    def test_send_critical_uses_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Критический алерт логируется на уровне ERROR."""
        import logging

        from quality.alerts.alerting import DriftAlert, LogAlertChannel

        channel = LogAlertChannel(name="test.drift.alert")
        alert = DriftAlert(
            severity="critical",
            features_drifted=["tenure"],
            max_psi=0.45,
            columns_checked=1,
            columns_with_drift=1,
            timestamp="2026-04-13T10:00:00+00:00",
        )
        with caplog.at_level(logging.ERROR, logger="test.drift.alert"):
            channel.send(alert)
        assert any("DRIFT ALERT" in r.message for r in caplog.records)


class TestWebhookAlertChannel:
    """Тесты для WebhookAlertChannel / WebhookAlertChannel tests."""

    def test_is_available(self) -> None:
        """httpx установлен — is_available() должен вернуть True."""
        from quality.alerts.alerting import WebhookAlertChannel

        # httpx есть в зависимостях проекта
        assert WebhookAlertChannel.is_available() is True

    def test_send_unreachable_url_returns_false(self) -> None:
        """Недостижимый URL → graceful degradation (False, без исключения)."""
        from quality.alerts.alerting import DriftAlert, WebhookAlertChannel

        channel = WebhookAlertChannel(url="http://localhost:9999/no-such-endpoint", timeout=0.5)
        alert = DriftAlert(
            severity="critical",
            features_drifted=["x"],
            max_psi=0.3,
            columns_checked=1,
            columns_with_drift=1,
            timestamp="2026-04-13T10:00:00+00:00",
        )
        # Не должно кидать исключение
        result = channel.send(alert)
        assert result is False


class TestAlertManager:
    """Тесты для AlertManager / AlertManager tests."""

    def test_no_alert_below_threshold(self) -> None:
        """Дрейф ниже порога → alert не отправляется (возвращает None)."""
        from quality.alerts.alerting import AlertManager

        manager = AlertManager(severity_threshold="warning")
        # PSI=0.05 → severity=ok — ниже порога warning
        report = {
            "drift_detected": False,
            "columns_checked": 1,
            "columns_with_drift": 0,
            "details": [{"column": "x", "psi": 0.05, "drift_detected": False}],
        }
        result = manager.process_drift_report(report)
        assert result is None

    def test_alert_sent_on_warning(self) -> None:
        """Дрейф warning → AlertManager создаёт и возвращает алерт."""
        from quality.alerts.alerting import AlertManager

        sent_alerts: list = []

        class MockChannel:
            def send(self, alert):  # noqa: ANN001
                sent_alerts.append(alert)
                return True

        manager = AlertManager(channels=[MockChannel()], severity_threshold="warning")  # type: ignore[list-item]
        report = {
            "drift_detected": True,
            "columns_checked": 1,
            "columns_with_drift": 1,
            "details": [{"column": "MonthlyCharges", "psi": 0.18, "drift_detected": True}],
        }
        alert = manager.process_drift_report(report)
        assert alert is not None
        assert alert.severity == "warning"
        assert len(sent_alerts) == 1

    def test_alert_sent_on_critical(self) -> None:
        """Дрейф critical → отправляется при threshold=warning."""
        from quality.alerts.alerting import AlertManager

        received: list = []

        class MockChannel:
            def send(self, alert):  # noqa: ANN001
                received.append(alert)
                return True

        manager = AlertManager(channels=[MockChannel()], severity_threshold="warning")  # type: ignore[list-item]
        report = {
            "drift_detected": True,
            "columns_checked": 2,
            "columns_with_drift": 2,
            "details": [
                {"column": "tenure", "psi": 0.35, "drift_detected": True},
                {"column": "score", "psi": 0.28, "drift_detected": True},
            ],
        }
        alert = manager.process_drift_report(report)
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.max_psi == 0.35
        assert len(alert.features_drifted) == 2

    def test_channel_error_doesnt_break_manager(self) -> None:
        """Ошибка в одном канале не ломает остальные."""
        from quality.alerts.alerting import AlertManager

        second_channel_called = []

        class FailingChannel:
            def send(self, alert):  # noqa: ANN001
                raise RuntimeError("channel down")

        class GoodChannel:
            def send(self, alert):  # noqa: ANN001
                second_channel_called.append(True)
                return True

        manager = AlertManager(  # type: ignore[list-item]
            channels=[FailingChannel(), GoodChannel()],
            severity_threshold="warning",
        )
        report = {
            "drift_detected": True,
            "columns_checked": 1,
            "columns_with_drift": 1,
            "details": [{"column": "x", "psi": 0.20, "drift_detected": True}],
        }
        # Не должно кидать исключение
        alert = manager.process_drift_report(report)
        assert alert is not None
        # Второй канал всё равно вызвался
        assert second_channel_called

    def test_default_channel_is_log(self) -> None:
        """По умолчанию AlertManager использует LogAlertChannel."""
        from quality.alerts.alerting import AlertManager, LogAlertChannel

        manager = AlertManager()
        assert len(manager.channels) == 1
        assert isinstance(manager.channels[0], LogAlertChannel)


class TestDriftAlertAPIEndpoint:
    """Тесты для /drift/alert endpoint / Drift alert API endpoint tests."""

    def test_drift_alert_no_drift(self, tmp_path) -> None:
        """Одинаковые данные → drift_report без дрейфа, alert=null."""
        import io

        import polars as pl
        from fastapi.testclient import TestClient
        from quality.api.app import app

        rng = np.random.default_rng(42)
        df = pl.DataFrame({"value": rng.normal(0, 1, 200).tolist()})

        ref_csv = df.write_csv()
        cur_csv = df.write_csv()

        client = TestClient(app)
        response = client.post(
            "/drift/alert",
            files={
                "reference": ("ref.csv", io.BytesIO(ref_csv.encode()), "text/csv"),
                "current": ("cur.csv", io.BytesIO(cur_csv.encode()), "text/csv"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "drift_report" in data
        assert "alert" in data
        assert data["alert"] is None  # no drift → no alert
        assert data["alert_sent"] is False

    def test_drift_alert_with_drift(self, tmp_path) -> None:
        """Сдвинутые данные → alert создаётся (без webhook)."""
        import io

        import polars as pl
        from fastapi.testclient import TestClient
        from quality.api.app import app

        rng = np.random.default_rng(42)
        ref_df = pl.DataFrame({"value": rng.normal(0, 1, 500).tolist()})
        cur_df = pl.DataFrame({"value": rng.normal(5, 1, 500).tolist()})

        client = TestClient(app)
        response = client.post(
            "/drift/alert",
            files={
                "reference": ("ref.csv", io.BytesIO(ref_df.write_csv().encode()), "text/csv"),
                "current": ("cur.csv", io.BytesIO(cur_df.write_csv().encode()), "text/csv"),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["drift_report"]["drift_detected"] is True
        assert data["alert"] is not None
        assert data["alert"]["severity"] in ("warning", "critical")


# ---------------------------------------------------------------------------
# Тесты Schema Registry / Schema Registry tests
# ---------------------------------------------------------------------------

import io  # noqa: E402

from quality.schema_registry.registry import SchemaRegistry, get_registry  # noqa: E402
from quality.schema_registry.schema import (  # noqa: E402
    ColumnSchema,
    ColumnType,
    Compatibility,
    DataSchema,
)
from quality.schema_registry.validator import (  # noqa: E402
    infer_schema_from_dataframe,
    validate_dataframe_against_schema,
)


def _make_schema(name: str = "test", extra_col: bool = False) -> DataSchema:
    """Вспомогательная фабрика схем для тестов."""
    cols = [
        ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False),
        ColumnSchema(name="score", dtype=ColumnType.FLOAT),
        ColumnSchema(name="label", dtype=ColumnType.STRING),
    ]
    if extra_col:
        cols.append(ColumnSchema(name="extra", dtype=ColumnType.FLOAT))
    return DataSchema(name=name, columns=cols)


class TestColumnSchemaAndDataSchema:
    """Unit-тесты структур данных / Unit tests for schema data structures."""

    def test_column_schema_defaults(self) -> None:
        col = ColumnSchema(name="age", dtype=ColumnType.INTEGER)
        assert col.nullable is True
        assert col.allowed_values is None
        assert col.description == ""

    def test_column_type_values(self) -> None:
        assert ColumnType.INTEGER.value == "integer"
        assert ColumnType.FLOAT.value == "float"
        assert ColumnType.STRING.value == "string"

    def test_data_schema_column_map(self) -> None:
        schema = _make_schema()
        col_map = schema.column_map()
        assert "id" in col_map
        assert col_map["id"].nullable is False
        assert col_map["score"].dtype == ColumnType.FLOAT

    def test_schema_version_to_dict(self) -> None:
        from quality.schema_registry.schema import SchemaVersion

        sv = SchemaVersion(schema_name="s", version="1.0.0", schema=_make_schema("s"))
        d = sv.to_dict()
        assert d["schema_name"] == "s"
        assert d["version"] == "1.0.0"
        assert isinstance(d["columns"], list)
        assert d["is_latest"] is True


class TestSchemaRegistryCore:
    """Тесты реестра схем / Schema registry core tests."""

    def _fresh(self) -> SchemaRegistry:
        return SchemaRegistry()

    def test_register_first_version(self) -> None:
        reg = self._fresh()
        sv = reg.register(_make_schema("users"))
        assert sv.version == "1.0.0"
        assert sv.is_latest is True
        assert "users" in reg.list_schemas()

    def test_register_explicit_version(self) -> None:
        reg = self._fresh()
        sv = reg.register(_make_schema("users"), version="2.5.0")
        assert sv.version == "2.5.0"

    def test_register_non_breaking_bumps_minor(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("ds"))
        # Добавляем nullable столбец — не breaking
        schema_v2 = _make_schema("ds", extra_col=True)
        sv2 = reg.register(schema_v2)
        assert sv2.version == "1.1.0"

    def test_register_breaking_raises(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("ds"))
        # Удаляем столбец — breaking
        schema_bad = DataSchema(
            name="ds",
            columns=[ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False)],
        )
        with pytest.raises(ValueError, match="Breaking change"):
            reg.register(schema_bad)

    def test_register_breaking_allowed(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("ds"))
        schema_bad = DataSchema(
            name="ds",
            columns=[ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False)],
        )
        sv2 = reg.register(schema_bad, allow_breaking=True)
        assert sv2.version == "2.0.0"

    def test_register_breaking_with_none_compat(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("ds"))
        schema_none = DataSchema(
            name="ds",
            columns=[ColumnSchema(name="new_col", dtype=ColumnType.STRING)],
            compatibility=Compatibility.NONE,
        )
        sv2 = reg.register(schema_none)
        assert sv2.version == "2.0.0"

    def test_get_latest(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("x"))
        reg.register(_make_schema("x", extra_col=True))
        sv = reg.get("x")
        assert sv is not None
        assert sv.is_latest is True

    def test_get_by_version(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("x"))
        reg.register(_make_schema("x", extra_col=True))
        sv = reg.get("x", "1.0.0")
        assert sv is not None
        assert sv.version == "1.0.0"

    def test_get_nonexistent_returns_none(self) -> None:
        reg = self._fresh()
        assert reg.get("missing") is None
        assert reg.get("missing", "1.0.0") is None

    def test_list_versions(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("y"))
        reg.register(_make_schema("y", extra_col=True))
        versions = reg.list_versions("y")
        assert versions == ["1.0.0", "1.1.0"]

    def test_list_schemas(self) -> None:
        reg = self._fresh()
        reg.register(_make_schema("a"))
        reg.register(_make_schema("b"))
        names = reg.list_schemas()
        assert "a" in names
        assert "b" in names


class TestBreakingChangeDetection:
    """Тесты детектора breaking changes / Breaking change detector tests."""

    def _reg(self) -> SchemaRegistry:
        return SchemaRegistry()

    def test_removed_column_is_breaking(self) -> None:
        reg = self._reg()
        reg.register(_make_schema("s"))
        schema_no_score = DataSchema(
            name="s",
            columns=[
                ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False),
                ColumnSchema(name="label", dtype=ColumnType.STRING),
            ],
        )
        result = reg.check_compatibility("s", schema_no_score)
        assert result["compatible"] is False
        assert any("removed" in c for c in result["breaking_changes"])

    def test_type_change_is_breaking(self) -> None:
        reg = self._reg()
        reg.register(_make_schema("s"))
        schema_bad_type = DataSchema(
            name="s",
            columns=[
                ColumnSchema(name="id", dtype=ColumnType.STRING),  # was INTEGER
                ColumnSchema(name="score", dtype=ColumnType.FLOAT),
                ColumnSchema(name="label", dtype=ColumnType.STRING),
            ],
        )
        result = reg.check_compatibility("s", schema_bad_type)
        assert result["compatible"] is False
        assert any("Type changed" in c for c in result["breaking_changes"])

    def test_nullable_to_not_null_is_breaking(self) -> None:
        reg = self._reg()
        reg.register(_make_schema("s"))
        schema_strict = DataSchema(
            name="s",
            columns=[
                ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False),
                ColumnSchema(name="score", dtype=ColumnType.FLOAT, nullable=False),  # was nullable
                ColumnSchema(name="label", dtype=ColumnType.STRING),
            ],
        )
        result = reg.check_compatibility("s", schema_strict)
        assert result["compatible"] is False
        assert any("NOT NULL" in c for c in result["breaking_changes"])

    def test_integer_to_float_widening_is_safe(self) -> None:
        reg = self._reg()
        schema_int = DataSchema(
            name="w",
            columns=[ColumnSchema(name="val", dtype=ColumnType.INTEGER)],
        )
        reg.register(schema_int)
        schema_float = DataSchema(
            name="w",
            columns=[ColumnSchema(name="val", dtype=ColumnType.FLOAT)],
        )
        result = reg.check_compatibility("w", schema_float)
        assert result["compatible"] is True
        assert result["breaking_changes"] == []

    def test_new_nullable_column_is_safe(self) -> None:
        reg = self._reg()
        reg.register(_make_schema("s"))
        result = reg.check_compatibility("s", _make_schema("s", extra_col=True))
        assert result["compatible"] is True

    def test_new_required_column_is_breaking(self) -> None:
        reg = self._reg()
        reg.register(_make_schema("s"))
        schema_new_required = DataSchema(
            name="s",
            columns=[
                ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False),
                ColumnSchema(name="score", dtype=ColumnType.FLOAT),
                ColumnSchema(name="label", dtype=ColumnType.STRING),
                ColumnSchema(name="mandatory", dtype=ColumnType.INTEGER, nullable=False),
            ],
        )
        result = reg.check_compatibility("s", schema_new_required)
        assert result["compatible"] is False
        assert any("required" in c for c in result["breaking_changes"])

    def test_no_existing_schema_is_compatible(self) -> None:
        reg = self._reg()
        result = reg.check_compatibility("nonexistent", _make_schema("nonexistent"))
        assert result["compatible"] is True
        assert result["current_version"] is None


class TestSchemaInference:
    """Тесты авто-инференса схемы / Schema auto-inference tests."""

    def test_infer_types(self) -> None:
        df = pl.DataFrame({"id": [1, 2], "score": [1.5, 2.5], "name": ["a", "b"]})
        schema = infer_schema_from_dataframe(df, name="test")
        col_map = schema.column_map()
        assert col_map["id"].dtype == ColumnType.INTEGER
        assert col_map["score"].dtype == ColumnType.FLOAT
        assert col_map["name"].dtype == ColumnType.STRING

    def test_infer_nullable(self) -> None:
        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        schema = infer_schema_from_dataframe(df)
        col_map = schema.column_map()
        assert col_map["a"].nullable is True
        assert col_map["b"].nullable is False

    def test_infer_schema_name(self) -> None:
        df = pl.DataFrame({"x": [1]})
        schema = infer_schema_from_dataframe(df, name="my_schema")
        assert schema.name == "my_schema"


class TestDataValidation:
    """Тесты валидации данных / Data validation tests."""

    def _base_schema(self) -> DataSchema:
        return DataSchema(
            name="v",
            columns=[
                ColumnSchema(name="id", dtype=ColumnType.INTEGER, nullable=False),
                ColumnSchema(
                    name="age",
                    dtype=ColumnType.INTEGER,
                    nullable=True,
                    min_value=0,
                    max_value=120,
                ),
                ColumnSchema(
                    name="status",
                    dtype=ColumnType.STRING,
                    allowed_values=["active", "inactive"],
                ),
            ],
        )

    def test_valid_data_passes(self) -> None:
        schema = self._base_schema()
        df = pl.DataFrame({"id": [1, 2], "age": [25, 30], "status": ["active", "inactive"]})
        result = validate_dataframe_against_schema(df, schema)
        assert result["passed"] is True
        assert result["failed_checks"] == 0

    def test_missing_column_fails(self) -> None:
        schema = self._base_schema()
        df = pl.DataFrame({"id": [1], "age": [25]})  # missing "status"
        result = validate_dataframe_against_schema(df, schema)
        assert result["passed"] is False
        checks = [i["check"] for i in result["issues"]]
        assert "column_exists" in checks

    def test_null_in_non_nullable_fails(self) -> None:
        schema = self._base_schema()
        df = pl.DataFrame({"id": [1, None], "age": [25, 30], "status": ["active", "active"]})
        result = validate_dataframe_against_schema(df, schema)
        assert result["passed"] is False
        assert any(i["check"] == "nullable" for i in result["issues"])

    def test_out_of_range_fails(self) -> None:
        schema = self._base_schema()
        df = pl.DataFrame({"id": [1], "age": [200], "status": ["active"]})  # age > 120
        result = validate_dataframe_against_schema(df, schema)
        assert result["passed"] is False
        assert any(i["check"] == "value_range" for i in result["issues"])

    def test_invalid_allowed_value_fails(self) -> None:
        schema = self._base_schema()
        df = pl.DataFrame({"id": [1], "age": [25], "status": ["unknown"]})
        result = validate_dataframe_against_schema(df, schema)
        assert result["passed"] is False
        assert any(i["check"] == "allowed_values" for i in result["issues"])

    def test_extra_columns_not_errors(self) -> None:
        schema = self._base_schema()
        df = pl.DataFrame({"id": [1], "age": [25], "status": ["active"], "extra": [99]})
        result = validate_dataframe_against_schema(df, schema)
        assert result["passed"] is True
        assert "extra" in result["extra_columns"]


class TestSchemaRegistryAPI:
    """Интеграционные тесты API реестра схем / Schema registry API integration tests."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> None:
        """Очистить реестр перед каждым тестом / Clear registry before each test."""
        get_registry()._versions.clear()

    @pytest.fixture()
    def client(self) -> TestClient:
        return TestClient(app)

    def _register_payload(self, schema_name: str = "features") -> dict:
        return {
            "schema_name": schema_name,
            "description": "Test schema",
            "compatibility": "BACKWARD",
            "columns": [
                {"name": "user_id", "dtype": "string", "nullable": False},
                {"name": "amount", "dtype": "float", "nullable": True},
            ],
        }

    def test_register_schema_201(self, client: TestClient) -> None:
        resp = client.post("/schema/register", json=self._register_payload())
        assert resp.status_code == 201
        data = resp.json()
        assert data["schema_name"] == "features"
        assert data["version"] == "1.0.0"

    def test_register_schema_auto_version_bump(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        payload_v2 = self._register_payload()
        payload_v2["columns"].append({"name": "extra", "dtype": "integer", "nullable": True})
        resp = client.post("/schema/register", json=payload_v2)
        assert resp.status_code == 201
        assert resp.json()["version"] == "1.1.0"

    def test_register_breaking_returns_409(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        payload_bad = {
            "schema_name": "features",
            "compatibility": "BACKWARD",
            "columns": [{"name": "user_id", "dtype": "string", "nullable": False}],
        }
        resp = client.post("/schema/register", json=payload_bad)
        assert resp.status_code == 409

    def test_list_schemas(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload("a"))
        client.post("/schema/register", json=self._register_payload("b"))
        resp = client.get("/schema/list")
        assert resp.status_code == 200
        assert "a" in resp.json()["schemas"]
        assert "b" in resp.json()["schemas"]

    def test_list_versions(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        payload_v2 = self._register_payload()
        payload_v2["columns"].append({"name": "extra", "dtype": "integer", "nullable": True})
        client.post("/schema/register", json=payload_v2)
        resp = client.get("/schema/features/versions")
        assert resp.status_code == 200
        assert resp.json()["versions"] == ["1.0.0", "1.1.0"]

    def test_list_versions_404_unknown(self, client: TestClient) -> None:
        resp = client.get("/schema/unknown/versions")
        assert resp.status_code == 404

    def test_get_schema_latest(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        resp = client.get("/schema/features")
        assert resp.status_code == 200
        assert resp.json()["version"] == "1.0.0"

    def test_get_schema_404_unknown(self, client: TestClient) -> None:
        resp = client.get("/schema/nonexistent")
        assert resp.status_code == 404

    def test_compatibility_check_compatible(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        candidate = {
            "schema_name": "features",
            "compatibility": "BACKWARD",
            "columns": [
                {"name": "user_id", "dtype": "string", "nullable": False},
                {"name": "amount", "dtype": "float", "nullable": True},
                {"name": "new_nullable", "dtype": "integer", "nullable": True},
            ],
        }
        resp = client.post("/schema/compatible", json=candidate)
        assert resp.status_code == 200
        assert resp.json()["compatible"] is True

    def test_compatibility_check_breaking(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        candidate = {
            "schema_name": "features",
            "compatibility": "BACKWARD",
            "columns": [{"name": "user_id", "dtype": "integer", "nullable": False}],
        }
        resp = client.post("/schema/compatible", json=candidate)
        assert resp.status_code == 200
        data = resp.json()
        assert data["compatible"] is False
        assert len(data["breaking_changes"]) > 0

    def test_validate_csv_against_schema(self, client: TestClient) -> None:
        client.post("/schema/register", json=self._register_payload())
        df = pl.DataFrame({"user_id": ["u1", "u2"], "amount": [10.0, 20.0]})
        resp = client.post(
            "/schema/features/validate",
            files={"file": ("data.csv", io.BytesIO(df.write_csv().encode()), "text/csv")},
        )
        assert resp.status_code == 200
        assert resp.json()["passed"] is True

    def test_infer_schema_endpoint(self, client: TestClient) -> None:
        df = pl.DataFrame({"id": [1, 2], "score": [1.5, 2.5]})
        resp = client.post(
            "/schema/infer",
            files={"file": ("data.csv", io.BytesIO(df.write_csv().encode()), "text/csv")},
            data={"schema_name": "my_schema"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_name"] == "my_schema"
        col_names = [c["name"] for c in data["columns"]]
        assert "id" in col_names
        assert "score" in col_names
