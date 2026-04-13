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
        assert data["alert_sent"] is True
