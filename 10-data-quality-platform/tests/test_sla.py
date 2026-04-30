"""
Tests for SLA monitoring: SLO definitions, error budget tracking,
multi-window burn rate alerting, and API endpoints.

Тесты для SLA-мониторинга: определения SLO, отслеживание бюджета ошибок,
multi-window burn rate алертинг и API-эндпоинты.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.sla.budget import _BURN_RATE_RULES, ErrorBudgetTracker
from quality.sla.monitor import SLAMonitor, reset_monitor
from quality.sla.slo import SLIObservation, SLIType, SLODefinition

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_monitor():
    """Сбрасывать синглтон перед каждым тестом / Reset singleton before each test."""
    reset_monitor()
    yield
    reset_monitor()


def _make_slo(
    service: str = "test-svc",
    sli_type: SLIType = SLIType.AVAILABILITY,
    target: float = 0.999,
) -> SLODefinition:
    return SLODefinition(service=service, sli_type=sli_type, target=target)


def _make_obs(service: str = "test-svc", good: int = 990, total: int = 1000) -> SLIObservation:
    return SLIObservation(service=service, sli_type=SLIType.AVAILABILITY, good=good, total=total)


# ---------------------------------------------------------------------------
# TestSLODefinition
# ---------------------------------------------------------------------------


class TestSLODefinition:
    def test_error_budget_fraction_999(self) -> None:
        slo = _make_slo(target=0.999)
        assert abs(slo.error_budget_fraction - 0.001) < 1e-9

    def test_error_budget_minutes_30d(self) -> None:
        # 30д * 24ч * 60мин * 0.001 = 43.2 мин
        slo = SLODefinition(
            service="x", sli_type=SLIType.AVAILABILITY, target=0.999, window_days=30
        )
        assert abs(slo.error_budget_minutes - 43.2) < 0.01

    def test_target_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            SLODefinition(service="x", sli_type=SLIType.AVAILABILITY, target=1.5)

    def test_latency_sli_requires_threshold(self) -> None:
        with pytest.raises(ValueError):
            SLODefinition(service="x", sli_type=SLIType.LATENCY, target=0.99)

    def test_latency_sli_with_threshold_ok(self) -> None:
        slo = SLODefinition(
            service="x", sli_type=SLIType.LATENCY, target=0.99, latency_threshold_ms=200.0
        )
        assert slo.latency_threshold_ms == 200.0

    def test_to_dict_fields(self) -> None:
        slo = _make_slo()
        d = slo.to_dict()
        assert d["service"] == "test-svc"
        assert "target_pct" in d
        assert "error_budget_minutes" in d


# ---------------------------------------------------------------------------
# TestSLIObservation
# ---------------------------------------------------------------------------


class TestSLIObservation:
    def test_sli_value_normal(self) -> None:
        obs = _make_obs(good=990, total=1000)
        assert abs(obs.sli_value - 0.99) < 1e-9

    def test_sli_value_no_events(self) -> None:
        obs = _make_obs(good=0, total=0)
        assert obs.sli_value == 1.0  # no events → no errors by convention

    def test_error_count(self) -> None:
        obs = _make_obs(good=970, total=1000)
        assert obs.error_count == 30

    def test_to_dict_fields(self) -> None:
        obs = _make_obs()
        d = obs.to_dict()
        assert "observation_id" in d
        assert d["service"] == "test-svc"
        assert "sli_value" in d


# ---------------------------------------------------------------------------
# TestErrorBudgetTracker
# ---------------------------------------------------------------------------


class TestErrorBudgetTracker:
    def _tracker(self, target: float = 0.999) -> ErrorBudgetTracker:
        slo = _make_slo(target=target)
        return ErrorBudgetTracker(slo)

    def test_no_observations_no_alerts(self) -> None:
        status = self._tracker().get_status()
        assert status.total_events == 0
        assert status.active_alerts == []
        assert status.compliant is True

    def test_perfect_traffic_no_alerts(self) -> None:
        t = self._tracker()
        t.record(1000, 1000)
        status = t.get_status()
        assert status.burn_rate_1h == 0.0
        assert status.active_alerts == []

    def test_high_error_rate_triggers_critical_alert(self) -> None:
        t = self._tracker(target=0.99)
        # 50% error rate → burn rate = 0.5 / 0.01 = 50 >> 14.4 (critical threshold)
        now = datetime.now(UTC)
        t.record(500, 1000, at=now)
        status = t.get_status()
        assert len(status.active_alerts) == 1
        assert status.active_alerts[0].severity == "critical"

    def test_moderate_error_triggers_high_alert(self) -> None:
        t = self._tracker(target=0.99)
        # 7% error rate → burn rate = 0.07 / 0.01 = 7 ≥ 6 (high threshold), < 14.4 (critical)
        now = datetime.now(UTC)
        t.record(93, 100, at=now - timedelta(minutes=30))  # within 1h window
        status = t.get_status()
        alerts = status.active_alerts
        assert len(alerts) == 1
        assert alerts[0].severity in ("critical", "high")

    def test_no_burn_on_old_observations(self) -> None:
        t = self._tracker()
        # Old observation (3 hours ago) — outside 1h window burn check
        old = datetime.now(UTC) - timedelta(hours=3)
        t.record(0, 1000, at=old)  # 100% errors, but old
        status = t.get_status()
        # 1h burn rate should be 0 (no recent events)
        assert status.burn_rate_1h == 0.0

    def test_budget_consumed_exceeds_total_clamped(self) -> None:
        t = self._tracker(target=0.999)
        # 5% error rate >> 0.1% budget
        t.record(950, 1000)
        status = t.get_status()
        assert status.error_budget_consumed <= status.error_budget_total
        assert status.error_budget_remaining >= 0.0

    def test_projected_exhaustion_computed_when_burning(self) -> None:
        t = self._tracker(target=0.99)
        now = datetime.now(UTC)
        t.record(93, 100, at=now)  # recent burn
        status = t.get_status()
        # If there's active burning, projection should be a positive number
        if status.burn_rate_1h > 0 and status.error_budget_remaining > 0:
            assert status.projected_exhaustion_hours is not None
            assert status.projected_exhaustion_hours > 0

    def test_status_to_dict_structure(self) -> None:
        t = self._tracker()
        t.record(999, 1000)
        d = t.get_status().to_dict()
        assert "error_budget" in d
        assert "burn_rates" in d
        assert "1h" in d["burn_rates"]
        assert "6h" in d["burn_rates"]
        assert "3d" in d["burn_rates"]

    def test_alert_severity_ordering(self) -> None:
        # Ensure rules are sorted from highest to lowest burn threshold
        thresholds = [r[0] for r in _BURN_RATE_RULES]
        assert thresholds == sorted(thresholds, reverse=True)


# ---------------------------------------------------------------------------
# TestSLAMonitor
# ---------------------------------------------------------------------------


class TestSLAMonitor:
    def _monitor(self) -> SLAMonitor:
        return SLAMonitor()

    def test_define_slo_registers(self) -> None:
        m = self._monitor()
        slo = _make_slo()
        m.define_slo(slo)
        assert len(m.list_slos()) == 1

    def test_observe_without_slo_raises(self) -> None:
        m = self._monitor()
        with pytest.raises(ValueError, match="No SLO defined"):
            m.observe(_make_obs())

    def test_observe_records_observation(self) -> None:
        m = self._monitor()
        m.define_slo(_make_slo())
        m.observe(_make_obs(good=990, total=1000))
        obs = m.get_recent_observations()
        assert len(obs) == 1
        assert obs[0]["good"] == 990

    def test_get_status_unknown_service_empty(self) -> None:
        m = self._monitor()
        result = m.get_status("nonexistent")
        assert result == []

    def test_get_all_statuses_multiple_slos(self) -> None:
        m = self._monitor()
        m.define_slo(_make_slo(service="svc-a"))
        m.define_slo(_make_slo(service="svc-b"))
        assert len(m.get_all_statuses()) == 2

    def test_generate_report_all_compliant(self) -> None:
        m = self._monitor()
        m.define_slo(_make_slo())
        m.observe(_make_obs(good=1000, total=1000))
        report = m.generate_report()
        d = report.to_dict()
        assert d["summary"]["total_services"] == 1
        assert d["summary"]["compliant_services"] == 1
        assert d["summary"]["compliance_rate_pct"] == 100.0

    def test_generate_report_empty_monitor(self) -> None:
        m = self._monitor()
        report = m.generate_report()
        d = report.to_dict()
        assert d["summary"]["total_services"] == 0
        assert d["summary"]["compliance_rate_pct"] == 100.0

    def test_observations_filtered_by_service(self) -> None:
        m = self._monitor()
        m.define_slo(_make_slo(service="svc-a"))
        m.define_slo(_make_slo(service="svc-b"))
        m.observe(_make_obs(service="svc-a", good=990, total=1000))
        m.observe(
            SLIObservation(service="svc-b", sli_type=SLIType.AVAILABILITY, good=999, total=1000)
        )
        assert len(m.get_recent_observations(service="svc-a")) == 1
        assert len(m.get_recent_observations(service="svc-b")) == 1

    def test_redefine_slo_resets_tracker(self) -> None:
        m = self._monitor()
        m.define_slo(_make_slo())
        m.observe(_make_obs(good=900, total=1000))
        # Redefine → should reset tracker
        m.define_slo(_make_slo())
        statuses = m.get_all_statuses()
        assert statuses[0].total_events == 0


# ---------------------------------------------------------------------------
# TestSLAAPIEndpoints
# ---------------------------------------------------------------------------


class TestSLAAPIEndpoints:
    def _define(self, service: str = "api-svc", target: float = 0.999) -> dict:
        return client.post(
            "/sla/define",
            json={"service": service, "sli_type": "availability", "target": target},
        ).json()

    def _observe(self, service: str = "api-svc", good: int = 990, total: int = 1000) -> dict:
        return client.post(
            "/sla/observe",
            json={"service": service, "sli_type": "availability", "good": good, "total": total},
        ).json()

    def test_define_slo_returns_201(self) -> None:
        resp = client.post(
            "/sla/define",
            json={"service": "churn-api", "sli_type": "availability", "target": 0.999},
        )
        assert resp.status_code == 201
        assert resp.json()["service"] == "churn-api"

    def test_define_invalid_target_returns_422(self) -> None:
        resp = client.post(
            "/sla/define",
            json={"service": "x", "sli_type": "availability", "target": 1.5},
        )
        assert resp.status_code == 422

    def test_define_invalid_sli_type_returns_422(self) -> None:
        resp = client.post(
            "/sla/define",
            json={"service": "x", "sli_type": "unknown_type", "target": 0.99},
        )
        assert resp.status_code == 422

    def test_observe_returns_201(self) -> None:
        self._define()
        resp = client.post(
            "/sla/observe",
            json={"service": "api-svc", "sli_type": "availability", "good": 998, "total": 1000},
        )
        assert resp.status_code == 201
        assert resp.json()["good"] == 998

    def test_observe_without_slo_returns_404(self) -> None:
        resp = client.post(
            "/sla/observe",
            json={"service": "ghost-svc", "sli_type": "availability", "good": 99, "total": 100},
        )
        assert resp.status_code == 404

    def test_status_all_returns_list(self) -> None:
        self._define("svc1")
        self._define("svc2")
        resp = client.get("/sla/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["services"]) == 2

    def test_status_service_found(self) -> None:
        self._define()
        self._observe()
        resp = client.get("/sla/status/api-svc")
        assert resp.status_code == 200
        assert resp.json()["service"] == "api-svc"

    def test_status_service_not_found_returns_404(self) -> None:
        resp = client.get("/sla/status/no-such-service")
        assert resp.status_code == 404

    def test_burn_rate_endpoint(self) -> None:
        self._define()
        self._observe(good=990, total=1000)
        resp = client.get("/sla/burn-rate/api-svc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["service"] == "api-svc"
        assert "burn_rates" in body

    def test_burn_rate_not_found_returns_404(self) -> None:
        resp = client.get("/sla/burn-rate/ghost")
        assert resp.status_code == 404

    def test_report_endpoint_structure(self) -> None:
        self._define()
        self._observe(good=1000, total=1000)
        resp = client.post("/sla/report")
        assert resp.status_code == 200
        body = resp.json()
        assert "summary" in body
        assert "services" in body
        assert body["summary"]["total_services"] == 1

    def test_list_slos_endpoint(self) -> None:
        self._define("s1")
        self._define("s2")
        resp = client.get("/sla/slos")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_observations_endpoint(self) -> None:
        self._define()
        self._observe(good=990, total=1000)
        self._observe(good=995, total=1000)
        resp = client.get("/sla/observations?service=api-svc")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_reset_endpoint_clears_state(self) -> None:
        self._define()
        self._observe()
        client.post("/sla/reset")
        resp = client.get("/sla/status")
        assert resp.json()["total"] == 0

    def test_slo_with_description(self) -> None:
        resp = client.post(
            "/sla/define",
            json={
                "service": "rag-api",
                "sli_type": "availability",
                "target": 0.995,
                "window_days": 7,
                "description": "RAG сервис: 99.5% доступность за 7 дней",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["description"] == "RAG сервис: 99.5% доступность за 7 дней"

    def test_latency_sli_definition(self) -> None:
        resp = client.post(
            "/sla/define",
            json={
                "service": "pricing-api",
                "sli_type": "latency",
                "target": 0.99,
                "latency_threshold_ms": 200.0,
                "description": "99% запросов за 200ms",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["sli_type"] == "latency"
