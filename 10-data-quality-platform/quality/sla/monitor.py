"""
SLAMonitor: центральный реестр SLO и SLI-наблюдений.

Central SLA monitoring registry — register SLOs, record SLI observations,
query error budgets and burn rates, generate compliance reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .budget import ErrorBudgetStatus, ErrorBudgetTracker
from .slo import SLIObservation, SLIType, SLODefinition

_MONITOR_SINGLETON: SLAMonitor | None = None


def get_monitor() -> SLAMonitor:
    """Singleton-accessor для FastAPI / Singleton accessor for FastAPI."""
    global _MONITOR_SINGLETON
    if _MONITOR_SINGLETON is None:
        _MONITOR_SINGLETON = SLAMonitor()
    return _MONITOR_SINGLETON


def reset_monitor() -> None:
    """Сбросить синглтон — только для тестов / Reset singleton — tests only."""
    global _MONITOR_SINGLETON
    _MONITOR_SINGLETON = None


@dataclass
class SLAComplianceReport:
    """Сводный отчёт о соблюдении SLA / Aggregate SLA compliance report."""

    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_services: int = 0
    compliant_services: int = 0
    non_compliant_services: int = 0
    critical_alerts: int = 0
    high_alerts: int = 0
    service_statuses: list[dict[str, Any]] = field(default_factory=list)

    @property
    def compliance_rate_pct(self) -> float:
        if self.total_services == 0:
            return 100.0
        return self.compliant_services / self.total_services * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": {
                "total_services": self.total_services,
                "compliant_services": self.compliant_services,
                "non_compliant_services": self.non_compliant_services,
                "compliance_rate_pct": round(self.compliance_rate_pct, 2),
                "critical_alerts": self.critical_alerts,
                "high_alerts": self.high_alerts,
            },
            "services": self.service_statuses,
        }


class SLAMonitor:
    """
    Центральный реестр SLO и отслеживания бюджетов ошибок.

    Register SLOs, record SLI observations, query error budget statuses,
    and generate aggregate compliance reports across all services.
    """

    def __init__(self) -> None:
        self._slos: dict[str, SLODefinition] = {}
        self._trackers: dict[str, ErrorBudgetTracker] = {}
        self._observations: list[SLIObservation] = []

    def define_slo(self, slo: SLODefinition) -> SLODefinition:
        """Зарегистрировать SLO / Register or replace an SLO definition."""
        key = f"{slo.service}:{slo.sli_type}"
        self._slos[key] = slo
        self._trackers[key] = ErrorBudgetTracker(slo)
        return slo

    def observe(self, obs: SLIObservation) -> SLIObservation:
        """
        Записать SLI-наблюдение / Record an SLI observation.

        Raises:
            ValueError: если для сервиса не зарегистрировано SLO.
        """
        key = f"{obs.service}:{obs.sli_type}"
        if key not in self._trackers:
            raise ValueError(
                f"No SLO defined for service='{obs.service}', sli_type='{obs.sli_type}'. "
                "Call define_slo() first."
            )
        self._trackers[key].record(obs.good, obs.total)
        self._observations.append(obs)
        return obs

    def get_status(
        self,
        service: str,
        sli_type: SLIType | None = None,
    ) -> list[ErrorBudgetStatus]:
        """Статус бюджета для сервиса (все SLI или конкретный)."""
        results = []
        for key, tracker in self._trackers.items():
            svc, sli = key.split(":", 1)
            if svc != service:
                continue
            if sli_type is not None and sli != sli_type.value:
                continue
            results.append(tracker.get_status())
        return results

    def get_all_statuses(self) -> list[ErrorBudgetStatus]:
        """Статус бюджета для всех зарегистрированных SLO."""
        return [t.get_status() for t in self._trackers.values()]

    def generate_report(self) -> SLAComplianceReport:
        """Сводный отчёт о соответствии SLA / Generate aggregate compliance report."""
        report = SLAComplianceReport()
        statuses = self.get_all_statuses()
        report.total_services = len(statuses)

        for status in statuses:
            if status.compliant:
                report.compliant_services += 1
            else:
                report.non_compliant_services += 1

            for alert in status.active_alerts:
                if alert.severity == "critical":
                    report.critical_alerts += 1
                elif alert.severity == "high":
                    report.high_alerts += 1

            report.service_statuses.append(status.to_dict())

        return report

    def list_slos(self) -> list[dict[str, Any]]:
        """Список всех зарегистрированных SLO / List all registered SLO definitions."""
        return [slo.to_dict() for slo in self._slos.values()]

    def get_recent_observations(
        self,
        service: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Последние SLI-наблюдения, опционально по сервису."""
        obs = (
            self._observations
            if service is None
            else [o for o in self._observations if o.service == service]
        )
        return [o.to_dict() for o in obs[-limit:]]
