"""
Error budget tracking and multi-window burn rate alerting.

Реализует алертинг по скорости сжигания бюджета ошибок по Google SRE Workbook.
Implements Google SRE Workbook multi-window, multi-burn-rate alerting (Table 5.1).

Reference: https://sre.google/workbook/alerting-on-slos/
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .slo import SLODefinition

# Google SRE Table 5.1: (threshold, window_h, severity, budget_pct, response_time)
# Условие алерта: burn_rate >= threshold за последние window_h часов.
# Тригерим только ОДИН алерт (наивысший severity) — принцип shortest-meaningful-burn.
_BURN_RATE_RULES: list[tuple[float, float, str, float, str]] = [
    (14.4, 1.0, "critical", 2.0, "5min"),  # 2% бюджета за 1ч  → немедленно
    (6.0, 6.0, "high", 5.0, "30min"),  # 5% бюджета за 6ч  → срочно
    (3.0, 72.0, "medium", 10.0, "next_business_day"),  # 10% за 3д  → тикет
    (1.0, 168.0, "low", 100.0, "next_sprint"),  # медленное сжигание → предупреждение
]


@dataclass
class BurnRateAlert:
    """Алерт о превышении скорости сжигания бюджета / Burn rate threshold alert."""

    service: str
    severity: str
    burn_rate: float
    burn_rate_threshold: float
    budget_consumed_pct: float
    budget_remaining_pct: float
    window_hours: float
    error_rate: float
    slo_target: float
    response_time: str
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    triggered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "service": self.service,
            "severity": self.severity,
            "burn_rate": round(self.burn_rate, 3),
            "burn_rate_threshold": self.burn_rate_threshold,
            "budget_consumed_pct": round(self.budget_consumed_pct, 2),
            "budget_remaining_pct": round(self.budget_remaining_pct, 2),
            "window_hours": self.window_hours,
            "error_rate": round(self.error_rate, 6),
            "slo_target": self.slo_target,
            "response_time": self.response_time,
            "triggered_at": self.triggered_at,
            "message": (
                f"[{self.severity.upper()}] {self.service}: burn rate {self.burn_rate:.1f}x "
                f"({self.budget_consumed_pct:.1f}% budget consumed). "
                f"Required response: {self.response_time}"
            ),
        }


@dataclass
class ErrorBudgetStatus:
    """Текущее состояние бюджета ошибок / Current error budget status for a service."""

    service: str
    slo: SLODefinition
    total_good: int
    total_events: int
    error_budget_total: float
    error_budget_consumed: float
    error_budget_remaining: float
    burn_rate_1h: float
    burn_rate_6h: float
    burn_rate_3d: float
    active_alerts: list[BurnRateAlert]
    compliant: bool
    projected_exhaustion_hours: float | None

    @property
    def budget_remaining_pct(self) -> float:
        if self.error_budget_total <= 0:
            return 100.0
        return self.error_budget_remaining / self.error_budget_total * 100

    @property
    def budget_consumed_pct(self) -> float:
        return 100.0 - self.budget_remaining_pct

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "slo": self.slo.to_dict(),
            "observations": {"good": self.total_good, "total": self.total_events},
            "error_budget": {
                "total_fraction": round(self.error_budget_total, 6),
                "consumed_fraction": round(self.error_budget_consumed, 6),
                "remaining_fraction": round(self.error_budget_remaining, 6),
                "consumed_pct": round(self.budget_consumed_pct, 2),
                "remaining_pct": round(self.budget_remaining_pct, 2),
                "remaining_minutes": round(
                    self.error_budget_remaining * self.slo.window_days * 24 * 60, 2
                ),
            },
            "burn_rates": {
                "1h": round(self.burn_rate_1h, 4),
                "6h": round(self.burn_rate_6h, 4),
                "3d": round(self.burn_rate_3d, 4),
            },
            "compliant": self.compliant,
            "projected_exhaustion_hours": (
                round(self.projected_exhaustion_hours, 1)
                if self.projected_exhaustion_hours is not None
                else None
            ),
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "alert_count": len(self.active_alerts),
        }


class ErrorBudgetTracker:
    """
    Отслеживает бюджет ошибок и вычисляет burn rate по нескольким окнам.

    Tracks error budget consumption and computes multi-window burn rates
    per Google SRE Workbook Chapter 5. Observations are kept in a rolling
    deque (max 10 000 entries) to bound memory usage.
    """

    _MAX_OBSERVATIONS = 10_000

    def __init__(self, slo: SLODefinition) -> None:
        self.slo = slo
        # (timestamp_utc, good_count, total_count)
        self._obs: deque[tuple[datetime, int, int]] = deque(maxlen=self._MAX_OBSERVATIONS)

    def record(self, good: int, total: int, at: datetime | None = None) -> None:
        """Записать порцию событий / Record a batch of SLI events."""
        ts = at if at is not None else datetime.now(UTC)
        self._obs.append((ts, good, total))

    def _window_totals(self, hours: float) -> tuple[int, int]:
        """Суммировать good/total за последние N часов / Sum good/total over last N hours."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        good = total = 0
        for ts, g, t in self._obs:
            if ts >= cutoff:
                good += g
                total += t
        return good, total

    def _error_rate(self, hours: float) -> float:
        good, total = self._window_totals(hours)
        if total == 0:
            return 0.0
        return (total - good) / total

    def _burn_rate(self, hours: float) -> float:
        """
        Burn rate = error_rate / error_budget_fraction.
        > 1 означает, что бюджет тратится быстрее нормы.
        """
        budget = self.slo.error_budget_fraction
        if budget <= 0:
            return 0.0
        return self._error_rate(hours) / budget

    def get_status(self) -> ErrorBudgetStatus:
        """Вычислить полный статус бюджета с multi-window burn rates."""
        total_good = sum(g for _, g, _ in self._obs)
        total_events = sum(t for _, _, t in self._obs)

        overall_error_rate = (total_events - total_good) / total_events if total_events > 0 else 0.0

        budget_total = self.slo.error_budget_fraction
        budget_consumed = min(overall_error_rate, budget_total)
        budget_remaining = max(0.0, budget_total - budget_consumed)

        burn_1h = self._burn_rate(1.0)
        burn_6h = self._burn_rate(6.0)
        burn_3d = self._burn_rate(72.0)

        consumed_pct = (budget_consumed / budget_total * 100) if budget_total > 0 else 0.0
        remaining_pct = 100.0 - consumed_pct

        active_alerts: list[BurnRateAlert] = []
        for threshold, window_h, severity, _budget_pct, response in _BURN_RATE_RULES:
            rate = self._burn_rate(window_h)
            if rate >= threshold:
                active_alerts.append(
                    BurnRateAlert(
                        service=self.slo.service,
                        severity=severity,
                        burn_rate=rate,
                        burn_rate_threshold=threshold,
                        budget_consumed_pct=consumed_pct,
                        budget_remaining_pct=remaining_pct,
                        window_hours=window_h,
                        error_rate=self._error_rate(window_h),
                        slo_target=self.slo.target,
                        response_time=response,
                    )
                )
                break  # emit only the highest-severity alert

        # Проецируем время до исчерпания бюджета по текущему 1h burn rate.
        if burn_1h > 0 and budget_remaining > 0 and budget_total > 0:
            hours_per_window = self.slo.window_days * 24.0
            projected: float | None = (budget_remaining / budget_total) / burn_1h * hours_per_window
        else:
            projected = None

        return ErrorBudgetStatus(
            service=self.slo.service,
            slo=self.slo,
            total_good=total_good,
            total_events=total_events,
            error_budget_total=budget_total,
            error_budget_consumed=budget_consumed,
            error_budget_remaining=budget_remaining,
            burn_rate_1h=burn_1h,
            burn_rate_6h=burn_6h,
            burn_rate_3d=burn_3d,
            active_alerts=active_alerts,
            compliant=budget_consumed <= budget_total,
            projected_exhaustion_hours=projected,
        )
