"""
SLO/SLA definitions and SLI observation dataclasses.

SLI (Service Level Indicator) — измеренная метрика качества сервиса.
SLO (Service Level Objective) — целевое значение SLI (например, 99.9% доступность).
SLA (Service Level Agreement) — договорное SLO с последствиями нарушения.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class SLIType(StrEnum):
    """Типы индикаторов уровня сервиса / Service Level Indicator types."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class SLODefinition:
    """
    Определение SLO для сервиса / SLO definition for a service.

    Args:
        service: уникальный идентификатор сервиса
        sli_type: тип измеряемого показателя
        target: доля успешных запросов (0 < target < 1), например 0.999 = 99.9%
        window_days: окно compliance-периода (обычно 30 дней)
        latency_threshold_ms: порог latency для LATENCY SLI
        description: описание SLO для документации
    """

    service: str
    sli_type: SLIType
    target: float
    window_days: int = 30
    latency_threshold_ms: float | None = None
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not 0 < self.target < 1:
            raise ValueError(f"SLO target must be between 0 and 1, got {self.target}")
        if self.sli_type == SLIType.LATENCY and self.latency_threshold_ms is None:
            raise ValueError("LATENCY SLI requires latency_threshold_ms")

    @property
    def error_budget_fraction(self) -> float:
        """Допустимая доля ошибок / Fraction of requests allowed to be bad."""
        return 1.0 - self.target

    @property
    def error_budget_minutes(self) -> float:
        """Бюджет ошибок в минутах за окно / Error budget in minutes over the window."""
        return self.window_days * 24 * 60 * self.error_budget_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "sli_type": self.sli_type.value,
            "target": self.target,
            "target_pct": f"{self.target * 100:.3f}%",
            "error_budget_fraction": round(self.error_budget_fraction, 6),
            "error_budget_minutes": round(self.error_budget_minutes, 2),
            "window_days": self.window_days,
            "latency_threshold_ms": self.latency_threshold_ms,
            "description": self.description,
            "created_at": self.created_at,
        }


@dataclass
class SLIObservation:
    """
    Одно измерение SLI / Single SLI measurement batch.

    Args:
        service: сервис, которому принадлежит измерение
        sli_type: тип SLI
        good: количество «хороших» событий (успешных запросов)
        total: общее количество событий
    """

    service: str
    sli_type: SLIType
    good: int
    total: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sli_value(self) -> float:
        """SLI = good / total; 1.0 если нет событий."""
        if self.total == 0:
            return 1.0
        return self.good / self.total

    @property
    def error_count(self) -> int:
        return self.total - self.good

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "service": self.service,
            "sli_type": self.sli_type.value,
            "good": self.good,
            "total": self.total,
            "sli_value": round(self.sli_value, 6),
            "error_count": self.error_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
