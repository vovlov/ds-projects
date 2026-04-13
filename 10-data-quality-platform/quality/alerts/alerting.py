"""
Система алертинга при обнаружении дрейфа данных.
Data drift alerting system.

Принцип работы: AlertManager принимает drift-отчёт от detect_drift(),
создаёт структурированный DriftAlert и рассылает его по каналам
(лог, webhook, будущие: Slack, PagerDuty, email).

Webhook-канал специально проектировался под /retraining/notify Project 01
(Customer Churn MLOps), чтобы замкнуть цикл мониторинга:
  качество данных (Project 10) → алерт → переобучение (Project 01)

Design: AlertManager receives a drift report from detect_drift(),
creates a structured DriftAlert, and routes it to configured channels
(log, webhook). The webhook channel targets Project 01's
/retraining/notify endpoint to close the monitoring loop:
  data quality (Project 10) → alert → retraining (Project 01)

Sources:
  - Evidently AI alerting patterns (2025)
  - MLOps Community: "Alert Routing for ML Systems" (2026)
  - DataOps Manifesto: observable pipelines
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Пороги серьёзности по PSI (совместимы с BCBS-стандартом из Project 01)
# Severity thresholds matching Project 01's BCBS standard
_PSI_THRESHOLDS = {
    "ok": 0.0,
    "warning": 0.1,
    "critical": 0.25,
}

# Порядок серьёзности для сравнения / Severity ordering for comparison
_SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


@dataclass
class DriftAlert:
    """Структурированный алерт о дрейфе данных.
    Structured data drift alert.

    Attributes:
        severity: Уровень критичности: "ok" | "warning" | "critical".
        features_drifted: Список признаков с обнаруженным дрейфом.
        max_psi: Максимальное значение PSI среди всех признаков.
        columns_checked: Сколько столбцов проверено.
        columns_with_drift: Сколько столбцов показали дрейф.
        timestamp: ISO-8601 timestamp создания алерта (UTC).
        source: Имя источника (сервис/пайплайн, создавший алерт).
        details: Полный drift-отчёт для аудит-трейла.
    """

    severity: str
    features_drifted: list[str]
    max_psi: float
    columns_checked: int
    columns_with_drift: int
    timestamp: str
    source: str = "data-quality-platform"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать в словарь для JSON-передачи / Serialize to dict."""
        return {
            "severity": self.severity,
            "features_drifted": self.features_drifted,
            "max_psi": self.max_psi,
            "columns_checked": self.columns_checked,
            "columns_with_drift": self.columns_with_drift,
            "timestamp": self.timestamp,
            "source": self.source,
            "details": self.details,
        }

    def is_actionable(self, threshold: str = "warning") -> bool:
        """Требует ли алерт реакции (severity >= threshold)?
        Does the alert require action (severity >= threshold)?
        """
        return _SEVERITY_ORDER.get(self.severity, 0) >= _SEVERITY_ORDER.get(threshold, 1)


def _compute_severity(max_psi: float) -> str:
    """Определить серьёзность по PSI / Map PSI value to severity level."""
    if max_psi >= _PSI_THRESHOLDS["critical"]:
        return "critical"
    if max_psi >= _PSI_THRESHOLDS["warning"]:
        return "warning"
    return "ok"


def create_alert_from_report(
    drift_report: dict[str, Any],
    source: str = "data-quality-platform",
) -> DriftAlert:
    """Создать DriftAlert из отчёта detect_drift().
    Create DriftAlert from detect_drift() report.

    Args:
        drift_report: Словарь из detect_drift() (keys: drift_detected,
                      columns_checked, columns_with_drift, details).
        source: Идентификатор источника алерта.

    Returns:
        DriftAlert с вычисленной серьёзностью и списком затронутых фич.
    """
    details = drift_report.get("details", [])

    # Собираем признаки с дрейфом и максимальный PSI
    # Collect drifted features and maximum PSI across all columns
    features_drifted: list[str] = []
    max_psi = 0.0

    for col_report in details:
        if col_report.get("drift_detected", False):
            features_drifted.append(col_report["column"])
        col_psi = col_report.get("psi", 0.0)
        if col_psi > max_psi:
            max_psi = col_psi

    severity = _compute_severity(max_psi)

    return DriftAlert(
        severity=severity,
        features_drifted=features_drifted,
        max_psi=round(max_psi, 6),
        columns_checked=drift_report.get("columns_checked", len(details)),
        columns_with_drift=drift_report.get("columns_with_drift", len(features_drifted)),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
        details=drift_report,
    )


# ---------------------------------------------------------------------------
# Протокол канала алертинга / Alert channel protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AlertChannel(Protocol):
    """Интерфейс канала доставки алертов / Alert delivery channel interface."""

    def send(self, alert: DriftAlert) -> bool:
        """Отправить алерт. Вернуть True при успехе / Send alert, return True on success."""
        ...


# ---------------------------------------------------------------------------
# Каналы / Channels
# ---------------------------------------------------------------------------


class LogAlertChannel:
    """Канал алертинга через Python logger.
    Delivers alerts via Python logger (always available, no dependencies).

    Используется как fallback и для аудит-трейла.
    Severity warning → logger.warning; critical → logger.error.
    """

    def __init__(self, name: str = "drift.alert") -> None:
        self._log = logging.getLogger(name)

    def send(self, alert: DriftAlert) -> bool:
        """Записать алерт в лог / Write alert to log."""
        msg = (
            f"[DRIFT ALERT] severity={alert.severity} "
            f"max_psi={alert.max_psi:.4f} "
            f"features_drifted={alert.features_drifted} "
            f"columns={alert.columns_with_drift}/{alert.columns_checked} "
            f"source={alert.source} ts={alert.timestamp}"
        )
        if alert.severity == "critical":
            self._log.error(msg)
        elif alert.severity == "warning":
            self._log.warning(msg)
        else:
            self._log.info(msg)
        return True


class WebhookAlertChannel:
    """Канал алертинга через HTTP-webhook (POST JSON).
    HTTP webhook alert channel (POST JSON payload).

    Разработан для интеграции с /retraining/notify Project 01 (Churn MLOps),
    чтобы автоматически инициировать переобучение при дрейфе.

    Designed for integration with Project 01's /retraining/notify endpoint
    to automatically trigger retraining when drift is detected.

    Graceful degradation: если httpx недоступен или webhook недостижим —
    логируем предупреждение и возвращаем False (не кидаем исключение).
    """

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Инициализировать webhook-канал / Initialize webhook channel.

        Args:
            url: URL webhook-эндпоинта (например, http://churn-svc/retraining/notify).
            timeout: Таймаут запроса в секундах (дефолт 5с).
            headers: Дополнительные HTTP-заголовки (авторизация и т.д.).
        """
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}

    def send(self, alert: DriftAlert) -> bool:
        """Отправить алерт на webhook URL / Send alert to webhook URL."""
        if not self.is_available():
            logger.warning("WebhookAlertChannel: httpx not installed, skipping webhook")
            return False

        import httpx

        payload = alert.to_dict()
        try:
            response = httpx.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json", **self.headers},
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.info("Drift alert sent to webhook %s: status=%d", self.url, response.status_code)
            return True
        except Exception as exc:
            # Не критично: webhook может быть недоступен в CI или при тестировании
            # Not critical: webhook may be unavailable in CI or during testing
            logger.warning("Drift alert webhook failed (%s): %s", self.url, exc)
            return False

    @staticmethod
    def is_available() -> bool:
        """Проверить наличие httpx / Check if httpx is installed."""
        try:
            import httpx  # noqa: F401

            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------


class AlertManager:
    """Оркестратор алертинга: создаёт алерт из drift-отчёта и рассылает по каналам.
    Orchestrates alerting: creates alert from drift report and routes to channels.

    Пример / Example:
        manager = AlertManager(
            channels=[
                LogAlertChannel(),
                WebhookAlertChannel("http://churn-svc:8000/retraining/notify"),
            ],
            severity_threshold="warning",  # только warning и critical
        )
        alert = manager.process_drift_report(drift_report)
        if alert:
            print(f"Alert sent: {alert.severity}")
    """

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        severity_threshold: str = "warning",
        source: str = "data-quality-platform",
    ) -> None:
        """Инициализировать AlertManager / Initialize AlertManager.

        Args:
            channels: Список каналов доставки алертов.
                      По умолчанию только LogAlertChannel.
            severity_threshold: Минимальная серьёзность для отправки алерта.
                                "warning" — слать при moderate и critical drift.
                                "critical" — только при серьёзном дрейфе.
            source: Имя источника для метаданных алерта.
        """
        self.channels: list[AlertChannel] = (
            channels if channels is not None else [LogAlertChannel()]
        )
        self.severity_threshold = severity_threshold
        self.source = source

    def process_drift_report(
        self,
        drift_report: dict[str, Any],
    ) -> DriftAlert | None:
        """Создать алерт из drift-отчёта и разослать по каналам (если нужно).
        Create alert from drift report and dispatch to channels if actionable.

        Args:
            drift_report: Выход detect_drift() из quality.quality.drift.

        Returns:
            DriftAlert если алерт был создан и отправлен; None если нет дрейфа
            или серьёзность ниже порога.
        """
        alert = create_alert_from_report(drift_report, source=self.source)

        if not alert.is_actionable(self.severity_threshold):
            logger.debug(
                "Drift severity=%s below threshold=%s — no alert sent",
                alert.severity,
                self.severity_threshold,
            )
            return None

        logger.info(
            "Dispatching drift alert: severity=%s max_psi=%.4f to %d channel(s)",
            alert.severity,
            alert.max_psi,
            len(self.channels),
        )

        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as exc:
                # Изолируем ошибки каналов — один сбойный канал не блокирует остальные
                # Isolate channel errors — one failed channel doesn't block others
                logger.error("Alert channel %s failed: %s", type(channel).__name__, exc)

        return alert
