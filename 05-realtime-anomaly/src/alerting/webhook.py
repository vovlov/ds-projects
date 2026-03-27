"""Webhook alerting for detected anomalies."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    timestamp: float
    metric_name: str
    value: float
    score: float
    threshold: float
    severity: str


def create_alert(
    timestamp: float,
    cpu: float,
    latency: float,
    requests: float,
    score: float,
    threshold: float,
) -> Alert:
    """Create an alert from anomaly detection result."""
    # Determine which metric is most anomalous
    metrics = {"cpu": cpu, "latency": latency, "requests": requests}
    worst_metric = max(metrics, key=lambda k: abs(metrics[k]))

    severity = "critical" if score > threshold * 2 else "warning"

    return Alert(
        timestamp=timestamp,
        metric_name=worst_metric,
        value=metrics[worst_metric],
        score=score,
        threshold=threshold,
        severity=severity,
    )


def format_alert_payload(alert: Alert) -> dict:
    """Format alert as webhook JSON payload."""
    return {
        "timestamp": datetime.fromtimestamp(alert.timestamp, tz=UTC).isoformat(),
        "severity": alert.severity,
        "metric": alert.metric_name,
        "value": alert.value,
        "anomaly_score": round(alert.score, 4),
        "threshold": alert.threshold,
        "message": (
            f"Anomaly detected in {alert.metric_name}: "
            f"value={alert.value:.2f}, score={alert.score:.2f} "
            f"(threshold={alert.threshold:.2f})"
        ),
    }


def send_alert(alert: Alert, webhook_url: str | None = None) -> bool:
    """Send alert via webhook. Logs if no URL configured."""
    payload = format_alert_payload(alert)

    if webhook_url:
        try:
            import urllib.request

            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Alert sent to {webhook_url}: {alert.severity} — {alert.metric_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    else:
        logger.warning(f"ALERT [{alert.severity}]: {payload['message']}")
        return True
