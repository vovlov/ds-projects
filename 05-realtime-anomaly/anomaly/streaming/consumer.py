"""Kafka metric stream consumer — detects anomalies in real-time."""

from __future__ import annotations

import json
import logging

import numpy as np

from ..models.detector import MultiMetricDetector

logger = logging.getLogger(__name__)


def consume_and_detect(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "metrics",
    group_id: str = "anomaly-detector",
    window_size: int = 50,
    threshold_sigma: float = 3.0,
) -> None:
    """Consume metrics from Kafka and detect anomalies."""
    try:
        from kafka import KafkaConsumer
    except ImportError:
        logger.error("kafka-python not installed. Run: pip install kafka-python")
        return

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
    )

    detector = MultiMetricDetector(window_size=window_size, threshold_sigma=threshold_sigma)
    buffer: list[dict] = []

    logger.info(f"Consuming from '{topic}', window={window_size}, σ={threshold_sigma}")

    for message in consumer:
        point = message.value
        buffer.append(point)

        # Need at least window_size+1 points for detection
        if len(buffer) > window_size + 1:
            buffer = buffer[-(window_size + 10) :]

            data = {
                "cpu": np.array([p["cpu"] for p in buffer]),
                "latency": np.array([p["latency"] for p in buffer]),
                "requests": np.array([p["requests"] for p in buffer]),
            }

            result = detector.detect(data)

            # Check last point
            if result.predictions[-1] == 1:
                logger.warning(
                    f"ANOMALY at ts={point['timestamp']:.0f}: "
                    f"cpu={point['cpu']:.1f}, latency={point['latency']:.1f}, "
                    f"requests={point['requests']:.0f}, score={result.scores[-1]:.2f}"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    consume_and_detect()
