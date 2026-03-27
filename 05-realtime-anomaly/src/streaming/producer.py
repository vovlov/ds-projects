"""Kafka metric stream producer — simulates infrastructure metrics."""

from __future__ import annotations

import json
import logging
import time

from ..data.generator import generate_timeseries

logger = logging.getLogger(__name__)


def produce_metrics(
    bootstrap_servers: str = "localhost:9092",
    topic: str = "metrics",
    delay: float = 0.1,
    n_points: int = 2000,
    anomaly_rate: float = 0.03,
) -> None:
    """Stream synthetic metrics to Kafka topic."""
    try:
        from kafka import KafkaProducer
    except ImportError:
        logger.error("kafka-python not installed. Run: pip install kafka-python")
        return

    data = generate_timeseries(n_points=n_points, anomaly_rate=anomaly_rate)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    logger.info(f"Producing {n_points} metrics to topic '{topic}'...")
    for i in range(n_points):
        message = {
            "timestamp": float(data["timestamps"][i]),
            "cpu": float(data["cpu"][i]),
            "latency": float(data["latency"][i]),
            "requests": float(data["requests"][i]),
            "is_anomaly": int(data["labels"][i]),
        }
        producer.send(topic, value=message)

        if (i + 1) % 100 == 0:
            logger.info(f"Sent {i + 1}/{n_points} metrics")

        time.sleep(delay)

    producer.flush()
    producer.close()
    logger.info("Done producing metrics")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    produce_metrics()
