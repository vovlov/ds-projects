"""Synthetic time series data generator with anomalies."""

from __future__ import annotations

import math
import random

import numpy as np


def generate_timeseries(
    n_points: int = 2000,
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> dict:
    """Generate synthetic metric time series with injected anomalies.

    Simulates CPU load, latency, and request count metrics.
    Anomalies are injected as spikes, level shifts, and pattern breaks.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    timestamps = np.arange(n_points, dtype=float)

    # Base signals with seasonality
    cpu = (
        40
        + 20 * np.sin(2 * math.pi * timestamps / 288)  # daily cycle
        + 5 * np.sin(2 * math.pi * timestamps / 48)  # sub-daily
        + np_rng.normal(0, 2, n_points)  # noise
    )

    latency = (
        50
        + 15 * np.sin(2 * math.pi * timestamps / 288)
        + np_rng.exponential(3, n_points)  # right-skewed noise
    )

    requests = (
        1000
        + 300 * np.sin(2 * math.pi * timestamps / 288)
        + 100 * np.sin(2 * math.pi * timestamps / 48)
        + np_rng.poisson(10, n_points)
    ).astype(float)

    # Inject anomalies
    labels = np.zeros(n_points, dtype=int)
    n_anomalies = int(n_points * anomaly_rate)

    anomaly_indices = sorted(rng.sample(range(50, n_points - 50), n_anomalies))

    for idx in anomaly_indices:
        anomaly_type = rng.choice(["spike", "level_shift", "drop"])
        duration = rng.randint(1, 5)
        end = min(idx + duration, n_points)

        if anomaly_type == "spike":
            cpu[idx:end] += np_rng.uniform(30, 50)
            latency[idx:end] *= np_rng.uniform(2, 5)
        elif anomaly_type == "level_shift":
            cpu[idx:end] += 25
            requests[idx:end] *= 0.3
        elif anomaly_type == "drop":
            requests[idx:end] = np_rng.uniform(0, 50, end - idx)
            latency[idx:end] *= np_rng.uniform(3, 8)

        labels[idx:end] = 1

    # Clip to realistic ranges
    cpu = np.clip(cpu, 0, 100)
    latency = np.clip(latency, 1, 5000)
    requests = np.clip(requests, 0, None)

    return {
        "timestamps": timestamps,
        "cpu": cpu,
        "latency": latency,
        "requests": requests,
        "labels": labels,
    }


def to_windows(
    data: dict,
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert time series to sliding windows for model input.

    Returns:
        X: (n_windows, window_size, n_features)
        y: (n_windows,) — 1 if any anomaly in window, else 0
    """
    features = np.column_stack([data["cpu"], data["latency"], data["requests"]])
    labels = data["labels"]
    n = len(labels)

    windows = []
    window_labels = []

    for i in range(0, n - window_size, stride):
        windows.append(features[i : i + window_size])
        # Label = 1 if any point in window is anomalous
        window_labels.append(int(labels[i : i + window_size].max()))

    X = np.array(windows)
    y = np.array(window_labels)
    return X, y
