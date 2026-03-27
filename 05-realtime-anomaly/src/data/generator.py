"""
Генератор синтетических временных рядов инфраструктурных метрик.

Имитирует три типичных метрики сервера:
- CPU load (%) — с дневной и суб-дневной сезонностью
- Latency (ms) — с right-skewed шумом (latency не бывает нормально распределён)
- Request count — с пуассоновским шумом поверх сезонности

Аномалии инжектируются тремя способами — это отражает реальные инциденты:
- Spike: утечка памяти или DDoS → резкий рост CPU и latency
- Level shift: деплой с багом → стабильно повышенная нагрузка
- Drop: отвалился upstream-сервис → трафик упал, latency вырос
"""

from __future__ import annotations

import math
import random

import numpy as np


def generate_timeseries(
    n_points: int = 2000,
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> dict:
    """Сгенерировать временной ряд с инжектированными аномалиями.

    Каждая точка — условная «минута». 2000 точек ≈ 33 часа.
    288 точек — один «день» (условный цикл сезонности).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    timestamps = np.arange(n_points, dtype=float)

    # Базовые сигналы с сезонностью
    # CPU: фоновая нагрузка ~40%, дневной пик +20%, суб-дневные колебания +5%
    cpu = (
        40
        + 20 * np.sin(2 * math.pi * timestamps / 288)
        + 5 * np.sin(2 * math.pi * timestamps / 48)
        + np_rng.normal(0, 2, n_points)
    )

    # Latency: базовая ~50ms, сезонность, экспоненциальный шум
    # (latency right-skewed по природе — бывают редкие большие значения)
    latency = 50 + 15 * np.sin(2 * math.pi * timestamps / 288) + np_rng.exponential(3, n_points)

    # Requests: ~1000 rpm, дневная и суб-дневная сезонность, пуассоновский шум
    requests = (
        1000
        + 300 * np.sin(2 * math.pi * timestamps / 288)
        + 100 * np.sin(2 * math.pi * timestamps / 48)
        + np_rng.poisson(10, n_points)
    ).astype(float)

    # Инжектируем аномалии
    labels = np.zeros(n_points, dtype=int)
    n_anomalies = int(n_points * anomaly_rate)

    # Не ставим аномалии в начало/конец — нужен контекст для детектора
    anomaly_indices = sorted(rng.sample(range(50, n_points - 50), n_anomalies))

    for idx in anomaly_indices:
        anomaly_type = rng.choice(["spike", "level_shift", "drop"])
        duration = rng.randint(1, 5)
        end = min(idx + duration, n_points)

        if anomaly_type == "spike":
            # Утечка памяти / DDoS: CPU и latency резко растут
            cpu[idx:end] += np_rng.uniform(30, 50)
            latency[idx:end] *= np_rng.uniform(2, 5)
        elif anomaly_type == "level_shift":
            # Плохой деплой: CPU повышается, трафик падает
            cpu[idx:end] += 25
            requests[idx:end] *= 0.3
        elif anomaly_type == "drop":
            # Отвалился upstream: трафик обвалился, latency вырос
            requests[idx:end] = np_rng.uniform(0, 50, end - idx)
            latency[idx:end] *= np_rng.uniform(3, 8)

        labels[idx:end] = 1

    # Ограничиваем физически возможными диапазонами
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
    """Нарезать временной ряд на скользящие окна для входа модели.

    Каждое окно — это (window_size, 3) матрица [cpu, latency, requests].
    Метка окна = 1, если хотя бы одна точка в окне аномальна.
    """
    features = np.column_stack([data["cpu"], data["latency"], data["requests"]])
    labels = data["labels"]
    n = len(labels)

    windows = []
    window_labels = []

    for i in range(0, n - window_size, stride):
        windows.append(features[i : i + window_size])
        window_labels.append(int(labels[i : i + window_size].max()))

    X = np.array(windows)
    y = np.array(window_labels)
    return X, y
