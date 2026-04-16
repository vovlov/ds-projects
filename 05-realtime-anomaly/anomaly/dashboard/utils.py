"""Утилиты для Streamlit-дашборда аномалий (без зависимости от streamlit).

Вынесены в отдельный модуль чтобы:
- их можно было протестировать без запуска Streamlit-сервера
- логика генерации данных была переиспользуема
"""

from __future__ import annotations

import numpy as np


def generate_metric_stream(
    n_points: int = 200,
    inject_anomaly: bool = False,
    anomaly_start: int = 150,
    anomaly_magnitude: float = 5.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Генерирует синтетический поток метрик SRE-стека.

    CPU: базовый уровень ~40%, суточный синус, шум.
    Latency: ~80ms, коррелирует с CPU.
    Requests: ~500 rps, бизнес-паттерн.

    Если inject_anomaly=True, после anomaly_start точки добавляем резкий
    скачок во всех метриках — имитация инцидента (DDoS, memory leak и т.д.).

    Args:
        n_points: количество точек временного ряда
        inject_anomaly: добавить ли инцидент в конце серии
        anomaly_start: с какой точки начинается аномалия
        anomaly_magnitude: в единицах std — насколько сильна аномалия
        seed: seed для воспроизводимости

    Returns:
        dict с ключами "cpu", "latency", "requests" — numpy массивы длиной n_points
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_points)

    # CPU: синус + шум, имитируем суточный паттерн
    cpu = 40.0 + 15.0 * np.sin(2 * np.pi * t / n_points) + rng.normal(0, 3, n_points)
    cpu = np.clip(cpu, 5, 95)

    # Latency: коррелирует с CPU + собственный шум
    latency = 80.0 + 0.5 * (cpu - 40.0) + rng.normal(0, 5, n_points)
    latency = np.clip(latency, 10, 500)

    # Requests: независимый поток с умеренным шумом
    requests = 500.0 + 50.0 * np.sin(2 * np.pi * t / (n_points / 3)) + rng.normal(0, 20, n_points)
    requests = np.clip(requests, 100, 1000)

    if inject_anomaly and anomaly_start < n_points:
        # Инцидент: резкий скачок всех метрик — имитация реального инцидента.
        # Если anomaly_start=0, используем std всего ряда как fallback.
        anomaly_len = n_points - anomaly_start
        ref_slice = cpu[:anomaly_start] if anomaly_start > 0 else cpu
        cpu_std = float(np.std(ref_slice)) or 3.0  # fallback 3% если std=0
        ref_slice = latency[:anomaly_start] if anomaly_start > 0 else latency
        latency_std = float(np.std(ref_slice)) or 5.0
        ref_slice = requests[:anomaly_start] if anomaly_start > 0 else requests
        requests_std = float(np.std(ref_slice)) or 20.0

        cpu[anomaly_start:] += anomaly_magnitude * cpu_std
        latency[anomaly_start:] += anomaly_magnitude * latency_std
        # Requests при инциденте падают (circuit breaker)
        requests[anomaly_start:] -= anomaly_magnitude * requests_std * 0.5

        # Добавляем шум на аномальный участок
        cpu[anomaly_start:] += rng.normal(0, cpu_std * 0.5, anomaly_len)
        latency[anomaly_start:] += rng.normal(0, latency_std * 0.5, anomaly_len)
        requests[anomaly_start:] += rng.normal(0, requests_std * 0.5, anomaly_len)

        cpu = np.clip(cpu, 5, 100)
        latency = np.clip(latency, 10, 2000)
        requests = np.clip(requests, 0, 1500)

    return {"cpu": cpu, "latency": latency, "requests": requests}


def compute_detection_summary(
    predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    """Вычисляет агрегированную статистику по результатам детекции.

    Args:
        predictions: бинарный массив (0/1), 1 = аномалия
        scores: непрерывные Z-score значения

    Returns:
        dict с метриками: n_total, n_anomalies, anomaly_rate, max_score, mean_score
    """
    n_total = len(predictions)
    n_anomalies = int(np.sum(predictions))
    anomaly_rate = n_anomalies / n_total if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_anomalies": n_anomalies,
        "anomaly_rate": anomaly_rate,
        # При пустом scores np.max/mean вернут nan — заменяем на 0 для UI
        "max_score": float(np.max(scores)) if n_total > 0 else 0.0,
        "mean_score": float(np.mean(scores)) if n_total > 0 else 0.0,
    }


def generate_reference_data(
    n_points: int = 200,
    seed: int = 0,
) -> list[list[float]]:
    """Генерирует нормальные данные для MMD reference (без аномалий).

    Returns:
        Список точек [[cpu, latency, requests], ...] для /drift/check endpoint
    """
    stream = generate_metric_stream(n_points=n_points, inject_anomaly=False, seed=seed)
    return [
        [float(stream["cpu"][i]), float(stream["latency"][i]), float(stream["requests"][i])]
        for i in range(n_points)
    ]


def generate_current_data(
    n_points: int = 100,
    inject_drift: bool = False,
    drift_magnitude: float = 4.0,
    seed: int = 99,
) -> list[list[float]]:
    """Генерирует текущие данные для MMD current — с дрейфом или без.

    Returns:
        Список точек [[cpu, latency, requests], ...] для /drift/check endpoint
    """
    stream = generate_metric_stream(
        n_points=n_points,
        inject_anomaly=inject_drift,
        anomaly_start=0,  # дрейф с самого начала текущего окна
        anomaly_magnitude=drift_magnitude,
        seed=seed,
    )
    return [
        [float(stream["cpu"][i]), float(stream["latency"][i]), float(stream["requests"][i])]
        for i in range(n_points)
    ]
