"""Inference cost tracking and cloud cost estimation.

Tracks per-request latency with a rolling window and estimates monthly
cloud costs based on observed throughput vs on-demand EC2/GCP pricing.

Cloud pricing baseline (2026):
    - AWS c6i.large  : $0.085/hr  (2 vCPU, 4 GiB)  — standard ML serving
    - AWS c6i.xlarge : $0.170/hr  (4 vCPU, 8 GiB)  — higher-throughput
    - GCP n2-standard-2: $0.097/hr (2 vCPU, 8 GiB)
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

# Cloud pricing constants (USD per hour, on-demand 2026)
_CLOUD_PRICING: dict[str, float] = {
    "aws_c6i_large": 0.085,
    "aws_c6i_xlarge": 0.170,
    "gcp_n2_standard_2": 0.097,
    "azure_f2s_v2": 0.099,
}

# Predefined RPS thresholds for instance sizing recommendations
_INSTANCE_SIZING: list[tuple[float, str]] = [
    (10.0, "aws_c6i_large"),
    (25.0, "aws_c6i_xlarge"),
    (float("inf"), "aws_c6i_xlarge"),  # scale horizontally above this
]


@dataclass
class LatencyStats:
    """Статистика задержки инференса / Inference latency statistics.

    Attributes:
        n_requests: Количество запросов в окне.
        p50_ms: Медианная задержка (мс).
        p95_ms: 95-й перцентиль задержки (мс).
        p99_ms: 99-й перцентиль задержки (мс).
        mean_ms: Средняя задержка (мс).
        max_ms: Максимальная задержка (мс).
        throughput_rps: Фактический throughput (запросов/сек).
    """

    n_requests: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    throughput_rps: float


@dataclass
class CostEstimate:
    """Оценка месячной стоимости в облаке / Monthly cloud cost estimate.

    Attributes:
        rps: Запросов в секунду (входной параметр или измеренный).
        instance_type: Рекомендуемый тип инстанса.
        cost_per_hour_usd: Стоимость в час (USD).
        cost_per_month_usd: Стоимость в месяц (USD, 720 часов).
        cost_per_million_requests_usd: Стоимость на 1M запросов (USD).
        n_instances_recommended: Количество инстансов для RPS.
        notes: Текстовые рекомендации.
    """

    rps: float
    instance_type: str
    cost_per_hour_usd: float
    cost_per_month_usd: float
    cost_per_million_requests_usd: float
    n_instances_recommended: int
    notes: str


@dataclass
class BatchOptimizationResult:
    """Результаты оптимизации размера батча / Batch size optimization results.

    Attributes:
        optimal_batch_size: Рекомендуемый размер батча.
        throughput_rps: Throughput при оптимальном батче.
        latency_p95_ms: P95 задержка при оптимальном батче (оценка).
        efficiency_score: Оценка эффективности (0-1, throughput vs latency trade-off).
        recommendations: Список рекомендаций.
    """

    optimal_batch_size: int
    throughput_rps: float
    latency_p95_ms: float
    efficiency_score: float
    recommendations: list[str] = field(default_factory=list)


class CostTracker:
    """Трекер стоимости инференса с скользящим окном наблюдений.

    Inference cost tracker with rolling observation window.

    Собирает задержку каждого запроса (через context manager `track()`),
    вычисляет перцентили и оценивает месячную стоимость в облаке.

    Collects per-request latency via `track()` context manager,
    computes percentiles, and estimates monthly cloud cost.

    Args:
        window_size: Размер скользящего окна (количество последних запросов).
    """

    def __init__(self, window_size: int = 1000) -> None:
        self._window_size = window_size
        self._latencies_ms: deque[float] = deque(maxlen=window_size)
        self._window_start_time: float = time.monotonic()
        self._total_requests: int = 0

    @contextmanager
    def track(self) -> Generator[None, None, None]:
        """Context manager для измерения задержки одного запроса.

        Context manager for measuring single-request latency.

        Usage::

            with tracker.track():
                result = model.predict(X)
        """
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            self._latencies_ms.append(elapsed_ms)
            self._total_requests += 1

    def record_latency(self, latency_ms: float) -> None:
        """Записать задержку напрямую (без context manager).

        Record latency directly (without context manager).
        Полезно при измерении батчевого инференса / Useful for batch inference.
        """
        self._latencies_ms.append(float(latency_ms))
        self._total_requests += 1

    def get_stats(self) -> LatencyStats:
        """Вернуть статистику задержки для текущего окна наблюдений.

        Return latency stats for the current observation window.

        Returns:
            LatencyStats с перцентилями p50/p95/p99 и throughput.
        """
        if not self._latencies_ms:
            return LatencyStats(
                n_requests=0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                mean_ms=0.0,
                max_ms=0.0,
                throughput_rps=0.0,
            )

        arr = np.array(list(self._latencies_ms))
        elapsed_window = time.monotonic() - self._window_start_time
        throughput = len(arr) / max(elapsed_window, 1e-9)

        return LatencyStats(
            n_requests=len(arr),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(arr.mean()),
            max_ms=float(arr.max()),
            throughput_rps=round(throughput, 2),
        )

    def reset(self) -> None:
        """Сбросить окно наблюдений / Reset observation window."""
        self._latencies_ms.clear()
        self._window_start_time = time.monotonic()

    @property
    def total_requests(self) -> int:
        """Всего запросов с момента создания трекера / Total requests since creation."""
        return self._total_requests


def estimate_monthly_cost(
    rps: float,
    instance_type: str | None = None,
    n_instances: int | None = None,
) -> CostEstimate:
    """Оценить месячную стоимость обслуживания модели в облаке.

    Estimate monthly cloud cost for model serving at given RPS.

    Аргументы:
        rps: Запросов в секунду (целевой throughput).
        instance_type: Тип инстанса (из _CLOUD_PRICING). None → автовыбор.
        n_instances: Количество инстансов. None → минимально необходимое.

    Returns:
        CostEstimate с рекомендациями и стоимостью.
    """
    if rps <= 0:
        raise ValueError(f"rps must be positive, got {rps}")

    # Автовыбор типа инстанса по RPS
    if instance_type is None:
        instance_type = "aws_c6i_large"
        for threshold, itype in _INSTANCE_SIZING:
            if rps <= threshold:
                instance_type = itype
                break

    if instance_type not in _CLOUD_PRICING:
        available = list(_CLOUD_PRICING)
        raise ValueError(f"Unknown instance type: {instance_type}. Available: {available}")

    price_per_hour = _CLOUD_PRICING[instance_type]

    # Ёмкость инстанса: ~50 RPS для c6i.large (sklearn на 2 vCPU)
    # Instance capacity: ~50 RPS for c6i.large (sklearn on 2 vCPU)
    capacity_per_instance = 50.0 if "large" in instance_type else 100.0

    if n_instances is None:
        n_instances = max(1, int(np.ceil(rps / capacity_per_instance)))

    total_price_per_hour = price_per_hour * n_instances
    cost_per_month = total_price_per_hour * 720  # 30 дней × 24 часа
    cost_per_million_req = (total_price_per_hour / (rps * 3600)) * 1_000_000

    notes_parts = [
        f"{n_instances}× {instance_type} @ ${price_per_hour:.3f}/hr each.",
        f"Capacity: ~{capacity_per_instance * n_instances:.0f} RPS total.",
    ]
    if rps > capacity_per_instance * n_instances * 0.8:
        notes_parts.append("Consider horizontal scaling — load > 80% capacity.")

    return CostEstimate(
        rps=rps,
        instance_type=instance_type,
        cost_per_hour_usd=round(total_price_per_hour, 4),
        cost_per_month_usd=round(cost_per_month, 2),
        cost_per_million_requests_usd=round(cost_per_million_req, 4),
        n_instances_recommended=n_instances,
        notes=" ".join(notes_parts),
    )


def optimize_batch_size(
    latencies_by_batch: dict[int, float],
    sla_p95_ms: float = 200.0,
) -> BatchOptimizationResult:
    """Найти оптимальный размер батча из профиля throughput/latency.

    Find optimal batch size from a throughput/latency profile.

    Оптимум балансирует throughput (больше = лучше) и latency SLA (меньше = лучше).
    Optimum balances throughput (higher = better) vs latency SLA (lower = better).

    Args:
        latencies_by_batch: {batch_size → median_latency_ms} — профиль измерений.
        sla_p95_ms: SLA для P95 задержки. Батчи с latency > SLA исключаются.

    Returns:
        BatchOptimizationResult с оптимальным batch_size и рекомендациями.

    Raises:
        ValueError: Если профиль пустой или нет батчей удовлетворяющих SLA.
    """
    if not latencies_by_batch:
        raise ValueError("latencies_by_batch cannot be empty")

    feasible = {bs: lat for bs, lat in latencies_by_batch.items() if lat <= sla_p95_ms}
    recommendations: list[str] = []

    if not feasible:
        min_lat_bs = min(latencies_by_batch, key=lambda bs: latencies_by_batch[bs])
        recommendations.append(
            f"No batch size meets SLA ({sla_p95_ms}ms). "
            f"Using batch_size={min_lat_bs} with lowest latency."
        )
        feasible = {min_lat_bs: latencies_by_batch[min_lat_bs]}

    # Throughput ∝ batch_size / latency → максимизируем
    def throughput(bs: int) -> float:
        return bs / latencies_by_batch[bs]

    max_throughput = max(throughput(bs) for bs in feasible)
    optimal_bs = max(feasible, key=throughput)
    optimal_lat = feasible[optimal_bs]

    # Нормализованная эффективность: throughput vs SLA overhead
    efficiency = min(1.0, throughput(optimal_bs) / max(max_throughput, 1e-9))
    if optimal_lat > sla_p95_ms * 0.8:
        recommendations.append(
            f"Latency {optimal_lat:.1f}ms is close to SLA ({sla_p95_ms}ms). "
            "Consider reducing batch_size for safety margin."
        )

    if optimal_bs == max(feasible):
        recommendations.append(
            "Largest feasible batch is optimal — consider profiling larger batches."
        )

    return BatchOptimizationResult(
        optimal_batch_size=optimal_bs,
        throughput_rps=round(throughput(optimal_bs) * 1000, 2),
        latency_p95_ms=round(optimal_lat, 2),
        efficiency_score=round(efficiency, 3),
        recommendations=recommendations,
    )
