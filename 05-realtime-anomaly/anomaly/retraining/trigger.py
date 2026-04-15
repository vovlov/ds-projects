"""
Автоматический триггер переобучения детектора аномалий.

Continuous Training (CT) для real-time сервиса: когда статистическое
распределение метрик (cpu, latency, requests) меняется настолько, что
порог детекции Z-score устаревает, нужно переобучить/перекалибровать
детектор на свежих данных.

## Стратегия обнаружения дрейфа

Используем MMD (Maximum Mean Discrepancy) — многомерный непараметрический
тест. В отличие от PSI (работает на 1D попеременно), MMD анализирует
совместное распределение всех метрик: ловит изменения корреляций,
которые PSI пропустит.

Условие переобучения:
    MMD(reference, current) > bootstrap_threshold(alpha=0.05)

## Аудит-трейл (EU AI Act compliance)

Каждое решение логируется в MLflow с полными метаданными:
- audit_id: UUID для трассировки
- timestamp: ISO 8601 UTC
- mmd_statistic, threshold, is_drift
- reason: человекочитаемое объяснение
- features_monitored: список признаков

Источник: Gretton et al. 2012 "A Kernel Two-Sample Test", JMLR.
         EU AI Act Article 9 (risk management system requirements).
         MLOps Level 3: automated retraining on drift signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..drift.mmd import DriftResult, MMDDriftDetector

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRetrainingResult:
    """Результат оценки необходимости переобучения детектора аномалий.

    Attributes:
        should_retrain: Нужно ли переобучить/перекалибровать детектор.
        reason: Человекочитаемая причина решения.
        drift_result: Полный результат MMD-теста.
        triggered_by: Что вызвало решение ("mmd_drift" | "manual" | "none").
        details: Дополнительные метаданные для MLflow-тега.
    """

    should_retrain: bool
    reason: str
    drift_result: DriftResult
    triggered_by: str
    details: dict[str, Any] = field(default_factory=dict)


class AnomalyRetrainingTrigger:
    """Анализирует MMD-дрейф метрик и решает о переобучении детектора.

    Логика:
    1. При инициализации: сохраняем reference window (нормальные метрики)
    2. evaluate(current): считаем MMD(reference, current)
    3. Если MMD > bootstrap-порог → trigger retraining
    4. Логируем решение в MLflow (graceful degradation без MLflow)

    Пример:
        trigger = AnomalyRetrainingTrigger(reference_metrics)
        result = trigger.evaluate(current_metrics)
        if result.should_retrain:
            recalibrate_detector(current_metrics)
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        features: list[str] | None = None,
        alpha: float = 0.05,
        n_bootstrap: int = 200,
    ) -> None:
        """Инициализировать триггер с эталонными данными.

        Args:
            reference_data: Эталонная матрица метрик (n, d).
                           Обычно: нормальные данные за период обучения.
                           Для SRE: метрики из «золотого периода» без инцидентов.
            features: Имена признаков для аудит-лога (e.g. ["cpu", "latency", "requests"]).
            alpha: Уровень значимости для bootstrap-порога (0.05 = 95% confidence).
            n_bootstrap: Число итераций bootstrap (больше = точнее, но медленнее).
        """
        self.features = features or ["f0", "f1", "f2"]
        self._detector = MMDDriftDetector(
            reference_data=reference_data,
            features=self.features,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
        )
        logger.info(
            "AnomalyRetrainingTrigger ready: features=%s, alpha=%.2f, threshold=%.6f",
            self.features,
            alpha,
            self._detector.threshold,
        )

    def evaluate(self, current_data: np.ndarray) -> AnomalyRetrainingResult:
        """Полная оценка: MMD-тест + решение о переобучении.

        Args:
            current_data: Текущие метрики (m, d) — производственные данные
                         за мониторируемый период.

        Returns:
            AnomalyRetrainingResult с решением и полным аудит-логом.
        """
        drift_result = self._detector.detect(current_data)

        should_retrain = drift_result.is_drift
        triggered_by = "mmd_drift" if should_retrain else "none"

        if should_retrain:
            reason = f"MMD drift triggered retraining: {drift_result.reason}"
        else:
            reason = f"No drift detected, retraining skipped: {drift_result.reason}"

        result = AnomalyRetrainingResult(
            should_retrain=should_retrain,
            reason=reason,
            drift_result=drift_result,
            triggered_by=triggered_by,
            details={
                "mmd_statistic": drift_result.mmd_statistic,
                "threshold": drift_result.threshold,
                "gamma": drift_result.gamma,
                "p_value": drift_result.p_value,
                "features_monitored": self.features,
                "audit_id": drift_result.audit_id,
                "timestamp": drift_result.timestamp,
            },
        )

        self._log_to_mlflow(result)
        return result

    def _log_to_mlflow(self, result: AnomalyRetrainingResult) -> None:
        """Логировать решение о переобучении в MLflow (аудит-трейл).

        EU AI Act Article 9 требует документировать все автоматические решения
        AI-системы, включая решения о переобучении модели. Каждый запуск
        сохраняет audit_id → сквозная трассировка от drift-события до retrain-run.

        Graceful degradation: если MLflow недоступен (CI, локальная разработка)
        — пишем только в лог, не прерываем основной поток.
        """
        try:
            import mlflow

            with mlflow.start_run(run_name="anomaly-retraining-check", nested=True):
                mlflow.log_metric("mmd_statistic", result.drift_result.mmd_statistic)
                mlflow.log_metric("mmd_threshold", result.drift_result.threshold)
                mlflow.log_metric("mmd_p_value", result.drift_result.p_value)
                mlflow.log_metric("is_drift", int(result.drift_result.is_drift))
                mlflow.log_metric("should_retrain", int(result.should_retrain))

                mlflow.set_tag("audit.id", result.drift_result.audit_id)
                mlflow.set_tag("audit.timestamp", result.drift_result.timestamp)
                mlflow.set_tag("retraining.triggered_by", result.triggered_by)
                mlflow.set_tag("retraining.reason", result.reason[:500])
                mlflow.set_tag(
                    "retraining.decision",
                    "RETRAIN" if result.should_retrain else "SKIP",
                )
                mlflow.set_tag(
                    "drift.features",
                    ",".join(result.drift_result.features),
                )

        except Exception as exc:
            # MLflow недоступен — не критично для основной логики
            logger.debug("MLflow logging skipped: %s", exc)

        decision = "RETRAIN" if result.should_retrain else "SKIP"
        logger.info(
            "Retraining decision: %s | audit_id=%s | %s",
            decision,
            result.drift_result.audit_id,
            result.reason,
        )
