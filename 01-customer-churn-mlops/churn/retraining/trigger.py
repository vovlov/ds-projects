"""
Автоматический триггер переобучения модели предсказания оттока.

Continuous Training (CT) — ключевой компонент MLOps Level 3.
Вместо ручного запуска обучения раз в квартал, CT отвечает на вопрос:
«Изменились ли данные настолько, что модель стала хуже?»

## Стратегия обнаружения дрейфа

Используем PSI (Population Stability Index) — отраслевой стандарт
в телекоме и банкинге. PSI измеряет, насколько распределение новых
данных отличается от обучающего:

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

Пороги (стандарт BCBS 2011):
- PSI < 0.1:  стабильно, нет дрейфа
- PSI < 0.2:  умеренный дрейф, предупреждение
- PSI >= 0.2: значительный дрейф, нужно переобучение

## Условия переобучения (OR-логика)

Переобучение запускается, если ХОТЯ БЫ одно из условий выполнено:
1. Дрейф данных: max(PSI по всем фичам) >= threshold_psi
2. Деградация качества: текущий AUC < baseline_auc - delta_auc

Источник: Evidently AI v0.5+, BCBS Basel III standards,
         Medium "Data Quality Assurance in MLOps" Mar 2026.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Пороги PSI для принятия решений (BCBS 2011 стандарт)
PSI_GREEN = 0.1  # нет дрейфа
PSI_YELLOW = 0.2  # умеренный дрейф
# PSI >= PSI_YELLOW → критический дрейф


@dataclass
class DriftReport:
    """Отчёт о дрейфе данных по всем мониторируемым признакам.

    Attributes:
        feature_psi: PSI по каждому признаку {feature: psi_value}
        max_psi: Максимальный PSI среди всех признаков
        drifted_features: Список признаков с PSI >= threshold
        drift_detected: True если обнаружен критический дрейф
        baseline_size: Размер эталонной выборки (обучающая)
        current_size: Размер текущей выборки (продакшн)
    """

    feature_psi: dict[str, float] = field(default_factory=dict)
    max_psi: float = 0.0
    drifted_features: list[str] = field(default_factory=list)
    drift_detected: bool = False
    baseline_size: int = 0
    current_size: int = 0


@dataclass
class RetrainingResult:
    """Результат оценки необходимости переобучения.

    Attributes:
        should_retrain: Нужно ли запускать переобучение
        reason: Человекочитаемая причина решения
        drift_report: Полный отчёт о дрейфе данных
        baseline_auc: AUC на обучающей выборке (эталон)
        current_auc: Текущий AUC на новых данных (None если нет разметки)
        perf_degraded: True если AUC упал ниже порога
        details: Дополнительные метаданные для MLflow-тега
    """

    should_retrain: bool
    reason: str
    drift_report: DriftReport
    baseline_auc: float
    current_auc: float | None = None
    perf_degraded: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Вычислить PSI между эталонным и текущим распределениями.

    Биннинг по квантилям эталонной выборки — так каждый бин получает
    ~равное число точек в baseline, что даёт стабильные оценки PSI
    даже при малом n_bins.

    Args:
        baseline: Массив значений из обучающей выборки.
        current: Массив значений из текущей (продакшн) выборки.
        n_bins: Количество бинов (рекомендуется 10 для PSI).
        eps: Защита от log(0) при нулевых долях.

    Returns:
        Значение PSI ∈ [0, +∞). Интерпретация: < 0.1 ок, >= 0.2 дрейф.
    """
    # Квантильные границы по baseline (исключаем дубликаты)
    percentiles = np.linspace(0, 100, n_bins + 1)
    breakpoints = np.unique(np.percentile(baseline, percentiles))

    if len(breakpoints) < 2:
        # Все значения одинаковые — дрейф не определить
        return 0.0

    # Частоты попадания в каждый бин
    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    baseline_freq = baseline_counts / (len(baseline) + eps)
    current_freq = current_counts / (len(current) + eps)

    # PSI = Σ (A - E) × ln(A / E), где A=actual (current), E=expected (baseline)
    # eps защищает от log(0) при нулевых бинах
    psi = np.sum(
        (current_freq - baseline_freq) * np.log((current_freq + eps) / (baseline_freq + eps))
    )
    return float(psi)


class RetrainingTrigger:
    """Анализирует дрейф данных и деградацию качества, решает о переобучении.

    Логика:
    1. Сравниваем числовые признаки (PSI) с обучающей выборкой
    2. Если есть разметка — сравниваем AUC с baseline
    3. Решение: retrain если PSI >= threshold_psi ИЛИ AUC деградировал

    Пример использования:
        trigger = RetrainingTrigger(baseline_df, baseline_auc=0.84)
        result = trigger.evaluate(current_df)
        if result.should_retrain:
            launch_training_pipeline()
    """

    def __init__(
        self,
        baseline_features: dict[str, np.ndarray],
        baseline_auc: float = 0.80,
        threshold_psi: float = PSI_YELLOW,
        delta_auc: float = 0.05,
        features_to_monitor: list[str] | None = None,
    ) -> None:
        """Инициализировать триггер.

        Args:
            baseline_features: Словарь {feature_name: array} из обучающей выборки.
            baseline_auc: AUC модели на обучающей выборке (эталон).
            threshold_psi: Порог PSI для срабатывания (дефолт: 0.2 по BCBS).
            delta_auc: Допустимое падение AUC (дефолт: 0.05 = 5 пп).
            features_to_monitor: Список признаков для мониторинга.
                                 None = все из baseline_features.
        """
        self.baseline_features = baseline_features
        self.baseline_auc = baseline_auc
        self.threshold_psi = threshold_psi
        self.delta_auc = delta_auc
        self.features_to_monitor = features_to_monitor or list(baseline_features.keys())

    def check_drift(self, current_features: dict[str, np.ndarray]) -> DriftReport:
        """Вычислить PSI для всех мониторируемых признаков.

        Args:
            current_features: Словарь {feature_name: array} из продакшн-данных.

        Returns:
            DriftReport с PSI по каждому признаку и итоговым флагом drift_detected.
        """
        feature_psi: dict[str, float] = {}

        for feat in self.features_to_monitor:
            if feat not in self.baseline_features or feat not in current_features:
                logger.warning("Feature '%s' missing from one of the datasets, skipping", feat)
                continue

            baseline_arr = np.asarray(self.baseline_features[feat], dtype=float)
            current_arr = np.asarray(current_features[feat], dtype=float)

            # Пропускаем признаки с нулевой дисперсией — PSI не информативен
            if baseline_arr.std() < 1e-9:
                logger.debug("Feature '%s' has zero variance, PSI=0.0", feat)
                feature_psi[feat] = 0.0
                continue

            psi = compute_psi(baseline_arr, current_arr)
            feature_psi[feat] = round(psi, 6)

        drifted = [f for f, p in feature_psi.items() if p >= self.threshold_psi]
        max_psi = max(feature_psi.values(), default=0.0)

        # Определяем цвет статуса для логирования
        if max_psi < PSI_GREEN:
            status = "GREEN (stable)"
        elif max_psi < PSI_YELLOW:
            status = "YELLOW (moderate drift)"
        else:
            status = "RED (significant drift)"

        logger.info(
            "Drift check: max_psi=%.4f [%s], drifted_features=%s",
            max_psi,
            status,
            drifted,
        )

        return DriftReport(
            feature_psi=feature_psi,
            max_psi=max_psi,
            drifted_features=drifted,
            drift_detected=len(drifted) > 0,
            baseline_size=len(next(iter(self.baseline_features.values()), [])),
            current_size=len(next(iter(current_features.values()), [])),
        )

    def check_performance(self, current_auc: float) -> bool:
        """Проверить деградацию AUC относительно baseline.

        Args:
            current_auc: Текущий AUC на свежей разметке (если есть).

        Returns:
            True если AUC упал на delta_auc или больше.
        """
        drop = self.baseline_auc - current_auc
        degraded = drop >= self.delta_auc
        logger.info(
            "Performance check: baseline_auc=%.4f, current_auc=%.4f, drop=%.4f, degraded=%s",
            self.baseline_auc,
            current_auc,
            drop,
            degraded,
        )
        return degraded

    def evaluate(
        self,
        current_features: dict[str, np.ndarray],
        current_auc: float | None = None,
    ) -> RetrainingResult:
        """Полная оценка: дрейф данных + (опционально) деградация AUC.

        Переобучение запускается если ХОТЯ БЫ одно условие выполнено:
        - data_drift: max(PSI) >= threshold_psi
        - perf_degraded: current_auc < baseline_auc - delta_auc

        Если current_auc не передан (нет разметки в реальном времени),
        решение основывается только на дрейфе данных.

        Args:
            current_features: Данные из продакшн за мониторируемый период.
            current_auc: AUC на данных с разметкой (необязательно).

        Returns:
            RetrainingResult с итоговым решением и полным аудит-логом.
        """
        drift_report = self.check_drift(current_features)

        perf_degraded = False
        if current_auc is not None:
            perf_degraded = self.check_performance(current_auc)

        # OR-логика: любое из двух условий достаточно для переобучения
        should_retrain = drift_report.drift_detected or perf_degraded

        # Формируем человекочитаемую причину для аудит-лога
        reasons = []
        if drift_report.drift_detected:
            reasons.append(
                f"data drift: PSI={drift_report.max_psi:.4f} >= {self.threshold_psi} "
                f"in features={drift_report.drifted_features}"
            )
        if perf_degraded:
            drop = self.baseline_auc - (current_auc or 0.0)
            reasons.append(f"performance degradation: AUC drop={drop:.4f} >= {self.delta_auc}")
        if not reasons:
            reasons.append(
                f"stable: max_psi={drift_report.max_psi:.4f} < {self.threshold_psi}, "
                f"no AUC degradation"
            )

        reason = "; ".join(reasons)

        result = RetrainingResult(
            should_retrain=should_retrain,
            reason=reason,
            drift_report=drift_report,
            baseline_auc=self.baseline_auc,
            current_auc=current_auc,
            perf_degraded=perf_degraded,
            details={
                "threshold_psi": self.threshold_psi,
                "delta_auc": self.delta_auc,
                "monitored_features": self.features_to_monitor,
                "feature_psi": drift_report.feature_psi,
            },
        )

        self._log_to_mlflow(result)
        return result

    def _log_to_mlflow(self, result: RetrainingResult) -> None:
        """Логировать решение в MLflow для аудит-трейла.

        Аудит-трейл — требование EU AI Act и внутренних ML governance политик:
        нужно фиксировать КТО и КОГДА принял решение о переобучении и ПОЧЕМУ.

        Graceful degradation: если MLflow недоступен — пишем только в logger.
        """
        try:
            import mlflow

            with mlflow.start_run(run_name="retraining-check", nested=True):
                mlflow.log_metric("max_psi", result.drift_report.max_psi)
                mlflow.log_metric("drift_detected", int(result.drift_report.drift_detected))
                mlflow.log_metric("should_retrain", int(result.should_retrain))
                mlflow.log_metric("baseline_auc", result.baseline_auc)

                if result.current_auc is not None:
                    mlflow.log_metric("current_auc", result.current_auc)

                for feat, psi_val in result.drift_report.feature_psi.items():
                    mlflow.log_metric(f"psi_{feat}", psi_val)

                mlflow.set_tag("retraining.reason", result.reason[:500])
                decision = "RETRAIN" if result.should_retrain else "SKIP"
                mlflow.set_tag("retraining.decision", decision)

        except Exception as exc:
            # MLflow недоступен в CI или при file-store — не критично
            logger.debug("MLflow logging skipped: %s", exc)

        logger.info(
            "Retraining decision: %s | Reason: %s",
            "RETRAIN" if result.should_retrain else "SKIP",
            result.reason,
        )
