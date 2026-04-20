"""
A/B тестирование моделей предсказания оттока клиентов.

Проблема: как безопасно раскатить новую версию модели?
Случайный traffic split создаёт noise: один клиент получает
разные предсказания при разных обращениях — это артефакт,
не сигнал от модели.

Решение: детерминированный роутинг по хешу customer_id.
Один клиент → всегда один вариант → нет switching noise.

Статистические тесты:
- Z-тест для долей: сравниваем high-risk rate (конверсионная метрика)
- Welch's t-test: сравниваем средние churn probability
- p < 0.05 при n >= min_samples_per_variant → statistically significant

Источники:
  AWS Blog "Dynamic A/B testing for ML models" 2025,
  MLOps Community "The What, Why, and How of A/B Testing in ML",
  Marvelous MLOps "Traffic Splits Aren't True A/B Testing" 2025.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Graceful degradation: scipy для точных тестов, numpy-fallback без него
try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def is_scipy_available() -> bool:
    """Check whether scipy is installed for statistical tests."""
    return SCIPY_AVAILABLE


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class VariantConfig:
    """Конфигурация варианта эксперимента / Experiment variant configuration.

    Attributes:
        name: Идентификатор варианта ("control", "treatment").
        traffic_weight: Доля трафика [0, 1]. Сумма по всем вариантам = 1.0.
        model_version: Версия модели, обслуживающей вариант.
        description: Человекочитаемое описание изменений в модели.
    """

    name: str
    traffic_weight: float
    model_version: str = "v1"
    description: str = ""


@dataclass
class PredictionRecord:
    """Запись предсказания для одного клиента / Single prediction log entry.

    Attributes:
        customer_id: Идентификатор клиента (используется для детерминированного роутинга).
        variant: Вариант, получивший этого клиента.
        churn_probability: Предсказанная вероятность оттока [0, 1].
        risk_level: Бизнес-категория риска ("low", "medium", "high").
        timestamp: ISO-8601 UTC время предсказания.
        actual_churn: Фактический отток (заполняется когда ground truth доступен).
    """

    customer_id: str
    variant: str
    churn_probability: float
    risk_level: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    actual_churn: bool | None = None


@dataclass
class VariantStats:
    """Агрегированная статистика варианта / Aggregated variant statistics.

    Attributes:
        name: Имя варианта.
        n_predictions: Количество предсказаний.
        mean_churn_prob: Средняя вероятность оттока.
        high_risk_rate: Доля предсказаний с risk_level="high".
        n_outcomes: Количество записей с известным ground truth.
        actual_churn_rate: Фактический процент оттока (если ground truth доступен).
    """

    name: str
    n_predictions: int
    mean_churn_prob: float
    high_risk_rate: float
    n_outcomes: int
    actual_churn_rate: float | None


@dataclass
class ExperimentResult:
    """Результат статистического анализа эксперимента / Statistical experiment result.

    Attributes:
        status: "not_enough_data" | "running" | "significant" | "inconclusive"
        winner: Победивший вариант или None если разница не значима.
        p_value_prob: p-value Welch's t-test для churn probability.
        p_value_rate: p-value z-test для high-risk rate.
        control_stats: Статистика контрольного варианта.
        treatment_stats: Статистика варианта-претендента.
        relative_effect: Относительное изменение high-risk rate (%).
        recommendation: Рекомендация для ML-инженера.
        min_samples_per_variant: Минимальный размер выборки для мощности теста 0.8.
        scipy_available: Флаг доступности scipy.
    """

    status: str
    winner: str | None
    p_value_prob: float | None
    p_value_rate: float | None
    control_stats: VariantStats
    treatment_stats: VariantStats
    relative_effect: float | None
    recommendation: str
    min_samples_per_variant: int
    scipy_available: bool


# ---------------------------------------------------------------------------
# Core experiment engine
# ---------------------------------------------------------------------------

# Минимальный размер выборки при: alpha=0.05, power=0.8, MDE=5% по доле оттока ~26%
# Рассчитано по формуле Cohen's h для пропорций.
_DEFAULT_MIN_SAMPLES = 385


class ABExperiment:
    """Движок A/B эксперимента с детерминированным роутингом.

    Детерминированный роутинг: customer_id → MD5 → [0,1) → вариант.
    Гарантирует стабильное назначение: один клиент всегда в одном варианте.
    Это устраняет switching noise — ключевую проблему случайного трафик-сплита
    для ML-моделей в отличие от классических A/B-тестов для веб-конверсий.

    Usage:
        experiment = ABExperiment(
            variants=[
                VariantConfig("control", 0.5, "v1", "current production model"),
                VariantConfig("treatment", 0.5, "v2", "retrained with Q4 data"),
            ]
        )
        variant = experiment.route("cust_12345")
        experiment.record_prediction("cust_12345", variant, 0.73, "high")
        result = experiment.compute_results()
    """

    def __init__(
        self,
        variants: list[VariantConfig] | None = None,
        min_samples_per_variant: int = _DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._min_samples = min_samples_per_variant
        self._variants: list[VariantConfig] = variants or [
            VariantConfig("control", 0.5, "v1", "current production model"),
            VariantConfig("treatment", 0.5, "v2", "challenger model"),
        ]
        self._validate_weights()
        self._predictions: list[PredictionRecord] = []

    def _validate_weights(self) -> None:
        """Веса вариантов должны суммироваться в 1.0 с допуском 1e-6."""
        total = sum(v.traffic_weight for v in self._variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Variant traffic weights must sum to 1.0, got {total:.6f}")

    def route(self, customer_id: str) -> str:
        """Детерминированно назначить клиента в вариант по хешу ID.

        Deterministically assign customer to a variant via MD5 hash.
        Один и тот же customer_id всегда получает один и тот же вариант —
        нет шума от случайного переключения между запросами.

        Args:
            customer_id: Уникальный идентификатор клиента.

        Returns:
            Имя назначенного варианта.
        """
        digest = hashlib.md5(customer_id.encode(), usedforsecurity=False).digest()
        # Берём первые 4 байта → uint32 → [0, 1)
        bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        cumulative = 0.0
        for variant in self._variants:
            cumulative += variant.traffic_weight
            if bucket < cumulative:
                return variant.name
        # Floating point edge: последний вариант ловит остаток
        return self._variants[-1].name

    def record_prediction(
        self,
        customer_id: str,
        variant: str,
        churn_probability: float,
        risk_level: str,
    ) -> PredictionRecord:
        """Зафиксировать предсказание для последующего статистического анализа.

        Args:
            customer_id: ID клиента.
            variant: Вариант, сделавший предсказание.
            churn_probability: Предсказанная вероятность [0, 1].
            risk_level: "low" | "medium" | "high".

        Returns:
            Созданная запись предсказания.
        """
        record = PredictionRecord(
            customer_id=customer_id,
            variant=variant,
            churn_probability=churn_probability,
            risk_level=risk_level,
        )
        self._predictions.append(record)
        return record

    def record_outcome(self, customer_id: str, actual_churn: bool) -> bool:
        """Записать фактический исход для клиента (ground truth).

        Record actual churn outcome — used for outcome-based analysis
        once ground truth labels become available (days/weeks after prediction).

        Args:
            customer_id: ID клиента.
            actual_churn: Фактически ли клиент ушёл.

        Returns:
            True если запись найдена и обновлена, False если нет.
        """
        for record in reversed(self._predictions):
            if record.customer_id == customer_id and record.actual_churn is None:
                record.actual_churn = actual_churn
                return True
        return False

    def get_variant_predictions(self, variant_name: str) -> list[PredictionRecord]:
        """Вернуть все предсказания для заданного варианта."""
        return [p for p in self._predictions if p.variant == variant_name]

    def _compute_variant_stats(self, variant_name: str) -> VariantStats:
        """Агрегировать статистику для одного варианта."""
        preds = self.get_variant_predictions(variant_name)
        n = len(preds)
        if n == 0:
            return VariantStats(
                name=variant_name,
                n_predictions=0,
                mean_churn_prob=0.0,
                high_risk_rate=0.0,
                n_outcomes=0,
                actual_churn_rate=None,
            )

        mean_prob = sum(p.churn_probability for p in preds) / n
        high_risk_rate = sum(1 for p in preds if p.risk_level == "high") / n

        outcomes = [p for p in preds if p.actual_churn is not None]
        actual_churn_rate = (
            sum(1 for p in outcomes if p.actual_churn) / len(outcomes) if outcomes else None
        )

        return VariantStats(
            name=variant_name,
            n_predictions=n,
            mean_churn_prob=round(mean_prob, 4),
            high_risk_rate=round(high_risk_rate, 4),
            n_outcomes=len(outcomes),
            actual_churn_rate=(
                round(actual_churn_rate, 4) if actual_churn_rate is not None else None
            ),
        )

    def _z_test_proportions(self, n1: int, p1: float, n2: int, p2: float) -> float | None:
        """Z-тест для разницы двух пропорций (high-risk rate).

        Двусторонний тест. Если scipy доступен — используем его,
        иначе вычисляем вручную через нормальное приближение.

        Returns:
            p-value или None если размер выборки слишком мал.
        """
        if n1 < 10 or n2 < 10:
            return None

        if SCIPY_AVAILABLE:
            import numpy as np
            from scipy.stats import chi2_contingency

            # Строим contingency table для chi2 — эквивалент z-test для 2x2
            high1 = round(p1 * n1)
            high2 = round(p2 * n2)
            table = [[high1, n1 - high1], [high2, n2 - high2]]
            if any(cell < 5 for row in table for cell in row):
                return None
            _, p_val, _, _ = chi2_contingency(np.array(table), correction=False)
            return float(p_val)

        # Нормальное приближение без scipy
        if n1 == 0 or n2 == 0:
            return None
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        if p_pool <= 0 or p_pool >= 1:
            return None
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            return None
        z = (p1 - p2) / se
        # Approximation: two-tailed p-value using normal CDF approximation
        # (Abramowitz & Stegun 26.2.17)
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        # fmt: off
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))  # noqa: E501
        # fmt: on
        p_one_tail = poly * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        return float(min(2 * p_one_tail, 1.0))

    def _welch_t_test(self, probs1: list[float], probs2: list[float]) -> float | None:
        """Welch's t-test для разницы средних churn probability.

        Welch's (не Student's) — не предполагает равных дисперсий,
        что важно при сравнении разных моделей.

        Returns:
            p-value или None если scipy недоступен или выборка мала.
        """
        if not SCIPY_AVAILABLE or len(probs1) < 10 or len(probs2) < 10:
            return None
        _, p_val = scipy_stats.ttest_ind(probs1, probs2, equal_var=False)
        return float(p_val)

    def compute_results(self) -> ExperimentResult:
        """Вычислить статистические результаты эксперимента.

        Compute statistical results of the A/B experiment.

        Returns:
            ExperimentResult со статусом, p-value, победителем и рекомендацией.
        """
        if len(self._variants) < 2:
            raise ValueError("Need at least 2 variants for A/B test")

        control_name = self._variants[0].name
        treatment_name = self._variants[1].name

        ctrl_stats = self._compute_variant_stats(control_name)
        trt_stats = self._compute_variant_stats(treatment_name)

        n_ctrl = ctrl_stats.n_predictions
        n_trt = trt_stats.n_predictions

        # Недостаточно данных для анализа
        if n_ctrl < self._min_samples or n_trt < self._min_samples:
            needed = self._min_samples - min(n_ctrl, n_trt)
            return ExperimentResult(
                status="not_enough_data",
                winner=None,
                p_value_prob=None,
                p_value_rate=None,
                control_stats=ctrl_stats,
                treatment_stats=trt_stats,
                relative_effect=None,
                recommendation=(
                    f"Need {needed} more samples per variant "
                    f"(current: ctrl={n_ctrl}, trt={n_trt}, "
                    f"min={self._min_samples}). Keep collecting data."
                ),
                min_samples_per_variant=self._min_samples,
                scipy_available=SCIPY_AVAILABLE,
            )

        # Статистические тесты
        p_val_rate = self._z_test_proportions(
            n_ctrl,
            ctrl_stats.high_risk_rate,
            n_trt,
            trt_stats.high_risk_rate,
        )

        ctrl_probs = [p.churn_probability for p in self.get_variant_predictions(control_name)]
        trt_probs = [p.churn_probability for p in self.get_variant_predictions(treatment_name)]
        p_val_prob = self._welch_t_test(ctrl_probs, trt_probs)

        # Относительный эффект по primary метрике (high_risk_rate)
        relative_effect = None
        if ctrl_stats.high_risk_rate > 0:
            relative_effect = round(
                (trt_stats.high_risk_rate - ctrl_stats.high_risk_rate)
                / ctrl_stats.high_risk_rate
                * 100,
                2,
            )

        # Значимость: p < 0.05 хотя бы по одному тесту
        primary_p = p_val_rate if p_val_rate is not None else p_val_prob
        is_significant = primary_p is not None and primary_p < 0.05

        if not is_significant:
            return ExperimentResult(
                status="inconclusive",
                winner=None,
                p_value_prob=p_val_prob,
                p_value_rate=p_val_rate,
                control_stats=ctrl_stats,
                treatment_stats=trt_stats,
                relative_effect=relative_effect,
                recommendation=(
                    f"No significant difference (p={primary_p:.3f} ≥ 0.05). "
                    "Continue experiment or accept null hypothesis."
                ),
                min_samples_per_variant=self._min_samples,
                scipy_available=SCIPY_AVAILABLE,
            )

        # Победитель: вариант с меньшим high_risk_rate (меньше оттока = лучше)
        # Если ниже прогнозируемый отток → модель лучше удерживает клиентов
        winner = (
            treatment_name if trt_stats.high_risk_rate < ctrl_stats.high_risk_rate else control_name
        )
        effect_str = f"{abs(relative_effect):.1f}%" if relative_effect is not None else "?"

        return ExperimentResult(
            status="significant",
            winner=winner,
            p_value_prob=p_val_prob,
            p_value_rate=p_val_rate,
            control_stats=ctrl_stats,
            treatment_stats=trt_stats,
            relative_effect=relative_effect,
            recommendation=(
                f"Statistically significant result (p={primary_p:.4f} < 0.05). "
                f"Winner: '{winner}' with {effect_str} change in high-risk rate. "
                f"Recommend promoting '{winner}' to 100% traffic."
            ),
            min_samples_per_variant=self._min_samples,
            scipy_available=SCIPY_AVAILABLE,
        )

    def reset(self) -> None:
        """Сбросить все предсказания — начать новый эксперимент."""
        self._predictions.clear()

    @property
    def total_predictions(self) -> int:
        """Общее количество предсказаний во всех вариантах."""
        return len(self._predictions)

    @property
    def variant_names(self) -> list[str]:
        """Имена всех вариантов."""
        return [v.name for v in self._variants]

    def get_status_summary(self) -> dict[str, Any]:
        """Краткая сводка эксперимента для /ab/status endpoint."""
        return {
            "total_predictions": self.total_predictions,
            "min_samples_per_variant": self._min_samples,
            "scipy_available": SCIPY_AVAILABLE,
            "variants": [
                {
                    "name": v.name,
                    "traffic_weight": v.traffic_weight,
                    "model_version": v.model_version,
                    "n_predictions": len(self.get_variant_predictions(v.name)),
                }
                for v in self._variants
            ],
        }
