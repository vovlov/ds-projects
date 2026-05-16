"""T-Learner Causal Uplift Modeling for customer retention campaigns.

Causal Uplift Modeling for targeted customer retention.

Conventional churn models predict P(churn | X) — but targeting everyone
with high churn probability wastes budget on "Sure Things" (customers who
would have stayed anyway) and "Sleeping Dogs" (customers who churn MORE
when contacted). Uplift modeling estimates the CAUSAL effect of an intervention:

    CATE(X) = E[Y(1) - Y(0) | X]
             = P(churn | treatment, X) - P(churn | control, X)

Customers with CATE < 0 are "Persuadables" — the discount REDUCES their
churn probability. These are the only customers worth targeting.

Источник: Radcliffe & Surry 1999, Gutierrez & Gérardy 2016 (CausalML),
         Large-Scale Meta-Learner Comparison (arxiv 2604.06123), Criteo Uplift v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class UpliftSegment(Enum):
    """Persuasion Matrix quadrants (Radcliffe & Surry 1999).

    Матрица убеждения: 4 сегмента клиентов.
    Таргетировать нужно только PERSUADABLE клиентов.
    """

    PERSUADABLE = "persuadable"
    SURE_THING = "sure_thing"
    LOST_CAUSE = "lost_cause"
    SLEEPING_DOG = "sleeping_dog"


@dataclass
class UpliftConfig:
    """Конфигурация T-Learner модели.

    Configuration for the T-Learner uplift model.
    """

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    # Порог CATE: ниже этого значения — persuadable
    # CATE threshold: below this value — customer is persuadable
    persuadable_threshold: float = -0.05
    # Клиент "уходит" если P(churn) > 0.5 в обеих ветках
    # Customer "churns" if P(churn) > 0.5 in both branches
    churn_threshold: float = 0.5
    random_state: int = 42


@dataclass
class UpliftPrediction:
    """Результат предсказания для одного клиента.

    Per-customer uplift prediction result.
    """

    cate: float
    p_churn_treatment: float
    p_churn_control: float
    segment: UpliftSegment


@dataclass
class UpliftResult:
    """Результат batch-предсказания + бизнес-метрики.

    Batch uplift prediction result + business metrics.
    """

    predictions: list[UpliftPrediction]
    n_persuadable: int
    n_sure_thing: int
    n_lost_cause: int
    n_sleeping_dog: int
    avg_cate: float
    targeting_uplift: float


@dataclass
class QiniResult:
    """Qini-коэффициент для оценки качества uplift-модели.

    Qini coefficient for uplift model evaluation.
    Измеряет площадь между кривой Qini и случайным таргетингом.
    Measures area between Qini curve and random targeting baseline.
    Источник: Radcliffe 2007 «Using control groups to target on predicted lift».
    """

    qini_coefficient: float
    auuc: float
    random_auuc: float
    n_treated: int
    n_control: int


@dataclass
class _SklearnModel:
    """Обёртка над sklearn GradientBoostingClassifier для lazy import.

    Wrapper around sklearn GradientBoostingClassifier for lazy import.
    Isolates sklearn dependency so is_available() still works without it.
    """

    model: Any = field(default=None)


def is_available() -> bool:
    """Проверить наличие sklearn для T-Learner.

    Check if sklearn is available for T-Learner training.
    """
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def _make_gbm(config: UpliftConfig) -> Any:
    """Создать GradientBoostingClassifier с заданной конфигурацией.

    Create a GradientBoostingClassifier with the given configuration.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
    )


class TLearnerUplift:
    """T-Learner Uplift Model — два отдельных GBM для treatment и control.

    T-Learner fits separate models for treated (T=1) and control (T=0) groups.
    CATE = μ₁(X) - μ₀(X), where μₜ(X) = E[Y | T=t, X].

    Для churn-задачи: Y=1 означает отток, T=1 означает получил скидку.
    CATE < 0 → скидка снижает вероятность оттока → клиент «Persuadable».

    For churn: Y=1 means churned, T=1 means received discount.
    CATE < 0 → discount reduces churn probability → customer is "Persuadable".

    Источник: Künzel et al. 2019 «Metalearners for Estimating HCTE» (PNAS).
    """

    def __init__(self, config: UpliftConfig | None = None) -> None:
        self._config = config or UpliftConfig()
        self._treatment_model: Any = None
        self._control_model: Any = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> "TLearnerUplift":  # noqa: UP037
        """Обучить T-Learner на исторических данных с известным treatment.

        Fit T-Learner on historical data with known treatment assignment.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary outcome (1=churn, 0=retained)
            treatment: Binary treatment indicator (1=received offer, 0=control)

        Returns:
            self (для chaining)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        treatment = np.asarray(treatment, dtype=int)

        mask_t = treatment == 1
        mask_c = treatment == 0

        if mask_t.sum() < 10 or mask_c.sum() < 10:
            raise ValueError(
                f"Need ≥10 samples per group: treatment={mask_t.sum()}, control={mask_c.sum()}"
            )

        self._treatment_model = _make_gbm(self._config)
        self._control_model = _make_gbm(self._config)

        self._treatment_model.fit(X[mask_t], y[mask_t])
        self._control_model.fit(X[mask_c], y[mask_c])

        self._is_fitted = True
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Предсказать CATE (Conditional Average Treatment Effect).

        Predict CATE for each customer.

        Returns:
            CATE array (n_samples,): negative = discount reduces churn
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_cate().")

        X = np.asarray(X, dtype=float)
        p_t = self._treatment_model.predict_proba(X)[:, 1]
        p_c = self._control_model.predict_proba(X)[:, 1]
        return p_t - p_c  # CATE = μ₁(x) - μ₀(x)

    def predict_segment(self, X: np.ndarray) -> list[UpliftPrediction]:  # noqa: N803
        """Сегментировать клиентов по Persuasion Matrix.

        Segment customers using the Persuasion Matrix quadrants.

        Логика сегментации (Radcliffe & Surry 1999):
        - Persuadable:   treatment снижает отток (CATE < -threshold)
        - Sure Thing:    уйдёт без treatment, вернётся с ним (CATE < 0, p_c > 0.5)
          Wait — используем упрощённую схему:
          CATE < -threshold → Persuadable
          CATE >  threshold → Sleeping Dog
          p_c > 0.5 AND CATE ≈ 0 → Sure Thing
          p_c ≤ 0.5 AND CATE ≈ 0 → Lost Cause

        Simplified segmentation logic based on CATE and control churn probability.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_segment().")

        X = np.asarray(X, dtype=float)
        p_t = self._treatment_model.predict_proba(X)[:, 1]
        p_c = self._control_model.predict_proba(X)[:, 1]
        cates = p_t - p_c

        thresh = abs(self._config.persuadable_threshold)
        churn_th = self._config.churn_threshold

        results = []
        for cate, pt, pc in zip(cates, p_t, p_c, strict=True):
            if cate < -thresh:
                segment = UpliftSegment.PERSUADABLE
            elif cate > thresh:
                # treatment УВЕЛИЧИВАЕТ отток — не трогать клиента
                segment = UpliftSegment.SLEEPING_DOG
            elif pc >= churn_th:
                # Высокая вероятность оттока в контроле, treatment не помогает
                segment = UpliftSegment.LOST_CAUSE
            else:
                # Низкая вероятность оттока в обоих случаях
                segment = UpliftSegment.SURE_THING

            results.append(
                UpliftPrediction(
                    cate=float(cate),
                    p_churn_treatment=float(pt),
                    p_churn_control=float(pc),
                    segment=segment,
                )
            )
        return results

    def compute_qini(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> QiniResult:
        """Вычислить Qini-коэффициент для оценки uplift-модели.

        Compute Qini coefficient to evaluate uplift model quality.

        Qini curve: отсортировать по убыванию CATE, накопительно считать
        (conversions_treated / total_treated - conversions_control / total_control).
        Qini coefficient = AUUC_model - AUUC_random.

        Чем выше, тем лучше модель находит «Persuadables».
        Higher Qini → model better identifies "Persuadables".

        Источник: Radcliffe 2007, Gutierrez & Gérardy 2016.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        treatment = np.asarray(treatment, dtype=int)

        cate = self.predict_cate(X)
        order = np.argsort(-cate)  # По убыванию CATE

        y_sorted = y[order]
        t_sorted = treatment[order]

        n = len(y)
        n_t = treatment.sum()
        n_c = n - n_t

        # Накопительная кривая Qini
        cumulative_qini = np.zeros(n + 1)
        cum_treated = 0.0
        cum_control = 0.0

        for i in range(n):
            if t_sorted[i] == 1:
                cum_treated += y_sorted[i]
            else:
                cum_control += y_sorted[i]

            t_frac = (i + 1) / n
            if n_t > 0 and n_c > 0:
                cumulative_qini[i + 1] = cum_treated / n_t - cum_control / n_c * t_frac
            else:
                cumulative_qini[i + 1] = 0.0

        # AUUC модели (трапециевидное правило)
        # np.trapezoid заменил np.trapz в NumPy 2.0
        x_axis = np.arange(n + 1) / n
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auuc = float(_trapz(cumulative_qini, x_axis))

        # AUUC случайного таргетинга (диагональ от 0 до конечного значения)
        random_curve = np.linspace(0, cumulative_qini[-1], n + 1)
        random_auuc = float(_trapz(random_curve, x_axis))

        return QiniResult(
            qini_coefficient=auuc - random_auuc,
            auuc=auuc,
            random_auuc=random_auuc,
            n_treated=int(n_t),
            n_control=int(n_c),
        )

    def get_params(self) -> dict[str, Any]:
        """Вернуть параметры модели для логирования в MLflow.

        Return model parameters for MLflow logging.
        """
        return {
            "model_type": "T-Learner",
            "base_estimator": "GradientBoostingClassifier",
            "n_estimators": self._config.n_estimators,
            "max_depth": self._config.max_depth,
            "learning_rate": self._config.learning_rate,
            "persuadable_threshold": self._config.persuadable_threshold,
        }


def summarize_uplift(predictions: list[UpliftPrediction]) -> UpliftResult:
    """Сводная статистика по batch-предсказанию.

    Summarize batch uplift predictions into business metrics.
    """
    counts = {seg: 0 for seg in UpliftSegment}
    for p in predictions:
        counts[p.segment] += 1

    cates = [p.cate for p in predictions]
    # targeting_uplift: средний CATE среди persuadable клиентов
    persuadable_cates = [p.cate for p in predictions if p.segment == UpliftSegment.PERSUADABLE]
    targeting_uplift = float(np.mean(persuadable_cates)) if persuadable_cates else 0.0

    return UpliftResult(
        predictions=predictions,
        n_persuadable=counts[UpliftSegment.PERSUADABLE],
        n_sure_thing=counts[UpliftSegment.SURE_THING],
        n_lost_cause=counts[UpliftSegment.LOST_CAUSE],
        n_sleeping_dog=counts[UpliftSegment.SLEEPING_DOG],
        avg_cate=float(np.mean(cates)) if cates else 0.0,
        targeting_uplift=targeting_uplift,
    )
