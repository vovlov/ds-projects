"""T-Learner meta-learner для uplift-моделирования retention-кампаний.

T-Learner meta-learner for uplift modeling of retention campaigns.

Идея: обучить два отдельных классификатора на treated и control группах,
затем оценить CATE(x) = μ₁(x) - μ₀(x) для каждого клиента.

Idea: fit two separate classifiers on treated vs control groups,
then estimate CATE(x) = μ₁(x) - μ₀(x) per customer.

Сегменты по значению CATE (Uplift-квадрант):
 +  Persuadables:  CATE > threshold  → таргетировать (кампания помогает)
 ~  Uncertain:     |CATE| ≤ threshold → осторожно (неясный эффект)
 −  Sleeping Dogs: CATE < -threshold  → НЕ таргетировать (кампания вредит)
 ~  Sure Things / Lost Causes: оба P низкие или оба высокие

Источники:
- Gutierrez & Gerardy 2017 (JMLR Workshop & Conf. Proc.)
- Künzel et al. 2019 "Metalearners for estimating heterogeneous treatment
  effects using machine learning" (PNAS 116:4156-4165)
- Devriendt et al. 2018 (Decision Support Systems)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class UpliftConfig:
    """Конфигурация T-Learner.
    T-Learner configuration.
    """

    max_depth: int = 5
    # min_samples_leaf защищает от переобучения на малых treatment-group-ах
    min_samples_leaf: int = 20
    random_state: int = 42
    # Порог CATE для классификации клиентов в сегменты
    # CATE threshold for customer segmentation
    target_threshold: float = 0.05
    do_not_target_threshold: float = -0.05


@dataclass
class UpliftPrediction:
    """Результат uplift-предсказания для одного клиента.

    Uplift prediction result for a single customer.
    """

    cate: float
    """Conditional Average Treatment Effect ∈ (−1, 1).
    Positive = intervention increases positive outcome (retention).
    Negative = intervention decreases positive outcome (sleeping dogs).
    """

    p_treated: float
    """P(positive outcome | treatment=1, features)"""

    p_control: float
    """P(positive outcome | treatment=0, features)"""

    segment: str
    """
    Customer uplift segment:
    - 'persuadable': CATE > threshold → target with retention offer
    - 'sleeping_dog': CATE < -threshold → avoid, intervention backfires
    - 'uncertain': |CATE| ≤ threshold → unclear effect
    """

    recommendation: str
    """Human-readable targeting recommendation."""


@dataclass
class TrainSummary:
    """Сводка по обучению T-Learner.
    T-Learner training summary.
    """

    n_total: int
    n_treated: int
    n_control: int
    treatment_rate: float
    outcome_rate_treated: float
    """Доля позитивных исходов в treated группе."""
    outcome_rate_control: float
    """Доля позитивных исходов в control группе."""
    avg_cate_train: float
    """Средний CATE на обучающих данных — проверка здравости."""
    pct_persuadable: float
    """% клиентов с CATE > threshold на обучающих данных."""
    pct_sleeping_dog: float
    """% клиентов с CATE < -threshold на обучающих данных."""
    sklearn_available: bool


class TLearner:
    """T-Learner (Two-Model Approach) для оценки CATE retention-кампаний.

    T-Learner (Two-Model Approach) for estimating CATE of retention campaigns.

    Обучает два отдельных base-learner:
    - μ₁(x): на treated (T=1) наблюдениях → P(outcome=1 | T=1, x)
    - μ₀(x): на control (T=0) наблюдениях → P(outcome=1 | T=0, x)

    Fits two separate base-learners:
    - μ₁(x): on treated (T=1) observations → P(outcome=1 | T=1, x)
    - μ₀(x): on control (T=0) observations → P(outcome=1 | T=0, x)

    CATE(x) = μ₁(x) − μ₀(x)

    В контексте churn/retention:
    - outcome=1 → клиент остался после контакта (положительный ответ)
    - outcome=0 → клиент ушёл несмотря на контакт

    In churn/retention context:
    - outcome=1 → customer stayed after contact (positive response)
    - outcome=0 → customer churned despite contact
    """

    def __init__(self, config: UpliftConfig | None = None) -> None:
        self._config = config or UpliftConfig()
        self._model_t: Any = None
        self._model_c: Any = None
        self._is_fitted: bool = False
        self._train_summary: TrainSummary | None = None
        self._feature_names: list[str] = []
        self._sklearn_available: bool = _check_sklearn()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        T: np.ndarray,  # noqa: N803
        Y: np.ndarray,  # noqa: N803
        feature_names: list[str] | None = None,
    ) -> TrainSummary:
        """Обучить два base-learner-а на treated и control группах.

        Fit two base-learners on treated and control groups.

        Args:
            X: (n_samples, n_features) признаки клиентов
            T: (n_samples,) treatment indicator {0=control, 1=treated}
            Y: (n_samples,) outcome {0=churned, 1=stayed/responded}
            feature_names: имена признаков для predict_one()

        Returns:
            TrainSummary с ключевыми статистиками по обучению.
        """
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y, dtype=int)

        mask_t = T == 1
        mask_c = T == 0
        X_t, Y_t = X[mask_t], Y[mask_t]
        X_c, Y_c = X[mask_c], Y[mask_c]

        if len(X_t) < 2:
            raise ValueError(f"Need ≥2 treated samples, got {len(X_t)}")
        if len(X_c) < 2:
            raise ValueError(f"Need ≥2 control samples, got {len(X_c)}")

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._model_t = self._make_model()
        self._model_c = self._make_model()
        self._model_t.fit(X_t, Y_t)
        self._model_c.fit(X_c, Y_c)
        self._is_fitted = True

        cate_train = self.predict_uplift(X)
        cfg = self._config
        n = len(X)

        self._train_summary = TrainSummary(
            n_total=n,
            n_treated=int(mask_t.sum()),
            n_control=int(mask_c.sum()),
            treatment_rate=float(mask_t.mean()),
            outcome_rate_treated=float(Y_t.mean()),
            outcome_rate_control=float(Y_c.mean()),
            avg_cate_train=float(cate_train.mean()),
            pct_persuadable=float((cate_train > cfg.target_threshold).mean() * 100),
            pct_sleeping_dog=float((cate_train < cfg.do_not_target_threshold).mean() * 100),
            sklearn_available=self._sklearn_available,
        )
        return self._train_summary

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """CATE(x) = P(Y=1|T=1,x) − P(Y=1|T=0,x) для массива клиентов.

        CATE(x) = P(Y=1|T=1,x) − P(Y=1|T=0,x) for an array of customers.

        Returns:
            (n_samples,) float array, значения в (−1, 1).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return self._proba_t(X) - self._proba_c(X)

    def predict_one(self, features: dict[str, float]) -> UpliftPrediction:
        """Предсказать uplift для одного клиента по словарю признаков.

        Predict uplift for a single customer given a feature dictionary.
        """
        self._check_fitted()
        x = np.array([[features.get(fn, 0.0) for fn in self._feature_names]], dtype=float)
        p_t = float(self._proba_t(x)[0])
        p_c = float(self._proba_c(x)[0])
        cate = p_t - p_c

        cfg = self._config
        if cate > cfg.target_threshold:
            segment = "persuadable"
            rec = (
                f"Target: CATE={cate:+.3f} (intervention increases retention by {cate * 100:.1f}pp)"
            )
        elif cate < cfg.do_not_target_threshold:
            segment = "sleeping_dog"
            rec = (
                f"Do NOT target: CATE={cate:+.3f} "
                f"(intervention reduces retention by {abs(cate) * 100:.1f}pp — backfire risk)"
            )
        else:
            segment = "uncertain"
            rec = f"Uncertain: CATE={cate:+.3f} (effect too small to act on confidently)"

        return UpliftPrediction(
            cate=round(cate, 4),
            p_treated=round(p_t, 4),
            p_control=round(p_c, 4),
            segment=segment,
            recommendation=rec,
        )

    def predict_batch(
        self,
        X: np.ndarray,  # noqa: N803
        feature_names: list[str] | None = None,
    ) -> list[UpliftPrediction]:
        """Предсказать uplift для батча клиентов.

        Predict uplift for a batch of customers.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        names = feature_names or self._feature_names
        return [self.predict_one(dict(zip(names, row.tolist(), strict=False))) for row in X]

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def train_summary(self) -> TrainSummary | None:
        return self._train_summary

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_model(self) -> Any:
        """Создать base-learner: sklearn DecisionTree или numpy fallback."""
        if self._sklearn_available:
            from sklearn.tree import DecisionTreeClassifier

            return DecisionTreeClassifier(
                max_depth=self._config.max_depth,
                min_samples_leaf=self._config.min_samples_leaf,
                random_state=self._config.random_state,
            )
        return _NaiveBayesClassifier()

    def _proba_t(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return _predict_proba_1(self._model_t, X)

    def _proba_c(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return _predict_proba_1(self._model_c, X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TLearner not fitted. Call fit() first.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


def _predict_proba_1(model: Any, X: np.ndarray) -> np.ndarray:  # noqa: N803
    """Извлечь P(class=1) из любого бинарного классификатора."""
    proba = model.predict_proba(X)
    return np.clip(proba[:, 1], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Numpy-only fallback classifier
# ---------------------------------------------------------------------------


class _NaiveBayesClassifier:
    """Laplase-сглаженный наивный байесовский классификатор (numpy-only fallback).

    Laplace-smoothed Naive Bayes classifier — pure-numpy fallback
    used when scikit-learn is not installed (lightweight CI environments).
    """

    def __init__(self) -> None:
        self._feature_means: dict[int, np.ndarray] = {}
        self._class_log_prior: np.ndarray = np.log([0.5, 0.5])
        self._n_features: int = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:  # noqa: N803
        self._n_features = X.shape[1]
        classes = [0, 1]
        n_total = len(Y)
        log_priors = []
        means = {}
        for c in classes:
            mask = c == Y
            # Laplace smoothing на prior
            log_priors.append(np.log((mask.sum() + 1) / (n_total + 2)))
            X_c = X[mask]
            # Среднее значение признаков = оценка вероятности срабатывания
            if len(X_c) > 0:
                means[c] = np.clip(X_c.mean(axis=0), 1e-6, 1 - 1e-6)
            else:
                means[c] = np.full(self._n_features, 0.5)
        self._class_log_prior = np.array(log_priors)
        self._feature_means = means

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        n = len(X)
        log_probs = np.zeros((n, 2))
        for c in [0, 1]:
            mu = self._feature_means.get(c, np.full(self._n_features, 0.5))
            # Bernoulli log-likelihood: X*log(mu) + (1-X)*log(1-mu)
            X_clipped = np.clip(X, 0.0, 1.0)
            ll = X_clipped * np.log(mu) + (1 - X_clipped) * np.log(1 - mu)
            log_probs[:, c] = self._class_log_prior[c] + ll.sum(axis=1)
        # Softmax normalization для стабильности
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return np.clip(probs, 0.0, 1.0)
