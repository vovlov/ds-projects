"""Cox Proportional Hazards model for customer churn survival analysis.

Модель пропорциональных рисков Кокса для анализа выживаемости клиентов.

Estimates WHEN customers will churn by modeling the hazard function:
    h(t | x) = h₀(t) · exp(β·x)

where h₀(t) is the baseline hazard and exp(β·x) is the individual risk multiplier.

Hazard ratio exp(β_j): customer with feature x_j = 1 vs x_j = 0 has
exp(β_j) times higher churn hazard (multiplicative effect).

Fitted via maximum partial likelihood (Breslow 1972).
Baseline hazard estimated by Breslow estimator.
Prediction: S(t|x) = S₀(t)^exp(β·x) where S₀ = exp(-Λ₀).

Sources:
    Cox 1972 J. R. Stat. Soc. B 34(2):187-220 (original Cox PH)
    Breslow 1972 Biometrics 28(1):89-99 (Breslow estimator)
    Efron 1977 JASA 72(359):557-565 (tied event handling)
    arxiv:2510.11604 (survival analysis for churn, 2025)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoxPHConfig:
    """Конфигурация модели Кокса.

    Configuration for the Cox PH model.
    """

    max_iter: int = 200
    lr: float = 0.01
    tol: float = 1e-6
    l2_reg: float = 0.01
    random_state: int = 42


@dataclass
class CoxPHResult:
    """Результат обучения модели Кокса.

    Fitted Cox PH model summary.
    """

    coef: list[float]
    hazard_ratios: list[float]
    feature_names: list[str]
    log_partial_likelihood: float
    concordance_index: float
    n_samples: int
    n_events: int

    def to_dict(self) -> dict:
        """Сериализовать в словарь для API."""
        return {
            "coef": [round(c, 6) for c in self.coef],
            "hazard_ratios": [round(hr, 6) for hr in self.hazard_ratios],
            "feature_names": self.feature_names,
            "log_partial_likelihood": round(self.log_partial_likelihood, 6),
            "concordance_index": round(self.concordance_index, 4),
            "n_samples": self.n_samples,
            "n_events": self.n_events,
        }


@dataclass
class SurvivalPrediction:
    """Предсказание выживаемости для одного клиента.

    Survival prediction for a single customer.
    """

    log_hazard: float
    hazard_ratio: float
    median_survival: float | None
    survival_at_times: list[float]
    risk_group: str


class CoxPHModel:
    """Cox Proportional Hazards model — partial likelihood via gradient descent.

    Модель пропорциональных рисков Кокса.

    Partial log-likelihood:
        l(β) = Σ_{i: E_i=1} [β·x_i - log(Σ_{j: T_j ≥ T_i} exp(β·x_j))]

    Gradient:
        ∂l/∂β = Σ_{i: E_i=1} [x_i - Ē_β(x | T ≥ T_i)]

    Optimised by gradient ascent with L2 regularisation (ridge).
    Baseline hazard: Breslow estimator.

    Sources:
        Cox 1972, Breslow 1972, Efron 1977.
    """

    def __init__(self, config: CoxPHConfig | None = None) -> None:
        self._config = config or CoxPHConfig()
        self._beta: np.ndarray | None = None
        self._feature_names: list[str] = []
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        # Breslow cumulative baseline hazard: list of (time, cumhazard)
        self._breslow_times: np.ndarray | None = None
        self._breslow_cumhaz: np.ndarray | None = None
        self._is_fitted = False
        self._result: CoxPHResult | None = None

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        durations: np.ndarray,
        events: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> CoxPHModel:
        """Обучить модель Кокса на данных выживаемости.

        Fit Cox PH on survival data via partial likelihood gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features).
            durations: Observed durations. Non-negative.
            events: Event indicators (1=churned, 0=censored).
            feature_names: Optional list of feature names for interpretability.

        Returns:
            self for chaining.
        """
        X = np.asarray(X, dtype=float)
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)

        n, p = X.shape
        if len(durations) != n or len(events) != n:
            raise ValueError("X, durations, and events must have the same number of rows")
        if events.sum() == 0:
            raise ValueError("No events (all censored) — cannot fit Cox PH")

        self._feature_names = feature_names or [f"x{i}" for i in range(p)]

        # Standardise features for numerical stability
        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0)
        # Avoid division by zero for constant features
        self._feature_std = np.where(self._feature_std < 1e-8, 1.0, self._feature_std)
        X_scaled = (X - self._feature_mean) / self._feature_std

        # Sort by duration (ascending) — required for risk-set computation
        order = np.argsort(durations, kind="stable")
        X_s = X_scaled[order]
        t_s = durations[order]
        e_s = events[order]

        # Gradient descent on negative partial log-likelihood
        rng = np.random.default_rng(self._config.random_state)
        beta = rng.standard_normal(p) * 0.01

        lr = self._config.lr
        l2 = self._config.l2_reg
        prev_nll = np.inf

        for _iteration in range(self._config.max_iter):
            nll, grad = _nll_and_grad(beta, X_s, e_s)
            nll_total = nll + 0.5 * l2 * float(np.dot(beta, beta))
            grad_total = grad + l2 * beta

            beta = beta - lr * grad_total

            if abs(prev_nll - nll_total) < self._config.tol:
                break
            prev_nll = nll_total

        self._beta = beta

        # Breslow estimator for baseline cumulative hazard
        self._breslow_times, self._breslow_cumhaz = _breslow_estimator(beta, X_s, t_s, e_s)

        # Concordance index
        eta = X_scaled @ beta
        c_index = _concordance_index(eta, durations, events)

        log_pll = -float(_nll_and_grad(beta, X_s, e_s)[0])

        # Coefficients in original (unscaled) space: β_orig = β_scaled / std
        beta_orig = beta / self._feature_std

        self._result = CoxPHResult(
            coef=beta_orig.tolist(),
            hazard_ratios=[float(np.exp(b)) for b in beta_orig],
            feature_names=self._feature_names,
            log_partial_likelihood=log_pll,
            concordance_index=c_index,
            n_samples=n,
            n_events=int(events.sum()),
        )
        self._is_fitted = True
        return self

    @property
    def result(self) -> CoxPHResult:
        """Результат обучения после fit()."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before accessing result.")
        return self._result  # type: ignore[return-value]

    def predict_log_hazard(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Предсказать логарифм рискового коэффициента для новых клиентов.

        Predict log-hazard ratio (β·x_scaled) for new customers.
        Higher value = higher hazard = shorter expected survival.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self._feature_mean) / self._feature_std  # type: ignore[operator]
        return X_scaled @ self._beta  # type: ignore[operator]

    def predict_survival_function(
        self,
        X: np.ndarray,  # noqa: N803
        times: np.ndarray,
    ) -> np.ndarray:
        """Предсказать кривую выживаемости S(t|x) для каждого клиента.

        Predict survival function S(t|x) = S₀(t)^exp(β·x).

        Args:
            X: Feature matrix (n_samples, n_features).
            times: Time points at which to evaluate S(t).

        Returns:
            Array of shape (n_samples, len(times)): S(t|x) values.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        log_hazard = self.predict_log_hazard(X)
        times = np.asarray(times, dtype=float)

        # Breslow cumulative baseline hazard at requested times
        cum_haz_baseline = np.interp(
            times,
            self._breslow_times,  # type: ignore[arg-type]
            self._breslow_cumhaz,  # type: ignore[arg-type]
            left=0.0,
            right=float(self._breslow_cumhaz[-1]),  # type: ignore[index]
        )

        # S(t|x) = exp(-exp(β·x) * Λ₀(t))
        # Shape: (n_samples, n_times)
        surv = np.exp(-np.outer(np.exp(log_hazard), cum_haz_baseline))
        return surv

    def predict_median_survival(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Предсказать медианное время до оттока для каждого клиента.

        Predict median survival time (50th percentile) for each customer.
        Returns NaN if S(t) never drops below 0.5 in the observed range.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array (n_samples,) of median survival times.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        log_hazard = self.predict_log_hazard(X)
        n_samples = len(log_hazard)

        # Λ₀(t) at event times
        cumhaz_base = self._breslow_cumhaz  # type: ignore[assignment]
        times = self._breslow_times  # type: ignore[assignment]

        medians = np.full(n_samples, np.nan)
        for i, lh in enumerate(log_hazard):
            # S(t|x) = exp(-exp(lh) * Λ₀(t)) ≤ 0.5
            # => Λ₀(t) ≥ log(2) / exp(lh)
            threshold = np.log(2.0) / np.exp(lh)
            idx = np.searchsorted(cumhaz_base, threshold)
            if idx < len(times):
                medians[i] = times[idx]

        return medians

    def predict(
        self,
        X: np.ndarray,  # noqa: N803
        eval_times: list[float] | None = None,
    ) -> list[SurvivalPrediction]:
        """Полное предсказание для каждого клиента — используется в API.

        Full per-customer prediction for API endpoints.

        Args:
            X: Feature matrix (n_samples, n_features).
            eval_times: Time points to evaluate S(t). Defaults to KM percentiles.

        Returns:
            List of SurvivalPrediction for each customer.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        X = np.asarray(X, dtype=float)
        n_samples = len(X)

        if eval_times is None:
            # Use percentiles of observed event times
            t_max = float(self._breslow_times[-1]) if len(self._breslow_times) > 0 else 60  # type: ignore[arg-type]
            eval_times = [t_max * q for q in [0.1, 0.25, 0.5, 0.75, 1.0]]

        eval_times_arr = np.asarray(eval_times, dtype=float)
        surv_matrix = self.predict_survival_function(X, eval_times_arr)
        median_times = self.predict_median_survival(X)
        log_hazards = self.predict_log_hazard(X)

        # Risk groups: low/medium/high based on log-hazard tertiles
        lh_sorted = np.sort(log_hazards)
        q33 = lh_sorted[max(0, int(n_samples * 0.33) - 1)]
        q67 = lh_sorted[max(0, int(n_samples * 0.67) - 1)]

        results = []
        for i in range(n_samples):
            lh = float(log_hazards[i])
            if n_samples < 3:
                risk_group = "medium"
            elif lh <= q33:
                risk_group = "low"
            elif lh <= q67:
                risk_group = "medium"
            else:
                risk_group = "high"

            median_t = median_times[i]
            results.append(
                SurvivalPrediction(
                    log_hazard=round(lh, 4),
                    hazard_ratio=round(float(np.exp(lh)), 4),
                    median_survival=None if np.isnan(median_t) else round(float(median_t), 2),
                    survival_at_times=[round(float(s), 4) for s in surv_matrix[i]],
                    risk_group=risk_group,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nll_and_grad(
    beta: np.ndarray,
    x_sorted: np.ndarray,  # noqa: N803
    events_sorted: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Negative partial log-likelihood and its gradient.

    Uses reversed cumulative sums for O(n·p) computation.
    Breslow approximation for tied event times.

    Нормализация для численной стабильности: η_i = β·x_i - max(β·x).
    Вычитание max не меняет значение ∂l/∂β, но предотвращает exp-overflow.
    """
    eta = x_sorted @ beta
    # Subtract max for numerical stability — cancels in ratio
    eta_shift = eta - eta.max()
    exp_eta = np.exp(eta_shift)

    n = len(eta)
    # Reversed cumulative sums: entry i = Σ_{j >= i} value
    rev_sum_exp = np.cumsum(exp_eta[::-1])[::-1]  # (n,)
    rev_sum_exp_x = np.cumsum((exp_eta[:, None] * x_sorted)[::-1], axis=0)[::-1]  # (n, p)

    nll = 0.0
    grad = np.zeros_like(beta)

    for i in range(n):
        if events_sorted[i] == 0:
            continue
        nll += float(np.log(rev_sum_exp[i]) - eta_shift[i])
        weighted_mean_x = rev_sum_exp_x[i] / rev_sum_exp[i]
        grad += weighted_mean_x - x_sorted[i]

    return nll, grad


def _breslow_estimator(
    beta: np.ndarray,
    x_sorted: np.ndarray,  # noqa: N803
    t_sorted: np.ndarray,
    e_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimator for cumulative baseline hazard Λ₀(t).

    Λ₀(t) = Σ_{t_i ≤ t, E_i=1} 1 / Σ_{j: T_j ≥ T_i} exp(β·X_j)

    Returns times and cumulative hazard arrays aligned at event times.
    """
    eta = x_sorted @ beta
    exp_eta = np.exp(eta - eta.max())

    n = len(t_sorted)
    rev_sum_exp = np.cumsum(exp_eta[::-1])[::-1]

    event_times: list[float] = []
    increments: list[float] = []

    i = 0
    while i < n:
        if e_sorted[i] == 0:
            i += 1
            continue

        t_i = t_sorted[i]
        risk_denom = float(rev_sum_exp[i])

        # Count tied events at t_i
        d_i = 0
        j = i
        while j < n and t_sorted[j] == t_i:
            if e_sorted[j] == 1:
                d_i += 1
            j += 1

        if risk_denom > 0 and d_i > 0:
            event_times.append(float(t_i))
            increments.append(d_i / risk_denom)

        i = j

    if not event_times:
        return np.array([0.0]), np.array([0.0])

    times_arr = np.array(event_times)
    cum_haz = np.cumsum(increments)

    # Prepend t=0 with Λ₀(0) = 0
    times_arr = np.concatenate([[0.0], times_arr])
    cum_haz = np.concatenate([[0.0], cum_haz])

    return times_arr, cum_haz


def _concordance_index(
    log_hazard: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> float:
    """Concordance index (C-index / Harrell's C).

    Индекс согласованности — Harrell's C — оценивает качество ранжирования:
    из всех пар (i, j) где T_i < T_j и событие у i наблюдалось,
    доля пар, где модель правильно присвоила бóльший риск i.

    C-index = 0.5 для случайной модели, 1.0 для идеальной.
    """
    n = len(durations)
    concordant = 0
    permissible = 0

    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if durations[j] > durations[i]:
                permissible += 1
                if log_hazard[i] > log_hazard[j]:
                    concordant += 1
                elif log_hazard[i] == log_hazard[j]:
                    concordant += 0.5

    if permissible == 0:
        return 0.5
    return float(concordant / permissible)


def generate_synthetic_survival_data(
    n_samples: int = 300,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Генерировать синтетические данные выживаемости для телеком чёрна.

    Generate synthetic telco-style survival data for demonstration.

    Data generation process:
    - Features: monthly_charges, contract_type, tech_support, senior_citizen,
                tenure_ratio (all standardised)
    - Hazard: λ(t|x) = λ₀ · exp(β·x) with λ₀ = 1/36 (36 month mean tenure)
    - Event times: T = -log(U) / λ(x) (exponential distribution)
    - Censoring: C ~ Uniform(6, 72), observed = min(T, C)

    Returns:
        X: Feature matrix (n_samples, n_features)
        durations: Observed durations in months
        events: 1 = churned, 0 = censored
        feature_names: List of feature names
    """
    rng = np.random.default_rng(seed)

    feature_names = [
        "monthly_charges",
        "contract_type",
        "tech_support",
        "senior_citizen",
        "num_services",
    ][:n_features]

    n_feat = min(n_features, 5)
    X = np.zeros((n_samples, n_feat))

    # monthly_charges: [20, 120]
    X[:, 0] = rng.uniform(20, 120, n_samples)
    if n_feat > 1:
        # contract_type: 0=month-to-month, 1=one-year, 2=two-year
        X[:, 1] = rng.choice([0, 1, 2], n_samples, p=[0.55, 0.25, 0.20])
    if n_feat > 2:
        # tech_support: 0=no, 1=yes
        X[:, 2] = rng.binomial(1, 0.45, n_samples).astype(float)
    if n_feat > 3:
        # senior_citizen: 0=no, 1=yes
        X[:, 3] = rng.binomial(1, 0.16, n_samples).astype(float)
    if n_feat > 4:
        # num_services: [1, 6]
        X[:, 4] = rng.integers(1, 7, n_samples).astype(float)

    # True coefficients: higher charges → higher hazard, longer contract → lower hazard
    true_beta = np.array([0.4, -0.5, -0.3, 0.25, -0.1])[:n_feat]

    # Standardise for hazard computation
    x_mean = X.mean(axis=0)
    x_std = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
    X_scaled = (X - x_mean) / x_std

    log_hazard = X_scaled @ true_beta
    # Baseline: mean tenure 36 months
    base_hazard = 1.0 / 36.0
    individual_hazard = base_hazard * np.exp(log_hazard)

    # Exponential event times
    u = rng.uniform(0, 1, n_samples)
    event_times = -np.log(u) / individual_hazard

    # Random censoring: uniform [6, 72] months
    censor_times = rng.uniform(6, 72, n_samples)

    durations = np.minimum(event_times, censor_times)
    events = (event_times <= censor_times).astype(int)

    return X, durations, events, feature_names
