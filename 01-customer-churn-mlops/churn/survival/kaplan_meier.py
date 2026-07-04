"""Kaplan-Meier non-parametric survival estimator for customer churn analysis.

Оценщик кривой выживаемости Каплана-Мейера для анализа оттока клиентов.

Answers «When will the customer churn?» vs. binary models that only answer «Will they?».
Handles right-censored data: customers who are still active at observation time.

Example:
    km = KaplanMeierEstimator()
    km.fit(durations=[12, 24, 6, 36], events=[1, 0, 1, 1])
    result = km.result
    print(result.median_survival)  # e.g. 12.0 months

Sources:
    Kaplan & Meier 1958 JASA 53(282):457-481 (original KM estimator)
    Greenwood 1926 J. Hyg. 33(2):216-226 (variance formula)
    Mantel 1966 Cancer Chemotherapy Reports 50(3) (log-rank test)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KMConfig:
    """Конфигурация оценщика Каплана-Мейера.

    Configuration for the Kaplan-Meier estimator.
    """

    alpha: float = 0.05


@dataclass
class KMResult:
    """Результат оценки кривой выживаемости.

    Kaplan-Meier survival curve result.
    """

    times: list[float]
    survival: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    n_at_risk: list[int]
    n_events: list[int]
    median_survival: float | None
    n_total: int
    n_events_total: int

    def to_dict(self) -> dict:
        """Сериализовать в словарь для API ответа."""
        return {
            "times": self.times,
            "survival": self.survival,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_at_risk": self.n_at_risk,
            "n_events": self.n_events,
            "median_survival": self.median_survival,
            "n_total": self.n_total,
            "n_events_total": self.n_events_total,
        }


@dataclass
class LogRankResult:
    """Результат логрангового критерия сравнения двух групп.

    Log-rank test result for comparing two survival groups.
    """

    statistic: float
    p_value: float
    reject_h0: bool


class KaplanMeierEstimator:
    """Non-parametric Kaplan-Meier survival curve estimator.

    Непараметрический оценщик кривой выживаемости.

    S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)

    Где:
    - t_i — моменты событий (отток)
    - d_i — количество событий в момент t_i
    - n_i — количество клиентов под наблюдением перед t_i

    Censoring: клиенты, ещё не ушедшие к моменту наблюдения (event=0),
    уменьшают n_i, но не вносят вклад в d_i.

    Sources:
        Kaplan & Meier 1958 JASA 53(282):457-481
        Greenwood 1926 J. Hyg. 33(2):216-226
    """

    def __init__(self, config: KMConfig | None = None) -> None:
        self._config = config or KMConfig()
        self._result: KMResult | None = None
        self._is_fitted = False

    def fit(
        self,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> KaplanMeierEstimator:
        """Обучить оценщик на данных выживаемости.

        Fit the estimator on survival data.

        Args:
            durations: Observed durations (e.g., months of tenure). Non-negative.
            events: Event indicators — 1=event observed (churned), 0=censored (active).

        Returns:
            self for chaining.
        """
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)

        if len(durations) == 0:
            raise ValueError("durations must not be empty")
        if len(durations) != len(events):
            raise ValueError("durations and events must have the same length")
        if np.any(durations < 0):
            raise ValueError("durations must be non-negative")

        order = np.argsort(durations, kind="stable")
        t = durations[order]
        e = events[order]

        n = len(t)
        S = 1.0
        z = float(
            np.abs(
                np.percentile(
                    np.random.default_rng(0).normal(size=100000), (1 - self._config.alpha / 2) * 100
                )
            )
        )
        z = 1.96 if abs(self._config.alpha - 0.05) < 1e-9 else z

        # Greenwood accumulator for log(-log(S)) CI
        greenwood_sum = 0.0

        unique_times: list[float] = []
        s_values: list[float] = []
        ci_lowers: list[float] = []
        ci_uppers: list[float] = []
        n_at_risk_list: list[int] = []
        n_events_list: list[int] = []

        i = 0
        while i < n:
            t_i = t[i]
            n_i = n - i

            j = i
            while j < n and t[j] == t_i:
                j += 1

            d_i = int(e[i:j].sum())

            if d_i > 0:
                S = S * (1.0 - d_i / n_i)

                if n_i > d_i and S > 0:
                    greenwood_sum += d_i / (n_i * (n_i - d_i))

                # Log-log confidence interval (Hall & Wellner, more stable than plain logit)
                if 0 < S < 1 and greenwood_sum > 0:
                    log_s = np.log(S)
                    log_neg_log_s = np.log(-log_s)
                    se_loglog = np.sqrt(greenwood_sum) / abs(log_s)
                    ci_lo = float(np.exp(-np.exp(log_neg_log_s + z * se_loglog)))
                    ci_hi = float(np.exp(-np.exp(log_neg_log_s - z * se_loglog)))
                elif S == 0:
                    ci_lo, ci_hi = 0.0, 0.0
                else:
                    ci_lo, ci_hi = max(0.0, S - 0.1), min(1.0, S + 0.1)

                unique_times.append(float(t_i))
                s_values.append(round(float(S), 6))
                ci_lowers.append(round(float(np.clip(ci_lo, 0.0, 1.0)), 6))
                ci_uppers.append(round(float(np.clip(ci_hi, 0.0, 1.0)), 6))
                n_at_risk_list.append(int(n_i))
                n_events_list.append(int(d_i))

            i = j

        # Median: first time where S(t) ≤ 0.5
        median_survival: float | None = None
        for t_val, s_val in zip(unique_times, s_values, strict=True):
            if s_val <= 0.5:
                median_survival = float(t_val)
                break

        self._result = KMResult(
            times=unique_times,
            survival=s_values,
            ci_lower=ci_lowers,
            ci_upper=ci_uppers,
            n_at_risk=n_at_risk_list,
            n_events=n_events_list,
            median_survival=median_survival,
            n_total=int(n),
            n_events_total=int(events.sum()),
        )
        self._is_fitted = True
        return self

    @property
    def result(self) -> KMResult:
        """Получить результат после fit().

        Get the result after fitting.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before accessing result.")
        return self._result  # type: ignore[return-value]

    @staticmethod
    def log_rank_test(
        durations_a: np.ndarray,
        events_a: np.ndarray,
        durations_b: np.ndarray,
        events_b: np.ndarray,
        alpha: float = 0.05,
    ) -> LogRankResult:
        """Логранговый критерий сравнения двух групп выживаемости.

        Log-rank test comparing survival curves of two groups.

        H_0: S_A(t) = S_B(t) для всех t (нет различий в выживаемости).
        H_0: Both groups have identical survival functions.

        Statistic ~ Chi²(1) under H_0 by Wilcoxon 1945 / Mantel 1966.

        Args:
            durations_a, events_a: Group A (e.g., high-churn-risk segment).
            durations_b, events_b: Group B (e.g., low-churn-risk segment).
            alpha: Significance level for reject_h0 flag.

        Returns:
            LogRankResult with chi2 statistic, p-value, and rejection flag.
        """
        t_a = np.asarray(durations_a, dtype=float)
        e_a = np.asarray(events_a, dtype=int)
        t_b = np.asarray(durations_b, dtype=float)
        e_b = np.asarray(events_b, dtype=int)

        # Union of event times
        all_event_times = np.unique(np.concatenate([t_a[e_a == 1], t_b[e_b == 1]]))

        O_minus_E_sum = 0.0
        var_sum = 0.0

        for t_j in all_event_times:
            n_aj = int((t_a >= t_j).sum())
            n_bj = int((t_b >= t_j).sum())
            n_j = n_aj + n_bj

            if n_j <= 1:
                continue

            d_aj = int(((t_a == t_j) & (e_a == 1)).sum())
            d_bj = int(((t_b == t_j) & (e_b == 1)).sum())
            d_j = d_aj + d_bj

            E_aj = n_aj * d_j / n_j

            # Hypergeometric variance
            if n_j > 1 and d_j > 0:
                var_j = n_aj * n_bj * d_j * (n_j - d_j) / (n_j**2 * (n_j - 1))
            else:
                var_j = 0.0

            O_minus_E_sum += d_aj - E_aj
            var_sum += var_j

        if var_sum <= 0:
            return LogRankResult(statistic=0.0, p_value=1.0, reject_h0=False)

        chi2 = float(O_minus_E_sum**2 / var_sum)
        p_value = _chi2_sf(chi2, df=1)

        return LogRankResult(
            statistic=round(chi2, 6),
            p_value=round(p_value, 6),
            reject_h0=bool(p_value < alpha),
        )


def _chi2_sf(x: float, df: int = 1) -> float:
    """Chi-squared survival function. Graceful fallback without scipy."""
    try:
        from scipy.stats import chi2

        return float(chi2.sf(x, df=df))
    except ImportError:
        # Wilson-Hilferty normal approximation for chi²(1): P(χ²₁ > x) ≈ P(Z > √x)
        z = float(np.sqrt(max(0.0, x)))
        # Standard normal survival: 1 - Φ(z) via erfc
        # erfc(z/√2) / 2 = P(Z > z) for Z ~ N(0,1)
        import math

        return float(math.erfc(z / math.sqrt(2.0)))
