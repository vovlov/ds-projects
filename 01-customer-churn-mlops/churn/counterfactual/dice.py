"""DiCE-style Counterfactual Explanations for Customer Churn.

Counterfactual Recourse: answering "What would need to change
for this customer to NOT churn?"

Standard churn models explain *why* a customer might leave (SHAP values),
but give no actionable path forward. Counterfactuals answer the retention
manager's real question: "What concrete actions can we take to keep this
customer?"

Algorithm: Random-Restart Greedy Search (gradient-free, sklearn/numpy only)
- Perturb only actionable features (contract, services — not gender or age)
- Accept perturbations that decrease churn probability
- Deduplication + greedy diversity selection (MMR-style) for final set
- Human-readable explanations for each counterfactual

Regulation: EU AI Act Article 22 — right to meaningful explanation and
actionable recourse for automated decisions affecting individuals.

Sources:
- Wachter et al. 2017 "Counterfactual Explanations Without Opening the Black Box"
  (Harvard JOLT 31:841, doi:10.2139/ssrn.3063289)
- Mothilal et al. 2020 "Explaining Machine Learning Classifiers through
  Diverse Counterfactual Explanations" (DiCE, FAT* '20)
- DiCE-Extended: arxiv 2504.19027 (robust counterfactuals 2025)
- EU AI Act Articles 13, 22 (right to explanation, human oversight)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Feature spaces
# ---------------------------------------------------------------------------

# Actionable continuous features with their valid ranges.
# Non-actionable numerics: SeniorCitizen (immutable), tenure (only increases in real life)
ACTIONABLE_NUMERIC: dict[str, tuple[float, float]] = {
    "MonthlyCharges": (18.0, 120.0),
    "TotalCharges": (0.0, 9000.0),
}

# Tenure is special: actionable but can only *increase* (can't travel back in time)
TENURE_RANGE: tuple[int, int] = (0, 72)

# Actionable categorical features with their valid values
ACTIONABLE_CATEGORICAL: dict[str, list[str]] = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "PaperlessBilling": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
}

_ALL_ACTIONABLE: list[str] = list(ACTIONABLE_NUMERIC) + ["tenure"] + list(ACTIONABLE_CATEGORICAL)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualConfig:
    """Configuration for the counterfactual generator.

    Конфигурация генератора контрфактических объяснений.
    """

    n_counterfactuals: int = 3
    max_iterations: int = 2000
    # Target: churn probability below this threshold = "not churn"
    target_probability: float = 0.35
    random_state: int = 42
    # Max number of features to change in one perturbation
    max_changes_per_step: int = 3


@dataclass
class Counterfactual:
    """Single counterfactual explanation.

    Одно контрфактическое объяснение: минимальные изменения → другой исход.
    """

    features: dict[str, Any]
    changes: dict[str, Any]
    churn_probability: float
    distance: float
    feasibility_score: float  # 1 - distance; higher = easier to implement

    def to_plain_text(self) -> list[str]:
        """Human-readable list of required changes.

        Человекочитаемый список действий для предотвращения оттока.
        """
        lines: list[str] = []
        for feat, new_val in self.changes.items():
            if feat == "Contract":
                lines.append(f"Switch contract to '{new_val}'")
            elif feat == "tenure":
                lines.append(f"Retain customer for at least {new_val} months")
            elif feat == "MonthlyCharges":
                lines.append(f"Offer plan at {new_val:.2f}/month")
            elif feat == "TotalCharges":
                lines.append(f"Adjust total charges to {new_val:.2f}")
            elif feat == "InternetService":
                lines.append(f"Change internet service to '{new_val}'")
            elif feat == "PaymentMethod":
                lines.append(f"Switch payment method to '{new_val}'")
            elif feat == "PaperlessBilling":
                action = "Enable" if new_val == "Yes" else "Disable"
                lines.append(f"{action} paperless billing")
            elif feat == "MultipleLines":
                lines.append(f"Set multiple lines to '{new_val}'")
            elif feat in ACTIONABLE_CATEGORICAL:
                action = "Add" if new_val == "Yes" else "Remove"
                readable = feat.replace("_", " ")
                lines.append(f"{action} {readable}")
            else:
                lines.append(f"Change {feat} to '{new_val}'")
        return lines


@dataclass
class CounterfactualResult:
    """Result of counterfactual generation for one customer.

    Результат генерации контрфактических объяснений для одного клиента.
    """

    original_probability: float
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    success: bool = False
    n_tried: int = 0


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DIcEChurn:
    """DiCE-inspired counterfactual generator for churn prediction.

    Генератор контрфактических объяснений для предсказания оттока клиентов.
    Использует gradient-free random-restart greedy search (numpy-only).
    """

    def __init__(self, config: CounterfactualConfig | None = None) -> None:
        self.config = config or CounterfactualConfig()

    def generate(
        self,
        original: dict[str, Any],
        predict_fn: Callable[[dict[str, Any]], float],
    ) -> CounterfactualResult:
        """Generate diverse counterfactual explanations.

        Сгенерировать разнообразные контрфактические объяснения.

        Args:
            original: Customer features (19 fields matching CustomerInput)
            predict_fn: Callable mapping feature dict → churn probability in [0, 1]

        Returns:
            CounterfactualResult with up to n_counterfactuals actionable suggestions
        """
        original_prob = predict_fn(original)
        rng = np.random.default_rng(self.config.random_state)
        candidates: list[Counterfactual] = []

        for _ in range(self.config.max_iterations):
            candidate = self._perturb(original, rng)
            prob = predict_fn(candidate)

            if prob < self.config.target_probability:
                dist = self._distance(original, candidate)
                changes = {k: v for k, v in candidate.items() if v != original.get(k)}
                candidates.append(
                    Counterfactual(
                        features=candidate,
                        changes=changes,
                        churn_probability=prob,
                        distance=dist,
                        feasibility_score=round(1.0 - dist, 4),
                    )
                )

        selected = self._select_diverse(candidates, self.config.n_counterfactuals)
        return CounterfactualResult(
            original_probability=round(original_prob, 4),
            counterfactuals=selected,
            success=len(selected) > 0,
            n_tried=self.config.max_iterations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perturb(
        self,
        original: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Randomly change 1–max_changes_per_step actionable features.

        Случайно изменить от 1 до max_changes_per_step признаков.
        """
        candidate = dict(original)
        n_changes = int(rng.integers(1, self.config.max_changes_per_step + 1))

        # Sample without replacement from actionable feature pool
        pool = np.array(_ALL_ACTIONABLE)
        idxs = rng.choice(len(pool), size=min(n_changes, len(pool)), replace=False)
        features_to_change = pool[idxs]

        for feat in features_to_change:
            if feat == "tenure":
                low, high = TENURE_RANGE
                current = int(candidate.get("tenure", 0))
                # Tenure can only increase (retention period)
                new_val = int(rng.integers(current, high + 1)) if current < high else current
                candidate["tenure"] = new_val

            elif feat in ACTIONABLE_NUMERIC:
                low, high = ACTIONABLE_NUMERIC[feat]
                current = float(candidate.get(feat, (low + high) / 2))
                step = (high - low) * 0.15
                delta = float(rng.uniform(-step, step))
                new_val = round(float(np.clip(current + delta, low, high)), 2)
                candidate[feat] = new_val

            elif feat in ACTIONABLE_CATEGORICAL:
                options = [v for v in ACTIONABLE_CATEGORICAL[feat] if v != candidate.get(feat)]
                if options:
                    candidate[feat] = str(rng.choice(options))

        return candidate

    def _distance(self, original: dict[str, Any], candidate: dict[str, Any]) -> float:
        """Normalised mean distance across actionable features.

        Нормализованное среднее расстояние по изменяемым признакам.
        Continuous: |delta| / range; Categorical: 0 if same, 1 if different.
        """
        total = 0.0
        n = 0

        # Numeric (tenure + 2 continuous)
        for feat, (low, high) in ACTIONABLE_NUMERIC.items():
            r = high - low
            if r > 0 and feat in original and feat in candidate:
                total += abs(float(candidate[feat]) - float(original[feat])) / r
                n += 1

        if "tenure" in original and "tenure" in candidate:
            lo, hi = TENURE_RANGE
            r = hi - lo
            if r > 0:
                total += abs(int(candidate["tenure"]) - int(original["tenure"])) / r
            n += 1

        for feat in ACTIONABLE_CATEGORICAL:
            if feat in original and feat in candidate:
                total += 0.0 if original[feat] == candidate[feat] else 1.0
                n += 1

        return round(total / n, 4) if n > 0 else 0.0

    def _select_diverse(
        self,
        candidates: list[Counterfactual],
        n: int,
    ) -> list[Counterfactual]:
        """Select N diverse counterfactuals using greedy MMR-style selection.

        Отобрать N разнообразных контрфактических объяснений через жадный MMR.
        Always prefers low churn probability, then maximises inter-CF distance.
        """
        if not candidates:
            return []

        # Deduplicate by change fingerprint
        seen: set[frozenset] = set()
        unique: list[Counterfactual] = []
        for c in sorted(candidates, key=lambda x: (x.churn_probability, x.distance)):
            key = frozenset((k, str(v)) for k, v in c.changes.items())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        if len(unique) <= n:
            return unique

        # Greedy diversity: always add CF most different from already-selected ones
        selected = [unique[0]]
        while len(selected) < n:
            best: Counterfactual | None = None
            best_min_dist = -1.0
            for cand in unique:
                if cand in selected:
                    continue
                min_d = min(self._distance(s.features, cand.features) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best = cand
            if best is None:
                break
            selected.append(best)

        return selected
