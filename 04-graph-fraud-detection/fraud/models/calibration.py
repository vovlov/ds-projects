"""Probability calibration for fraud scores.

Uncalibrated tree-ensemble outputs are often over-confident in the tails.
Calibration maps raw scores → reliable P(fraud|score) so that business
decision thresholds (e.g. "block if P > 0.7") mean what they say.

References:
- Zadrozny & Elkan 2001 (isotonic, ICML)
- Platt 1999 (sigmoid calibration, SVM)
- Guo et al. 2017 (ECE, ICML)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class CalibrationBin:
    """One bin in the reliability diagram."""

    bin_lower: float
    bin_upper: float
    mean_predicted: float
    fraction_positive: float
    count: int


@dataclass
class CalibrationResult:
    """Calibration quality metrics + reliability diagram data."""

    method: str
    n_calibration_samples: int
    ece: float  # Expected Calibration Error (Guo et al. 2017)
    mce: float  # Maximum Calibration Error (worst bin gap)
    brier_score: float  # Proper scoring rule: MSE(proba, label)
    n_bins: int
    bins: list[CalibrationBin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n_calibration_samples": self.n_calibration_samples,
            "ece": round(self.ece, 6),
            "mce": round(self.mce, 6),
            "brier_score": round(self.brier_score, 6),
            "n_bins": self.n_bins,
            "bins": [
                {
                    "bin_lower": round(b.bin_lower, 3),
                    "bin_upper": round(b.bin_upper, 3),
                    "mean_predicted": round(b.mean_predicted, 4),
                    "fraction_positive": round(b.fraction_positive, 4),
                    "count": b.count,
                }
                for b in self.bins
            ],
        }


def _compute_ece(
    probas: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, list[CalibrationBin]]:
    """Compute ECE, MCE, and reliability diagram bins.

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|  (Guo 2017).
    """
    bins_data: list[CalibrationBin] = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probas)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = (probas >= lo) & (probas <= hi)
        else:
            mask = (probas >= lo) & (probas < hi)

        count = int(mask.sum())
        if count == 0:
            bins_data.append(
                CalibrationBin(
                    bin_lower=float(lo),
                    bin_upper=float(hi),
                    mean_predicted=float((lo + hi) / 2),
                    fraction_positive=0.0,
                    count=0,
                )
            )
            continue

        mean_pred = float(probas[mask].mean())
        frac_pos = float(labels[mask].mean())
        gap = abs(frac_pos - mean_pred)
        ece += (count / n) * gap
        mce = max(mce, gap)

        bins_data.append(
            CalibrationBin(
                bin_lower=float(lo),
                bin_upper=float(hi),
                mean_predicted=mean_pred,
                fraction_positive=frac_pos,
                count=count,
            )
        )

    return float(ece), float(mce), bins_data


class _PlattScaler:
    """Platt scaling: fits sigmoid σ(a·s + b) via gradient descent.

    Preferred over logistic regression wrapper to avoid sklearn dependency
    in this module (keeps fraud/ portable).
    """

    def __init__(self, lr: float = 0.01, n_iter: int = 1000) -> None:
        self._a = 1.0
        self._b = 0.0
        self._lr = lr
        self._n_iter = n_iter
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        a, b = self._a, self._b
        for _ in range(self._n_iter):
            logits = a * scores + b
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            err = p - labels
            grad_a = float(np.mean(err * scores))
            grad_b = float(np.mean(err))
            a -= self._lr * grad_a
            b -= self._lr * grad_b
        self._a, self._b = a, b
        self.fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        logits = self._a * scores + self._b
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


class _IsotonicCalibrator:
    """Isotonic regression calibration (non-parametric, monotone).

    Uses sklearn's implementation which guarantees monotonicity via
    pool-adjacent-violators algorithm (Ayer et al. 1955).
    """

    def __init__(self) -> None:
        self._iso = None
        self.fitted = False

    def _get_iso(self):
        from sklearn.isotonic import IsotonicRegression  # lazy import

        return IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self._iso = self._get_iso()
        self._iso.fit(scores, labels)
        self.fitted = True

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(self._iso.transform(scores), dtype=float)


class FraudCalibrator:
    """Calibrate raw fraud scores → reliable P(fraud | score).

    Supports Platt scaling (sigmoid) and isotonic regression.
    Isotonic is preferred when n_calibration >= 1000 (Zadrozny & Elkan 2001).
    Platt is more stable on small calibration sets.

    Usage:
        calibrator = FraudCalibrator(method="isotonic")
        result = calibrator.fit(raw_scores, labels)
        calibrated = calibrator.calibrate(new_scores)
    """

    def __init__(
        self,
        method: Literal["platt", "isotonic"] = "isotonic",
        n_bins: int = 10,
    ) -> None:
        self.method = method
        self.n_bins = n_bins
        self._calibrator: _PlattScaler | _IsotonicCalibrator | None = None
        self._raw_ece: float | None = None
        self._cal_ece: float | None = None
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> CalibrationResult:
        """Fit calibrator, compute pre- and post-calibration ECE.

        Returns CalibrationResult with metrics AFTER calibration.
        """
        raw_scores = np.asarray(raw_scores, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()
        n = len(raw_scores)

        if n < 10:
            raise ValueError(f"Need at least 10 calibration samples, got {n}")

        # Build the right backend
        if self.method == "platt":
            cal: _PlattScaler | _IsotonicCalibrator = _PlattScaler()
        else:
            cal = _IsotonicCalibrator()

        cal.fit(raw_scores, labels)
        self._calibrator = cal
        self.fitted = True

        calibrated = cal.transform(raw_scores)
        ece, mce, bins = _compute_ece(calibrated, labels, self.n_bins)
        brier = float(np.mean((calibrated - labels) ** 2))

        raw_ece, _, _ = _compute_ece(raw_scores, labels, self.n_bins)
        self._raw_ece = raw_ece
        self._cal_ece = ece

        return CalibrationResult(
            method=self.method,
            n_calibration_samples=n,
            ece=ece,
            mce=mce,
            brier_score=brier,
            n_bins=self.n_bins,
            bins=bins,
        )

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw model scores to calibrated probabilities."""
        if not self.fitted or self._calibrator is None:
            raise RuntimeError("Call fit() before calibrate()")
        scores = np.asarray(raw_scores, dtype=float).ravel()
        return self._calibrator.transform(scores)

    def ece_improvement(self) -> float | None:
        """ECE reduction after calibration (positive = improvement).

        Returns None if fit() has not been called yet.
        """
        if self._raw_ece is None or self._cal_ece is None:
            return None
        return round(self._raw_ece - self._cal_ece, 6)

    @staticmethod
    def is_available() -> bool:
        try:
            from sklearn.isotonic import IsotonicRegression  # noqa: F401

            return True
        except ImportError:
            return False
