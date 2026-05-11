"""Document quality assessment pipeline.

Evaluates scan quality before OCR/classification — rejecting
low-quality inputs prevents silent model failures in production
insurance document pipelines.

All functions are numpy-only (no cv2/scipy/torch) for CI compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QualityMetrics:
    """Composite quality metrics for a scanned document image."""

    blur_score: float  # 1.0 = sharp, 0.0 = blurry
    brightness_score: float  # 1.0 = optimal mid-tone, 0.0 = too dark or too bright
    contrast_score: float  # 1.0 = high contrast, 0.0 = flat/washed-out
    noise_level: float  # 0.0 = clean, 1.0 = very noisy
    skew_angle_deg: float  # estimated tilt in degrees [-45, 45]
    overall_score: float  # weighted composite [0, 1]
    is_acceptable: bool  # True if scan quality is sufficient for classification
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        """Serialise to JSON-safe dict for API responses."""
        return {
            "blur_score": round(self.blur_score, 4),
            "brightness_score": round(self.brightness_score, 4),
            "contrast_score": round(self.contrast_score, 4),
            "noise_level": round(self.noise_level, 4),
            "skew_angle_deg": round(self.skew_angle_deg, 2),
            "overall_score": round(self.overall_score, 4),
            "is_acceptable": self.is_acceptable,
            "rejection_reason": self.rejection_reason,
        }


# ---------------------------------------------------------------------------
# Individual metric estimators
# ---------------------------------------------------------------------------


def _to_gray(pixels: np.ndarray) -> np.ndarray:
    """Convert array to 2-D float grayscale via ITU-R BT.601 coefficients."""
    if pixels.ndim == 3:
        return (0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]).astype(
            float
        )
    return pixels.astype(float)


def estimate_blur(pixels: np.ndarray) -> float:
    """Estimate image sharpness via Laplacian variance proxy.

    Replaces scipy.ndimage.laplace with a 5-point finite-difference stencil.
    Sharp images have high variance in second derivatives; blurry images smooth
    them out.  Variance ~500 → sharp text; ~20 → blurry scan.
    """
    gray = _to_gray(pixels)
    lap = (
        np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
        - 4.0 * gray
    )
    variance = float(np.var(lap))
    return float(np.clip(variance / 500.0, 0.0, 1.0))


def estimate_brightness(pixels: np.ndarray) -> float:
    """Score brightness by proximity to the optimal mid-point (mean ≈ 0.5).

    Penalises both under-exposed (too dark) and over-exposed (too bright)
    scans equally.  Score = 1 − 2|μ − 0.5|, so μ=0.5 → 1.0 and μ=0|1 → 0.0.
    """
    mean = float(np.mean(_to_gray(pixels))) / 255.0
    return float(max(0.0, 1.0 - 2.0 * abs(mean - 0.5)))


def estimate_contrast(pixels: np.ndarray) -> float:
    """Score contrast via normalised standard deviation.

    std=0.25 of normalised [0,1] pixels corresponds to excellent
    text/background separation; anything below 0.05 is a washed-out scan.
    """
    std = float(np.std(_to_gray(pixels))) / 255.0
    return float(np.clip(std / 0.25, 0.0, 1.0))


def estimate_noise(pixels: np.ndarray) -> float:
    """Estimate noise level via mean absolute adjacent-pixel difference.

    Noisy images show high local variation even in homogeneous regions.
    Normalised so that a mean difference of 25/255 per step corresponds to
    a noise level of 1.0.
    """
    gray = _to_gray(pixels)
    h, w = gray.shape
    if h < 2 or w < 2:
        return 0.0
    diff_h = np.abs(np.diff(gray, axis=0))
    diff_v = np.abs(np.diff(gray, axis=1))
    mean_diff = float((diff_h.mean() + diff_v.mean()) / 2.0)
    return float(np.clip(mean_diff / 25.0, 0.0, 1.0))


def estimate_skew(pixels: np.ndarray) -> float:
    """Estimate document skew angle via ink-centroid linear regression.

    Text lines produce a consistent horizontal centroid per row.  A tilted
    document shows a linear trend in centroid-X vs row-Y.  OLS slope → angle.
    Returns degrees in [-45, 45].  Requires ≥ 8 rows/cols to be meaningful.
    """
    gray = _to_gray(pixels)
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0

    threshold = float(np.mean(gray))
    binary = gray < threshold  # ink = dark pixels

    col_idx = np.arange(w, dtype=float)
    row_mass = binary.sum(axis=1).astype(float)
    valid = row_mass > (w * 0.05)  # skip nearly-blank rows

    if valid.sum() < 4:
        return 0.0

    # Weighted centroid X per row
    centroids = np.where(
        valid,
        (binary * col_idx).sum(axis=1) / np.maximum(row_mass, 1.0),
        np.nan,
    )
    mask = valid & ~np.isnan(centroids)
    ys = np.arange(h, dtype=float)[mask]
    xs = centroids[mask]

    if len(ys) < 4:
        return 0.0

    # OLS: centroid_x = a + b * row_y  →  b ≈ tan(skew_angle)
    ys_c = ys - ys.mean()
    xs_c = xs - xs.mean()
    slope = float(np.dot(ys_c, xs_c) / (np.dot(ys_c, ys_c) + 1e-10))
    return float(np.clip(np.degrees(np.arctan(slope)), -45.0, 45.0))


# ---------------------------------------------------------------------------
# Composite quality gate
# ---------------------------------------------------------------------------


def assess_quality(
    pixels: np.ndarray,
    *,
    blur_weight: float = 0.40,
    brightness_weight: float = 0.20,
    contrast_weight: float = 0.25,
    noise_weight: float = 0.15,
    accept_threshold: float = 0.40,
) -> QualityMetrics:
    """Compute composite quality score and decide whether to accept the scan.

    Weights reflect relative impact on downstream OCR/classifier accuracy:
    blur degrades recognition most severely (40%), noise least (15%).
    Scans below ``accept_threshold`` are rejected before classification.
    """
    blur = estimate_blur(pixels)
    brightness = estimate_brightness(pixels)
    contrast = estimate_contrast(pixels)
    noise = estimate_noise(pixels)
    skew = estimate_skew(pixels)

    # Low noise is good: invert noise so all terms are "higher = better"
    overall = float(
        np.clip(
            blur_weight * blur
            + brightness_weight * brightness
            + contrast_weight * contrast
            + noise_weight * (1.0 - noise),
            0.0,
            1.0,
        )
    )
    is_acceptable = overall >= accept_threshold

    rejection_reason: str | None = None
    if not is_acceptable:
        issues: list[str] = []
        if blur < 0.30:
            issues.append(f"blur={blur:.2f}<0.30")
        if brightness < 0.20:
            issues.append(f"brightness={brightness:.2f}<0.20")
        if contrast < 0.15:
            issues.append(f"contrast={contrast:.2f}<0.15")
        if noise > 0.80:
            issues.append(f"noise={noise:.2f}>0.80")
        rejection_reason = (
            "; ".join(issues) if issues else f"overall={overall:.2f}<{accept_threshold}"
        )

    return QualityMetrics(
        blur_score=blur,
        brightness_score=brightness,
        contrast_score=contrast,
        noise_level=noise,
        skew_angle_deg=skew,
        overall_score=overall,
        is_acceptable=is_acceptable,
        rejection_reason=rejection_reason,
    )
