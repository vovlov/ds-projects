"""Morphological document cleaning for scanned insurance forms.

Classic morphological pipeline (Haralick et al. 1987):
  binarise → erode → dilate → remove ruling lines → despeckle

Ruling lines (horizontal separators in form templates) and salt-and-pepper
scanner noise are the two main artifacts that confuse downstream classifiers
and OCR engines.  All operations are numpy-only — no cv2/scipy/PIL.

References:
  Haralick et al. 1987 "Image Analysis Using Mathematical Morphology"
    IEEE TPAMI 9(4):532–550.
  Otsu 1979 "A Threshold Selection Method from Gray-Level Histograms"
    IEEE Trans. SMC 9(1):62–66.
  Dougherty 1992 "Mathematical Morphology in Image Processing" CRC Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MorphConfig:
    """Pipeline configuration for morphological document cleaning."""

    kernel_size: int = 3  # structuring element side length (must be odd)
    remove_h_lines: bool = True  # remove horizontal ruling lines
    remove_v_lines: bool = False  # remove vertical ruling lines
    h_line_min_width_frac: float = 0.6  # row ink fraction threshold for h-line
    v_line_min_height_frac: float = 0.6  # col ink fraction threshold for v-line
    despeckle: bool = True  # remove isolated noise pixels
    despeckle_kernel: int = 3  # kernel for despeckle opening pass


@dataclass
class CleaningStats:
    """Pixel-level statistics from the cleaning pipeline."""

    otsu_threshold: int  # binarisation threshold [0, 255]
    original_ink_pixels: int  # dark pixels before cleaning
    cleaned_ink_pixels: int  # dark pixels after cleaning
    lines_removed: int  # rows/cols zeroed as ruling lines
    noise_pixels_removed: int  # pixels eliminated by despeckle

    def noise_reduction_pct(self) -> float:
        """Fraction of original ink pixels removed as noise/lines."""
        if self.original_ink_pixels == 0:
            return 0.0
        removed = self.original_ink_pixels - self.cleaned_ink_pixels
        return round(removed / self.original_ink_pixels, 4)

    def to_dict(self) -> dict:
        return {
            "otsu_threshold": self.otsu_threshold,
            "original_ink_pixels": self.original_ink_pixels,
            "cleaned_ink_pixels": self.cleaned_ink_pixels,
            "lines_removed": self.lines_removed,
            "noise_pixels_removed": self.noise_pixels_removed,
            "noise_reduction_pct": self.noise_reduction_pct(),
        }


@dataclass
class MorphResult:
    """Cleaned binary document image plus cleaning statistics."""

    cleaned_pixels: list[list[int]]  # binary 0/255, white background
    stats: CleaningStats
    config: MorphConfig = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "cleaned_pixels": self.cleaned_pixels,
            "height": len(self.cleaned_pixels),
            "width": len(self.cleaned_pixels[0]) if self.cleaned_pixels else 0,
            "stats": self.stats.to_dict(),
        }


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def otsu_threshold(arr: np.ndarray) -> int:
    """Compute optimal global binarisation threshold via Otsu's method.

    Maximises inter-class variance between foreground (ink) and background
    (paper) pixels.  O(256) time — no iteration over all pixels.
    """
    gray = arr.astype(np.float32)
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    total = gray.size

    sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
    sum_b, w_b = 0.0, 0
    max_var, threshold = 0.0, 0

    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t * hist[t])
        mu_b = sum_b / w_b
        mu_f = (sum_total - sum_b) / w_f
        var = float(w_b) * float(w_f) * (mu_b - mu_f) ** 2
        if var > max_var:
            max_var, threshold = var, t

    return int(threshold)


def binarize(arr: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Convert grayscale [0-255] to ink-mask (1=dark, 0=light).

    Uses Otsu's threshold when *threshold* is None.  Returns uint8 array.
    """
    gray = arr.astype(np.uint8)
    if gray.ndim == 3:
        # Convert colour to grayscale via BT.601 luminance coefficients
        gray = (0.299 * gray[:, :, 0] + 0.587 * gray[:, :, 1] + 0.114 * gray[:, :, 2]).astype(
            np.uint8
        )
    t = threshold if threshold is not None else otsu_threshold(gray)
    # Pixels at or below threshold are darker than paper → ink
    ink = (gray.astype(np.int32) <= t).astype(np.uint8)
    return ink


def erode(ink: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological erosion: shrink ink regions by the structuring element.

    A pixel survives erosion only if all pixels in its kernel window are ink.
    Removes thin single-pixel bridges and isolated salt noise.
    """
    k = kernel_size
    pad = k // 2
    padded = np.pad(ink, pad, mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (k, k))
    return windows.min(axis=(-2, -1)).astype(np.uint8)


def dilate(ink: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological dilation: expand ink regions by the structuring element.

    A pixel becomes ink if any pixel in its kernel window is ink.
    Fills small holes and reconnects broken character strokes.
    """
    k = kernel_size
    pad = k // 2
    padded = np.pad(ink, pad, mode="constant", constant_values=0)
    windows = sliding_window_view(padded, (k, k))
    return windows.max(axis=(-2, -1)).astype(np.uint8)


def opening(ink: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological opening: erosion → dilation.

    Removes thin noise and isolated small ink blobs while preserving
    the shape of larger objects (character bodies, thick strokes).
    """
    return dilate(erode(ink, kernel_size), kernel_size)


def closing(ink: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological closing: dilation → erosion.

    Fills small holes and gaps inside ink regions (broken character strokes
    caused by light ink or worn toner).
    """
    return erode(dilate(ink, kernel_size), kernel_size)


# ---------------------------------------------------------------------------
# Artifact-specific cleaners
# ---------------------------------------------------------------------------


def remove_horizontal_lines(
    ink: np.ndarray,
    min_width_frac: float = 0.6,
) -> tuple[np.ndarray, int]:
    """Zero out rows that are ruling lines (horizontal form separators).

    A row is classified as a ruling line when the fraction of ink pixels
    exceeds *min_width_frac* of the image width.  Ruling lines in insurance
    forms are typically 70–100 % wide; text rows are sparse by comparison.

    Returns (cleaned_ink, n_lines_removed).
    """
    result = ink.copy()
    n_cols = ink.shape[1]
    if n_cols == 0:
        return result, 0
    n_removed = 0
    row_sums = ink.sum(axis=1)
    mask = row_sums >= min_width_frac * n_cols
    result[mask] = 0
    n_removed = int(mask.sum())
    return result, n_removed


def remove_vertical_lines(
    ink: np.ndarray,
    min_height_frac: float = 0.6,
) -> tuple[np.ndarray, int]:
    """Zero out columns that are vertical ruling lines.

    Mirror of *remove_horizontal_lines* for table column separators.
    Returns (cleaned_ink, n_lines_removed).
    """
    result = ink.copy()
    n_rows = ink.shape[0]
    if n_rows == 0:
        return result, 0
    col_sums = ink.sum(axis=0)
    mask = col_sums >= min_height_frac * n_rows
    result[:, mask] = 0
    n_removed = int(mask.sum())
    return result, n_removed


def despeckle_ink(ink: np.ndarray, kernel_size: int = 3) -> tuple[np.ndarray, int]:
    """Remove isolated noise pixels via morphological opening.

    Opening with *kernel_size*×*kernel_size* structuring element eliminates
    connected ink blobs smaller than the kernel — typical scanner salt noise.
    Larger blobs (character strokes) survive because erosion+dilation is
    approximately shape-preserving for objects larger than the kernel.

    Returns (cleaned_ink, n_pixels_removed).
    """
    opened = opening(ink, kernel_size)
    removed = int(np.sum(ink) - np.sum(opened))
    return opened, max(0, removed)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def clean_document(arr: np.ndarray, config: MorphConfig | None = None) -> MorphResult:
    """Run the full morphological cleaning pipeline on a grayscale document.

    Pipeline stages:
      1. Otsu binarisation (grayscale → ink mask)
      2. Horizontal/vertical ruling line removal (if enabled)
      3. Morphological opening despeckle (if enabled)

    *arr* must be a 2-D (grayscale) or 3-D (RGB) uint8 array.
    Returns *MorphResult* with cleaned binary image (0/255) and stats.
    """
    cfg = config or MorphConfig()

    if arr.size == 0:
        raise ValueError("Input pixel array must not be empty")

    t = otsu_threshold(arr)
    ink = binarize(arr, threshold=t)
    original_ink = int(ink.sum())

    lines_removed = 0

    if cfg.remove_h_lines:
        ink, h_removed = remove_horizontal_lines(ink, min_width_frac=cfg.h_line_min_width_frac)
        lines_removed += h_removed

    if cfg.remove_v_lines:
        ink, v_removed = remove_vertical_lines(ink, min_height_frac=cfg.v_line_min_height_frac)
        lines_removed += v_removed

    noise_removed = 0
    if cfg.despeckle:
        ink, noise_removed = despeckle_ink(ink, kernel_size=cfg.despeckle_kernel)

    cleaned_ink = int(ink.sum())

    # Convert ink mask back to 0/255 image (white background, dark ink)
    output = ((1 - ink) * 255).astype(np.uint8)

    stats = CleaningStats(
        otsu_threshold=t,
        original_ink_pixels=original_ink,
        cleaned_ink_pixels=cleaned_ink,
        lines_removed=lines_removed,
        noise_pixels_removed=noise_removed,
    )

    return MorphResult(
        cleaned_pixels=output.tolist(),
        stats=stats,
        config=cfg,
    )
