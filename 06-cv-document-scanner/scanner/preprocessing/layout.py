"""Document layout segmentation via horizontal/vertical projection profiles.

Projection profiles count ink pixels per row/column, revealing text zone
boundaries through valleys (gaps) in the density curve.  Used to locate
header/body/footer zones for targeted field extraction in insurance pipelines —
the insurer needs to find signature blocks, amounts, and dates reliably.

All functions are numpy-only (no cv2/PIL) for CI compatibility.
Reference: O'Gorman 1993 Document Spectral Analysis, Kise et al. 1998 ICDAR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RegionType(str):
    """Document region labels — kept as str constants for JSON serialisation."""

    HEADER = "header"
    BODY = "body"
    FOOTER = "footer"
    MARGIN = "margin"
    BLANK = "blank"


@dataclass
class LayoutRegion:
    """Detected text zone with bounding box and ink statistics."""

    region_type: str  # one of RegionType constants
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    ink_density: float  # fraction of dark pixels in region [0, 1]

    def height(self) -> int:
        return self.row_end - self.row_start

    def width(self) -> int:
        return self.col_end - self.col_start

    def to_dict(self) -> dict:
        return {
            "region_type": self.region_type,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "height": self.height(),
            "width": self.width(),
            "ink_density": round(self.ink_density, 4),
        }


@dataclass
class LayoutResult:
    """Full layout analysis: detected regions plus high-level structural flags."""

    regions: list[LayoutRegion]
    n_text_zones: int
    has_header: bool
    has_footer: bool
    is_two_column: bool
    h_projection: list[float]  # unsmoothed per-row ink density for debug/visualisation

    def to_dict(self) -> dict:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "n_text_zones": self.n_text_zones,
            "has_header": self.has_header,
            "has_footer": self.has_footer,
            "is_two_column": self.is_two_column,
        }


@dataclass
class LayoutConfig:
    """Tunable parameters for layout segmentation."""

    ink_threshold: float = 128.0  # pixel value below which a pixel counts as ink
    gap_threshold: float = 0.02  # row ink density below this → treated as a gap
    min_zone_rows: int = 3  # discard text zones shorter than this
    smooth_window: int = 3  # box-filter width applied to projection before zone detection


# ---------------------------------------------------------------------------
# Core analysis functions (public for unit testing)
# ---------------------------------------------------------------------------


def _to_gray_float(pixels: np.ndarray) -> np.ndarray:
    """Convert to 2-D float grayscale (ITU-R BT.601 coefficients)."""
    if pixels.ndim == 3:
        return (0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]).astype(
            float
        )
    return pixels.astype(float)


def compute_horizontal_projection(gray: np.ndarray, ink_threshold: float = 128.0) -> np.ndarray:
    """Compute row-wise ink density: fraction of dark pixels in each row.

    Returns shape ``(H,)`` with values in [0, 1].
    """
    ink = gray < ink_threshold
    return ink.sum(axis=1).astype(float) / gray.shape[1]


def compute_vertical_projection(gray: np.ndarray, ink_threshold: float = 128.0) -> np.ndarray:
    """Compute column-wise ink density: fraction of dark pixels in each column.

    Returns shape ``(W,)`` with values in [0, 1].
    """
    ink = gray < ink_threshold
    return ink.sum(axis=0).astype(float) / gray.shape[0]


def find_gaps(projection: np.ndarray, gap_threshold: float = 0.02) -> list[tuple[int, int]]:
    """Return contiguous runs where ink density < gap_threshold.

    Each run is a ``(start, end)`` pair with end exclusive.
    """
    is_gap = projection < gap_threshold
    gaps: list[tuple[int, int]] = []
    in_gap = False
    start = 0
    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap = True
            start = i
        elif not g and in_gap:
            in_gap = False
            gaps.append((start, i))
    if in_gap:
        gaps.append((start, len(projection)))
    return gaps


def find_text_zones(
    projection: np.ndarray,
    gap_threshold: float = 0.02,
    min_rows: int = 3,
) -> list[tuple[int, int]]:
    """Return contiguous text zones (ink density >= gap_threshold, >= min_rows rows)."""
    is_text = projection >= gap_threshold
    h = len(projection)
    zones: list[tuple[int, int]] = []
    in_zone = False
    start = 0
    for i in range(h):
        if is_text[i] and not in_zone:
            in_zone = True
            start = i
        elif not is_text[i] and in_zone:
            in_zone = False
            if i - start >= min_rows:
                zones.append((start, i))
    if in_zone and h - start >= min_rows:
        zones.append((start, h))
    return zones


def _detect_two_column(
    gray: np.ndarray,
    row_start: int,
    row_end: int,
    ink_threshold: float = 128.0,
    min_gap_width: int = 5,
) -> bool:
    """Return True when the body region contains a significant vertical gap.

    A contiguous run of ``min_gap_width`` columns with near-zero ink density
    in the middle third of the region indicates a two-column layout.
    """
    body = gray[row_start:row_end, :]
    h, w = body.shape
    if h < 10 or w < 20:
        return False
    v_proj = compute_vertical_projection(body, ink_threshold)
    # Restrict search to middle third to avoid classifying page margins as columns
    third = max(w // 3, 1)
    middle_proj = v_proj[third : 2 * third]
    is_col_gap = middle_proj < 0.03
    max_gap = cur_gap = 0
    for g in is_col_gap:
        if g:
            cur_gap += 1
            if cur_gap > max_gap:
                max_gap = cur_gap
        else:
            cur_gap = 0
    return max_gap >= min_gap_width


def _label_zones(
    zones: list[tuple[int, int]],
    height: int,
    width: int,
    gray: np.ndarray,
    ink_threshold: float,
) -> list[LayoutRegion]:
    """Assign HEADER / BODY / FOOTER labels based on vertical position.

    A first zone that ends before 35% of the page is a header; a last zone
    that starts after 75% of the page is a footer.  Everything else is body.
    Single-zone documents are always labelled BODY — no context to distinguish.
    """
    regions = []
    n = len(zones)
    for i, (r0, r1) in enumerate(zones):
        ink_density = float((gray[r0:r1, :] < ink_threshold).mean())
        if n >= 2 and i == 0 and r1 / height < 0.35:
            rtype = RegionType.HEADER
        elif n >= 2 and i == n - 1 and r0 / height > 0.75:
            rtype = RegionType.FOOTER
        else:
            rtype = RegionType.BODY
        regions.append(
            LayoutRegion(
                region_type=rtype,
                row_start=r0,
                row_end=r1,
                col_start=0,
                col_end=width,
                ink_density=ink_density,
            )
        )
    return regions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def segment_layout(
    pixels: np.ndarray,
    config: LayoutConfig | None = None,
) -> LayoutResult:
    """Segment a document image into header / body / footer zones.

    Uses projection profile analysis — a classical document analysis technique
    without any ML or external dependencies.  Suitable for pre-processing
    before OCR or targeted field extraction in insurance pipelines.

    Args:
        pixels: H×W (grayscale) or H×W×3 (RGB) uint8/float image array.
        config: optional tuning parameters; defaults are calibrated for A4 scans.

    Returns:
        LayoutResult with detected regions, structural flags, and raw projection.
    """
    cfg = config or LayoutConfig()
    gray = _to_gray_float(pixels)
    h, w = gray.shape

    _empty = LayoutResult(
        regions=[],
        n_text_zones=0,
        has_header=False,
        has_footer=False,
        is_two_column=False,
        h_projection=[],
    )
    if h < 4 or w < 4:
        return _empty

    h_proj_raw = compute_horizontal_projection(gray, cfg.ink_threshold)

    # Smoothing merges near-adjacent gaps from noisy scans without erasing real gaps
    if cfg.smooth_window > 1:
        kernel = np.ones(cfg.smooth_window) / cfg.smooth_window
        h_proj = np.convolve(h_proj_raw, kernel, mode="same")
    else:
        h_proj = h_proj_raw

    zones = find_text_zones(h_proj, cfg.gap_threshold, cfg.min_zone_rows)
    if not zones:
        return _empty

    regions = _label_zones(zones, h, w, gray, cfg.ink_threshold)

    # Two-column check on the largest body zone
    body_regions = [r for r in regions if r.region_type == RegionType.BODY]
    is_two_col = False
    if body_regions:
        largest = max(body_regions, key=lambda r: r.height())
        is_two_col = _detect_two_column(gray, largest.row_start, largest.row_end, cfg.ink_threshold)

    return LayoutResult(
        regions=regions,
        n_text_zones=len(zones),
        has_header=any(r.region_type == RegionType.HEADER for r in regions),
        has_footer=any(r.region_type == RegionType.FOOTER for r in regions),
        is_two_column=is_two_col,
        h_projection=h_proj_raw.tolist(),
    )
