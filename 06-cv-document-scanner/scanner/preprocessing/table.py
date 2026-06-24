"""Table structure detection via projection-profile line analysis.

Identifies ruled-grid tables in scanned documents by locating horizontal and
vertical ink-dense lines (cell separators), then mapping their intersections
to a regular cell grid.

Suitable for insurance forms, bank statements, and contract tables — any
document with printed ruling lines.  Pure numpy, no cv2/PIL required.

References:
    Wang & Hu 1996 "A Machine Learning Based Approach for Table Detection"
      ACM SIGIR.
    Kieninger & Dengel 1998 "A Paper-to-HTML Table Converting System" IAPR DAS.
    Itonori 1994 ICDAR "Table structure recognition based on textblock
      arrangement and ruled-line position".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TableConfig:
    """Tuning parameters for table structure detection."""

    ink_threshold: float = 128.0  # pixel value below which a pixel is ink
    line_density_h: float = 0.70  # min row ink fraction to classify as h-separator
    line_density_v: float = 0.70  # min col ink fraction to classify as v-separator
    min_n_rows: int = 2  # minimum cell rows required to declare a table
    min_n_cols: int = 2  # minimum cell cols required to declare a table


@dataclass
class TableCell:
    """Bounding box of a single table cell in the image coordinate system."""

    row_idx: int  # 0-based row index within the table
    col_idx: int  # 0-based column index within the table
    row_start: int  # inclusive top pixel row
    row_end: int  # exclusive bottom pixel row (cell body ends before next separator)
    col_start: int  # inclusive left pixel column
    col_end: int  # exclusive right pixel column

    def area(self) -> int:
        """Pixel area of this cell."""
        return max(0, self.row_end - self.row_start) * max(0, self.col_end - self.col_start)

    def to_dict(self) -> dict:
        return {
            "row_idx": self.row_idx,
            "col_idx": self.col_idx,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "area": self.area(),
        }


@dataclass
class TableStructure:
    """Detected table grid: separator positions and cell bounding boxes."""

    n_rows: int  # number of cell rows (= len(row_separators) - 1)
    n_cols: int  # number of cell columns (= len(col_separators) - 1)
    cells: list[TableCell]
    has_grid: bool  # True when min_n_rows and min_n_cols thresholds are met
    confidence: float  # [0, 1] — inter-separator spacing regularity
    row_separators: list[int]  # pixel row indices of detected horizontal lines
    col_separators: list[int]  # pixel col indices of detected vertical lines
    config: TableConfig = field(repr=False, default_factory=TableConfig)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_cells": len(self.cells),
            "has_grid": self.has_grid,
            "confidence": round(self.confidence, 4),
            "row_separators": self.row_separators,
            "col_separators": self.col_separators,
            "cells": [c.to_dict() for c in self.cells],
        }


# ---------------------------------------------------------------------------
# Core algorithms (public for unit testing)
# ---------------------------------------------------------------------------


def _to_gray(pixels: np.ndarray) -> np.ndarray:
    """Convert to 2-D float grayscale using ITU-R BT.601 coefficients."""
    if pixels.ndim == 3:
        return (0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]).astype(
            float
        )
    return pixels.astype(float)


def _find_separator_lines(projection: np.ndarray, density_threshold: float) -> list[int]:
    """Find contiguous high-density runs and return their centre pixel indices.

    Each uninterrupted run of pixels with density >= *density_threshold* is
    collapsed to its midpoint — the representative coordinate of one separator
    line.  This handles multi-pixel-thick ruling lines from scanner aliasing.

    Args:
        projection: 1-D array of per-row or per-column ink density in [0, 1].
        density_threshold: minimum density to classify a position as a separator.

    Returns:
        Sorted list of separator midpoint indices.
    """
    is_line = projection >= density_threshold
    n = len(projection)
    separators: list[int] = []
    in_run = False
    start = 0

    for i in range(n):
        if is_line[i] and not in_run:
            in_run = True
            start = i
        elif not is_line[i] and in_run:
            in_run = False
            separators.append((start + i - 1) // 2)

    if in_run:
        separators.append((start + n - 1) // 2)

    return separators


def _spacing_regularity(separators: list[int]) -> float:
    """Score [0, 1] for how evenly spaced the separator lines are.

    A perfectly regular grid has equal inter-separator gaps (coefficient of
    variation = 0 → score = 1.0).  Irregular spacing reduces the score
    exponentially: score = exp(-CoV).

    Returns 0.0 when fewer than 3 separators exist (need ≥ 2 gaps to measure
    regularity).
    """
    if len(separators) < 3:
        return 0.0
    gaps = np.diff(separators).astype(float)
    mean_gap = float(gaps.mean())
    if mean_gap < 1e-9:
        return 0.0
    cv = float(gaps.std()) / mean_gap  # coefficient of variation
    return float(np.exp(-cv))  # 1.0 for perfectly even spacing


def _build_cells(row_seps: list[int], col_seps: list[int]) -> list[TableCell]:
    """Cross-product of row and column separator positions → cell bounding boxes.

    Cell (i, j) spans the image region between row_seps[i]–row_seps[i+1] and
    col_seps[j]–col_seps[j+1].  Cells are ordered row-major (left to right,
    top to bottom).
    """
    cells: list[TableCell] = []
    for i in range(len(row_seps) - 1):
        for j in range(len(col_seps) - 1):
            cells.append(
                TableCell(
                    row_idx=i,
                    col_idx=j,
                    row_start=row_seps[i],
                    row_end=row_seps[i + 1],
                    col_start=col_seps[j],
                    col_end=col_seps[j + 1],
                )
            )
    return cells


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_table(
    pixels: np.ndarray,
    config: TableConfig | None = None,
) -> TableStructure:
    """Detect a ruled-grid table in a document image.

    Algorithm:
      1. Convert to grayscale, compute per-row and per-column ink density.
      2. Rows with density >= line_density_h → horizontal separator candidates.
      3. Columns with density >= line_density_v → vertical separator candidates.
      4. Adjacent high-density rows/cols are merged into a single separator
         (handles multi-pixel-thick printing and scan aliasing).
      5. Grid is declared when at least min_n_rows cell rows and min_n_cols
         cell columns are formed by the detected separators.
      6. Confidence = mean(row_spacing_regularity, col_spacing_regularity).

    Args:
        pixels: H×W (grayscale) or H×W×3 (RGB) uint8/float image array.
        config: optional tuning parameters; defaults calibrated for A4 forms.

    Returns:
        TableStructure with separator positions, cell bounding boxes, and a
        confidence score reflecting how regular the grid spacing is.
    """
    cfg = config or TableConfig()

    _empty = TableStructure(
        n_rows=0,
        n_cols=0,
        cells=[],
        has_grid=False,
        confidence=0.0,
        row_separators=[],
        col_separators=[],
        config=cfg,
    )

    gray = _to_gray(pixels)
    h, w = gray.shape
    if h < 4 or w < 4:
        return _empty

    ink = gray < cfg.ink_threshold  # True = ink pixel

    # Per-row ink density: fraction of dark pixels in each row
    h_proj = ink.sum(axis=1) / w  # shape (H,)
    # Per-col ink density: fraction of dark pixels in each column
    v_proj = ink.sum(axis=0) / h  # shape (W,)

    row_seps = _find_separator_lines(h_proj, cfg.line_density_h)
    col_seps = _find_separator_lines(v_proj, cfg.line_density_v)

    n_rows = max(0, len(row_seps) - 1)
    n_cols = max(0, len(col_seps) - 1)
    has_grid = n_rows >= cfg.min_n_rows and n_cols >= cfg.min_n_cols

    if not has_grid:
        return TableStructure(
            n_rows=n_rows,
            n_cols=n_cols,
            cells=[],
            has_grid=False,
            confidence=0.0,
            row_separators=row_seps,
            col_separators=col_seps,
            config=cfg,
        )

    cells = _build_cells(row_seps, col_seps)

    # Confidence: average regularity of row and col spacing
    row_reg = _spacing_regularity(row_seps)
    col_reg = _spacing_regularity(col_seps)
    confidence = (row_reg + col_reg) / 2.0

    return TableStructure(
        n_rows=n_rows,
        n_cols=n_cols,
        cells=cells,
        has_grid=True,
        confidence=confidence,
        row_separators=row_seps,
        col_separators=col_seps,
        config=cfg,
    )
