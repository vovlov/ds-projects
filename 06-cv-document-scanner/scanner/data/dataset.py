"""
Synthetic document dataset for classification demo.

Real insurance documents can't be shipped in a public repo, so we generate
feature vectors that mimic what a preprocessing pipeline would extract from
scanned pages: aspect ratio, average brightness, text density (fraction of
pixels belonging to OCR-detected characters), edge density (Canny output),
and file size.  Distributions are chosen to be separable enough for a
tree-based baseline while still requiring a model to learn the boundaries.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

# five document classes an insurance company deals with daily
DOC_TYPES: list[str] = [
    "receipt",
    "id_card",
    "medical_report",
    "invoice",
    "contract",
]

# per-class feature distributions: (mean, std) for each feature
# tuned by hand so classes overlap a bit but are mostly separable
_CLASS_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "receipt": {
        "aspect_ratio": (0.35, 0.06),  # narrow thermal-paper rolls
        "brightness": (0.78, 0.08),  # white background, high brightness
        "text_density": (0.45, 0.10),  # lots of small text
        "edge_density": (0.30, 0.07),  # moderate edges from text lines
        "file_size_kb": (120.0, 40.0),  # small scans
    },
    "id_card": {
        "aspect_ratio": (1.58, 0.05),  # credit-card shaped, landscape
        "brightness": (0.55, 0.10),  # photos / holograms darken it
        "text_density": (0.25, 0.06),  # sparse text fields
        "edge_density": (0.50, 0.08),  # lots of edges from photo + borders
        "file_size_kb": (250.0, 60.0),  # colour scan, heavier
    },
    "medical_report": {
        "aspect_ratio": (0.77, 0.04),  # A4 portrait
        "brightness": (0.82, 0.06),  # clean printed page
        "text_density": (0.60, 0.08),  # dense paragraphs
        "edge_density": (0.20, 0.05),  # mostly text, few graphical edges
        "file_size_kb": (350.0, 100.0),  # multi-page PDFs rendered to image
    },
    "invoice": {
        "aspect_ratio": (0.75, 0.05),  # A4-ish, portrait
        "brightness": (0.80, 0.07),  # white background
        "text_density": (0.40, 0.09),  # tables + whitespace
        "edge_density": (0.35, 0.06),  # table grid lines add edges
        "file_size_kb": (200.0, 70.0),  # moderate
    },
    "contract": {
        "aspect_ratio": (0.76, 0.03),  # A4 portrait, very consistent
        "brightness": (0.85, 0.04),  # clean laser-printed pages
        "text_density": (0.65, 0.07),  # wall of legal text
        "edge_density": (0.15, 0.04),  # almost no graphics
        "file_size_kb": (500.0, 150.0),  # long documents
    },
}

FEATURE_COLS: list[str] = [
    "aspect_ratio",
    "brightness",
    "text_density",
    "edge_density",
    "file_size_kb",
]


def generate_synthetic_documents(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Create *n* synthetic document feature rows, balanced across classes.

    Returns a Polars DataFrame with columns for each feature plus `doc_type`.
    """
    rng = np.random.default_rng(seed)
    samples_per_class = n // len(DOC_TYPES)

    rows: dict[str, list] = {col: [] for col in FEATURE_COLS}
    rows["doc_type"] = []

    for doc_type in DOC_TYPES:
        profile = _CLASS_PROFILES[doc_type]
        for col in FEATURE_COLS:
            mean, std = profile[col]
            values = rng.normal(mean, std, size=samples_per_class)
            # clamp to physically plausible ranges
            if col == "aspect_ratio":
                values = np.clip(values, 0.1, 3.0)
            elif col in ("brightness", "text_density", "edge_density"):
                values = np.clip(values, 0.0, 1.0)
            elif col == "file_size_kb":
                values = np.clip(values, 10.0, 2000.0)
            rows[col].extend(values.tolist())
        rows["doc_type"].extend([doc_type] * samples_per_class)

    return pl.DataFrame(rows)


def get_feature_matrix(
    data: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Extract numpy arrays ready for sklearn.

    Returns (X, y_encoded, label_encoder) so callers can decode predictions
    back to human-readable class names.
    """
    X = data.select(FEATURE_COLS).to_numpy().astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(data["doc_type"].to_list())
    return X, y, le
