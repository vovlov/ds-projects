"""
Корреляционный анализ для платформы качества данных.
Correlation analysis for the data quality platform.

Поддерживает три метода:
- Pearson: линейная корреляция для числовых столбцов
- Spearman: ранговая корреляция (устойчива к outliers, монотонные зависимости)
- Cramér's V: ассоциация между категориальными столбцами

Supports three methods:
- Pearson:   linear correlation for numeric columns
- Spearman:  rank correlation (robust to outliers, monotone relationships)
- Cramér's V: association strength between categorical columns

Каждый метод возвращает CorrelationMatrix с флагированием подозрительных пар
(утечка данных, случайная мультиколлинеарность, алиасированные признаки).

Each method returns a CorrelationMatrix flagging suspicious pairs
(data leakage, accidental multicollinearity, aliased features).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

# Порог «подозрительной» корреляции — признак возможной утечки данных
# Threshold for "suspicious" correlation — possible data leakage signal
_LEAKAGE_THRESHOLD = 0.99

# Порог «сильной» корреляции — мультиколлинеарность, стоит проверить
# Threshold for "strong" correlation — multicollinearity worth noting
_STRONG_THRESHOLD = 0.95


@dataclass
class CorrelationPair:
    """
    Пара столбцов с корреляционным коэффициентом.
    Column pair with a correlation coefficient.
    """

    col_a: str
    col_b: str
    method: str
    coefficient: float
    p_value: float | None  # None для Cramér's V (нет аналитического p-value)
    flag: str  # "leakage" | "strong" | "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "col_a": self.col_a,
            "col_b": self.col_b,
            "method": self.method,
            "coefficient": round(self.coefficient, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "flag": self.flag,
        }


@dataclass
class CorrelationMatrix:
    """
    Полная корреляционная матрица с метаданными и подозрительными парами.
    Full correlation matrix with metadata and suspicious pairs.

    matrix[i][j] = None, если пара столбцов несовместима с методом
    (например, нечисловые для Pearson/Spearman).
    matrix[i][j] = None when the column pair is incompatible with the method
    (e.g. non-numeric for Pearson/Spearman).
    """

    method: str
    columns: list[str]
    matrix: list[list[float | None]]
    suspicious_pairs: list[CorrelationPair]
    n_total_pairs: int
    n_suspicious: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "columns": self.columns,
            "matrix": self.matrix,
            "suspicious_pairs": [p.to_dict() for p in self.suspicious_pairs],
            "n_total_pairs": self.n_total_pairs,
            "n_suspicious": self.n_suspicious,
        }


def _flag(coefficient: float) -> str:
    """Определить флаг по абсолютному значению коэффициента."""
    abs_c = abs(coefficient)
    if abs_c >= _LEAKAGE_THRESHOLD:
        return "leakage"
    if abs_c >= _STRONG_THRESHOLD:
        return "strong"
    return "ok"


def _suspicious_pairs_from_matrix(
    cols: list[str],
    matrix: list[list[float | None]],
    method: str,
    p_matrix: list[list[float | None]] | None = None,
) -> list[CorrelationPair]:
    """Извлечь подозрительные пары из заполненной матрицы (верхний треугольник)."""
    pairs: list[CorrelationPair] = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            c = matrix[i][j]
            if c is None:
                continue
            flag = _flag(c)
            if flag != "ok":
                p = p_matrix[i][j] if p_matrix else None
                pairs.append(
                    CorrelationPair(
                        col_a=cols[i],
                        col_b=cols[j],
                        method=method,
                        coefficient=round(c, 6),
                        p_value=round(p, 6) if p is not None else None,
                        flag=flag,
                    )
                )
    # Сортируем по убыванию |коэффициент| — самые подозрительные первыми
    pairs.sort(key=lambda x: abs(x.coefficient), reverse=True)
    return pairs


def pearson_matrix(df: pl.DataFrame) -> CorrelationMatrix:
    """
    Матрица корреляций Пирсона для числовых столбцов.
    Pearson correlation matrix for numeric columns.

    Использует scipy.stats.pearsonr для каждой пары — получаем p-value.
    Uses scipy.stats.pearsonr per pair to obtain p-values.
    """
    from scipy import stats

    num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    n = len(num_cols)

    # Инициализируем матрицы None — заполним только допустимые пары
    mat: list[list[float | None]] = [[None] * n for _ in range(n)]
    p_mat: list[list[float | None]] = [[None] * n for _ in range(n)]

    for i in range(n):
        mat[i][i] = 1.0
        p_mat[i][i] = 0.0
        a = df[num_cols[i]].drop_nulls().to_numpy().astype(float)
        for j in range(i + 1, n):
            b = df[num_cols[j]].drop_nulls().to_numpy().astype(float)
            min_len = min(len(a), len(b))
            if min_len < 3:
                continue
            try:
                r, p = stats.pearsonr(a[:min_len], b[:min_len])
                if math.isfinite(r):
                    mat[i][j] = mat[j][i] = round(r, 6)
                    p_mat[i][j] = p_mat[j][i] = round(p, 6)
            except Exception:
                pass

    suspicious = _suspicious_pairs_from_matrix(num_cols, mat, "pearson", p_mat)
    total = n * (n - 1) // 2

    return CorrelationMatrix(
        method="pearson",
        columns=num_cols,
        matrix=mat,
        suspicious_pairs=suspicious,
        n_total_pairs=total,
        n_suspicious=len(suspicious),
    )


def spearman_matrix(df: pl.DataFrame) -> CorrelationMatrix:
    """
    Матрица ранговых корреляций Спирмена для числовых столбцов.
    Spearman rank correlation matrix for numeric columns.

    Более устойчива к выбросам, чем Пирсон; выявляет монотонные зависимости.
    More robust to outliers than Pearson; captures monotone relationships.
    """
    from scipy import stats

    num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    n = len(num_cols)

    mat: list[list[float | None]] = [[None] * n for _ in range(n)]
    p_mat: list[list[float | None]] = [[None] * n for _ in range(n)]

    for i in range(n):
        mat[i][i] = 1.0
        p_mat[i][i] = 0.0
        a = df[num_cols[i]].drop_nulls().to_numpy().astype(float)
        for j in range(i + 1, n):
            b = df[num_cols[j]].drop_nulls().to_numpy().astype(float)
            min_len = min(len(a), len(b))
            if min_len < 3:
                continue
            try:
                r, p = stats.spearmanr(a[:min_len], b[:min_len])
                if math.isfinite(r):
                    mat[i][j] = mat[j][i] = round(r, 6)
                    p_mat[i][j] = p_mat[j][i] = round(p, 6)
            except Exception:
                pass

    suspicious = _suspicious_pairs_from_matrix(num_cols, mat, "spearman", p_mat)
    total = n * (n - 1) // 2

    return CorrelationMatrix(
        method="spearman",
        columns=num_cols,
        matrix=mat,
        suspicious_pairs=suspicious,
        n_total_pairs=total,
        n_suspicious=len(suspicious),
    )


def _cramers_v(contingency: np.ndarray) -> float:
    """
    Вычислить Cramér's V из contingency table.
    Compute Cramér's V from a contingency table.

    Cramér's V ∈ [0, 1]: 0 = нет ассоциации, 1 = полная ассоциация.
    Cramér's V ∈ [0, 1]: 0 = no association, 1 = perfect association.

    Bias-corrected version (Bergsma & Wicher 2013) для маленьких выборок:
    тилда-версия делает V не завышенной при случайных таблицах.
    Bias-corrected (Bergsma & Wicher 2013) to avoid upward bias in small samples.
    """
    from scipy.stats import chi2_contingency

    n = contingency.sum()
    if n == 0:
        return 0.0

    try:
        chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    except Exception:
        return 0.0

    phi2 = chi2 / n
    r, k = contingency.shape

    # Bias correction: тилда phi², r̃, k̃
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)

    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2_corr / denom))


def cramers_v_matrix(df: pl.DataFrame) -> CorrelationMatrix:
    """
    Матрица Cramér's V для категориальных / строковых столбцов.
    Cramér's V matrix for categorical / string columns.

    Также включает бинарные числовые (<=10 уникальных значений) — они часто
    кодируют категории. Включает bias-corrected версию (Bergsma & Wicher 2013).
    Also includes low-cardinality numeric (<=10 uniques) — often encoded categories.
    """
    cat_cols = [
        c
        for c in df.columns
        if df[c].dtype in (pl.Utf8, pl.String, pl.Categorical)
        or (df[c].dtype.is_numeric() and df[c].n_unique() <= 10)
    ]
    n = len(cat_cols)

    mat: list[list[float | None]] = [[None] * n for _ in range(n)]

    for i in range(n):
        mat[i][i] = 1.0
        col_a = df[cat_cols[i]].drop_nulls().cast(pl.String)
        for j in range(i + 1, n):
            col_b = df[cat_cols[j]].drop_nulls().cast(pl.String)
            min_len = min(len(col_a), len(col_b))
            if min_len < 5:
                continue
            try:
                a_vals = col_a[:min_len].to_numpy()
                b_vals = col_b[:min_len].to_numpy()
                a_cats, a_idx = np.unique(a_vals, return_inverse=True)
                b_cats, b_idx = np.unique(b_vals, return_inverse=True)
                contingency = np.zeros((len(a_cats), len(b_cats)), dtype=int)
                np.add.at(contingency, (a_idx, b_idx), 1)
                v = _cramers_v(contingency)
                if math.isfinite(v):
                    mat[i][j] = mat[j][i] = round(v, 6)
            except Exception:
                pass

    suspicious = _suspicious_pairs_from_matrix(cat_cols, mat, "cramers_v", p_matrix=None)
    total = n * (n - 1) // 2

    return CorrelationMatrix(
        method="cramers_v",
        columns=cat_cols,
        matrix=mat,
        suspicious_pairs=suspicious,
        n_total_pairs=total,
        n_suspicious=len(suspicious),
    )


def correlation_report(
    df: pl.DataFrame,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """
    Сводный отчёт корреляций для всех запрошенных методов.
    Unified correlation report for all requested methods.

    Args:
        df: входной DataFrame
        methods: список методов — "pearson", "spearman", "cramers_v".
                 По умолчанию ["pearson", "spearman", "cramers_v"].

    Returns:
        dict с ключом per method + "summary": {n_suspicious_total, top_suspicious_pairs}.
    """
    if methods is None:
        methods = ["pearson", "spearman", "cramers_v"]

    dispatch = {
        "pearson": pearson_matrix,
        "spearman": spearman_matrix,
        "cramers_v": cramers_v_matrix,
    }

    results: dict[str, Any] = {}
    all_suspicious: list[CorrelationPair] = []

    for m in methods:
        if m not in dispatch:
            raise ValueError(f"Unknown method '{m}'. Choose from: {list(dispatch)}")
        cm = dispatch[m](df)
        results[m] = cm.to_dict()
        all_suspicious.extend(cm.suspicious_pairs)

    # Дедуплицируем по паре (col_a, col_b) — берём наихудший флаг
    seen: dict[tuple[str, str], CorrelationPair] = {}
    for pair in all_suspicious:
        key = (min(pair.col_a, pair.col_b), max(pair.col_a, pair.col_b))
        if key not in seen or abs(pair.coefficient) > abs(seen[key].coefficient):
            seen[key] = pair

    top = sorted(seen.values(), key=lambda x: abs(x.coefficient), reverse=True)[:10]

    results["summary"] = {
        "methods_run": methods,
        "n_suspicious_total": len(seen),
        "top_suspicious_pairs": [p.to_dict() for p in top],
        "leakage_risk": any(p.flag == "leakage" for p in seen.values()),
    }

    return results
