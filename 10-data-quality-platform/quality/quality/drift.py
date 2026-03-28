"""
Детекция дрифта распределений / Distribution drift detection.

Модуль сравнивает эталонное (reference) и текущее (current) распределение
числовых признаков и сигнализирует, если распределение "уехало".

Compares reference vs. current distributions using PSI and KS test.
Returns a structured drift report with p-values and severity alerts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats


def psi(
    reference: np.ndarray | pl.Series,
    current: np.ndarray | pl.Series,
    bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).

    PSI показывает, насколько сильно текущее распределение отклонилось
    от эталонного. Интерпретация:
      < 0.1  — изменений нет
      0.1–0.25 — умеренный сдвиг, стоит обратить внимание
      > 0.25 — значительный дрифт, нужна реакция

    PSI < 0.1 means no shift, 0.1-0.25 moderate, > 0.25 significant drift.
    """
    ref = _to_numpy(reference)
    cur = _to_numpy(current)

    # Границы бинов строим по reference — это наш "эталон"
    breakpoints = np.linspace(
        min(ref.min(), cur.min()),
        max(ref.max(), cur.max()),
        bins + 1,
    )

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    # Добавляем маленькое число, чтобы не делить на ноль
    eps = 1e-8
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi_value, 6)


def ks_test(
    reference: np.ndarray | pl.Series,
    current: np.ndarray | pl.Series,
) -> dict[str, float]:
    """
    Двухвыборочный тест Колмогорова-Смирнова.
    Two-sample Kolmogorov-Smirnov test.

    Возвращает статистику KS и p-value.
    p-value < 0.05 означает, что распределения статистически различаются.
    """
    ref = _to_numpy(reference)
    cur = _to_numpy(current)

    statistic, p_value = stats.ks_2samp(ref, cur)
    return {
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
    }


def detect_drift(
    ref_df: pl.DataFrame,
    curr_df: pl.DataFrame,
    columns: list[str] | None = None,
    psi_bins: int = 10,
    psi_threshold: float = 0.1,
    ks_alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Запустить проверку дрифта по всем числовым столбцам.
    Run drift detection on specified (or all numeric) columns.

    Возвращает отчёт с результатами PSI и KS для каждого столбца,
    а также общий статус: есть дрифт или нет.

    Returns a report with PSI + KS results per column and an overall status.
    """
    # Если столбцы не указаны — берём все числовые, которые есть в обоих df
    if columns is None:
        ref_numeric = {c for c in ref_df.columns if ref_df[c].dtype.is_numeric()}
        cur_numeric = {c for c in curr_df.columns if curr_df[c].dtype.is_numeric()}
        columns = sorted(ref_numeric & cur_numeric)

    column_reports: list[dict[str, Any]] = []
    has_drift = False

    for col in columns:
        ref_series = ref_df[col].drop_nulls()
        cur_series = curr_df[col].drop_nulls()

        if len(ref_series) == 0 or len(cur_series) == 0:
            column_reports.append(
                {
                    "column": col,
                    "status": "skipped",
                    "reason": "Недостаточно данных / Not enough non-null data",
                }
            )
            continue

        psi_value = psi(ref_series, cur_series, bins=psi_bins)
        ks_result = ks_test(ref_series, cur_series)

        # Определяем уровень тревоги / Determine alert level
        psi_alert = _psi_severity(psi_value, psi_threshold)
        ks_alert = "drift" if ks_result["p_value"] < ks_alpha else "ok"

        # Если хотя бы один тест сработал — считаем, что дрифт есть
        col_has_drift = psi_alert != "ok" or ks_alert != "ok"
        if col_has_drift:
            has_drift = True

        column_reports.append(
            {
                "column": col,
                "psi": psi_value,
                "psi_alert": psi_alert,
                "ks_statistic": ks_result["statistic"],
                "ks_p_value": ks_result["p_value"],
                "ks_alert": ks_alert,
                "drift_detected": col_has_drift,
            }
        )

    return {
        "drift_detected": has_drift,
        "columns_checked": len(columns),
        "columns_with_drift": sum(1 for r in column_reports if r.get("drift_detected", False)),
        "details": column_reports,
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции / Helpers
# ---------------------------------------------------------------------------


def _to_numpy(data: np.ndarray | pl.Series) -> np.ndarray:
    """Привести к numpy-массиву / Convert to numpy array."""
    if isinstance(data, pl.Series):
        return data.to_numpy().astype(float)
    return np.asarray(data, dtype=float)


def _psi_severity(psi_value: float, threshold: float = 0.1) -> str:
    """
    Уровень тревоги по PSI / PSI severity level.
    ok < threshold < moderate < threshold*2.5 < critical
    """
    if psi_value < threshold:
        return "ok"
    if psi_value < threshold * 2.5:
        return "moderate"
    return "critical"
