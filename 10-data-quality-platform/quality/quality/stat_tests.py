"""
Расширенная батарея статистических тестов дрейфа / Extended drift test battery.

Дополняет базовый PSI+KS из drift.py тремя методами:
  - Wasserstein distance (Earth Mover's Distance) для непрерывных признаков
  - Jensen-Shannon divergence для дискретных / категориальных признаков
  - Chi-squared тест для категориальных признаков

Together these four methods (PSI / KS / Wasserstein / JS+χ²) give full
coverage of the 2026 MLOps monitoring trifecta: magnitude, rank, and
categorical distribution changes.

References:
  - Wasserstein: Villani 2008 "Optimal Transport"
  - JS divergence: Lin 1991 IEEE Trans. Inf. Theory 37(1)
  - Chi-squared: Pearson 1900 Philosophical Magazine 50(302)
  - MLOps drift battery: Evidently AI v0.5+, WhyLogs 1.3
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Wasserstein distance (Earth Mover's Distance)
# ---------------------------------------------------------------------------


def wasserstein_distance(
    reference: np.ndarray | list,
    current: np.ndarray | list,
) -> float:
    """
    Расстояние Вассерштейна 1-го порядка (Earth Mover's Distance).
    First-order Wasserstein distance between two continuous samples.

    Интерпретация / Interpretation:
      0.0        — идентичные распределения
      0.0–0.05   — незначительный сдвиг (в нормализованной шкале)
      0.05–0.15  — умеренный сдвиг
      > 0.15     — значительный дрейф

    Вычисляется через сортировку перцентилей (O(n log n)) — без scipy.
    Computed via sorted-percentile integral (O(n log n)), no external deps.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Равномерная сетка CDF через квантили для сравнения разного объёма выборок
    n_quantiles = 1000
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    ref_quantiles = np.quantile(ref, quantiles)
    cur_quantiles = np.quantile(cur, quantiles)

    # W1 = ∫|F_ref(x) - F_cur(x)| dx ≈ mean |Q_ref(p) - Q_cur(p)|
    distance = float(np.mean(np.abs(ref_quantiles - cur_quantiles)))
    return round(distance, 6)


def wasserstein_severity(distance: float, scale: float = 1.0) -> str:
    """
    Уровень тревоги по расстоянию Вассерштейна / Severity from Wasserstein distance.

    scale — характерный масштаб признака (std reference).
    Нормализуем расстояние на масштаб, чтобы пороги были инвариантны к единицам.
    Normalise by feature scale so thresholds are unit-invariant.
    """
    if scale <= 0:
        scale = 1.0
    normalised = distance / scale
    if normalised < 0.05:
        return "ok"
    if normalised < 0.15:
        return "moderate"
    return "critical"


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------


def js_divergence(
    reference: np.ndarray | list,
    current: np.ndarray | list,
    bins: int = 20,
) -> float:
    """
    Дивергенция Дженсена-Шеннона / Jensen-Shannon divergence.

    Симметричная версия KL-дивергенции, ограниченная [0, 1] (в бит-нормировке).
    Symmetric, bounded [0, 1] version of KL divergence (bit normalisation).

    JS = (KL(P||M) + KL(Q||M)) / 2,  M = (P + Q) / 2

    Хорошо работает для категориальных и бинаризованных непрерывных признаков.
    Works well for categorical and binned continuous features.

    Интерпретация / Interpretation:
      < 0.05  — ok
      0.05–0.1 — moderate
      > 0.1   — critical
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Бинаризация: единые границы по объединённому диапазону
    combined_min = min(ref.min(), cur.min())
    combined_max = max(ref.max(), cur.max())

    if combined_min == combined_max:
        return 0.0

    breakpoints = np.linspace(combined_min, combined_max, bins + 1)
    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    # Нормировка вероятностей с защитой от нулей (Laplace smoothing)
    eps = 1e-10
    ref_p = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
    cur_q = (cur_counts + eps) / (cur_counts.sum() + eps * bins)

    m = (ref_p + cur_q) / 2.0

    # KL(P||M) = Σ P·log(P/M)
    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    js = (_kl(ref_p, m) + _kl(cur_q, m)) / 2.0
    # Привести к [0,1]: ln-based JS ∈ [0, ln2]; делим на ln(2)
    js_normalised = js / np.log(2)
    return round(float(np.clip(js_normalised, 0.0, 1.0)), 6)


def js_severity(js_value: float) -> str:
    """Уровень тревоги по JS-дивергенции / Severity from JS divergence."""
    if js_value < 0.05:
        return "ok"
    if js_value < 0.10:
        return "moderate"
    return "critical"


# ---------------------------------------------------------------------------
# Chi-squared test for categorical features
# ---------------------------------------------------------------------------


def chi2_test(
    reference: np.ndarray | list,
    current: np.ndarray | list,
) -> dict[str, float]:
    """
    Тест хи-квадрат для категориальных признаков / Chi-squared test for categorical features.

    Сравнивает наблюдаемые частоты current с ожидаемыми из reference.
    Compares observed current counts to expected counts from reference.

    Возвращает / Returns:
      statistic: χ² statistic
      p_value: p-value (< 0.05 → статистически значимый сдвиг)
      dof: степени свободы / degrees of freedom

    Graceful degradation без scipy: использует таблицу квантилей χ².
    Работает с numpy-only.
    """
    ref = np.asarray(reference).flatten()
    cur = np.asarray(current).flatten()

    # Определяем общий словарь категорий
    all_cats = sorted(set(ref.tolist()) | set(cur.tolist()))
    if len(all_cats) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "dof": 0}

    ref_counts = np.array([np.sum(ref == c) for c in all_cats], dtype=float)
    cur_counts = np.array([np.sum(cur == c) for c in all_cats], dtype=float)

    # Ожидаемые частоты = ref_pct * n_current
    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()

    if ref_total == 0 or cur_total == 0:
        return {"statistic": 0.0, "p_value": 1.0, "dof": len(all_cats) - 1}

    expected = (ref_counts / ref_total) * cur_total

    # χ² = Σ (O - E)² / E, пропускаем бины с E < 1 (нестабильно)
    mask = expected >= 1.0
    if mask.sum() < 2:
        return {"statistic": 0.0, "p_value": 1.0, "dof": int(mask.sum()) - 1}

    obs = cur_counts[mask]
    exp = expected[mask]
    chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
    dof = int(mask.sum()) - 1

    # p-value через scipy если доступно, иначе консервативная аппроксимация
    try:
        from scipy import stats as scipy_stats

        p_value = float(scipy_stats.chi2.sf(chi2_stat, dof))
    except ImportError:
        # Консервативная аппроксимация: χ²/dof > 3.84 (α=0.05 для dof=1) → drift
        threshold = 3.84 * dof
        p_value = 0.01 if chi2_stat > threshold else 0.5

    return {
        "statistic": round(chi2_stat, 6),
        "p_value": round(p_value, 6),
        "dof": dof,
    }


# ---------------------------------------------------------------------------
# Расширенный комплексный тест дрейфа
# ---------------------------------------------------------------------------


def extended_drift_test(
    reference: np.ndarray | list,
    current: np.ndarray | list,
    feature_type: str = "continuous",
    bins: int = 20,
) -> dict[str, Any]:
    """
    Батарея всех доступных тестов для одного признака / All-test battery for one feature.

    feature_type:
      "continuous" → Wasserstein + JS divergence
      "categorical" → Chi-squared + JS divergence
      "auto"       → auto-detect (few unique values → categorical)

    Возвращает:
      tests: результаты каждого теста
      drift_detected: True если хотя бы один тест сигнализирует о дрейфе
      severity: max severity ("ok" / "moderate" / "critical")
      confidence: доля тестов, зафиксировавших дрейф
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if feature_type == "auto":
        n_unique = len(set(ref.tolist()))
        feature_type = "categorical" if n_unique <= 20 else "continuous"

    results: dict[str, Any] = {}
    severities: list[str] = []

    if feature_type == "continuous":
        w_dist = wasserstein_distance(ref, cur)
        ref_std = float(np.std(ref)) if len(ref) > 1 else 1.0
        w_sev = wasserstein_severity(w_dist, scale=ref_std)
        results["wasserstein"] = {"distance": w_dist, "severity": w_sev}
        severities.append(w_sev)

        js_val = js_divergence(ref, cur, bins=bins)
        js_sev = js_severity(js_val)
        results["js_divergence"] = {"value": js_val, "severity": js_sev}
        severities.append(js_sev)

    else:  # categorical
        chi2 = chi2_test(ref.astype(int), cur.astype(int))
        chi2_sev = "drift" if chi2["p_value"] < 0.05 else "ok"
        results["chi2"] = {**chi2, "severity": chi2_sev}
        severities.append(chi2_sev)

        js_val = js_divergence(ref, cur, bins=bins)
        js_sev = js_severity(js_val)
        results["js_divergence"] = {"value": js_val, "severity": js_sev}
        severities.append(js_sev)

    # Агрегированный вывод
    severity_order = {"ok": 0, "moderate": 1, "drift": 1, "critical": 2}
    max_sev = max(severities, key=lambda s: severity_order.get(s, 0))
    drift_flags = [s != "ok" for s in severities]
    confidence = sum(drift_flags) / len(drift_flags) if drift_flags else 0.0

    return {
        "feature_type": feature_type,
        "tests": results,
        "drift_detected": any(drift_flags),
        "severity": max_sev,
        "confidence": round(confidence, 3),
    }


def batch_extended_drift(
    reference_data: dict[str, list],
    current_data: dict[str, list],
    feature_types: dict[str, str] | None = None,
    bins: int = 20,
) -> dict[str, Any]:
    """
    Батарея тестов по всем признакам / Battery across all features.

    reference_data, current_data: {column_name: [values...]}
    feature_types: опциональный словарь "col" → "continuous"|"categorical"|"auto"

    Возвращает сводный отчёт + результаты по каждому столбцу.
    Returns a summary report + per-column results.
    """
    if feature_types is None:
        feature_types = {}

    columns = sorted(set(reference_data) & set(current_data))
    column_results: list[dict[str, Any]] = []
    n_drift = 0
    critical_cols: list[str] = []

    for col in columns:
        ref_vals = reference_data[col]
        cur_vals = current_data[col]
        ftype = feature_types.get(col, "auto")

        try:
            result = extended_drift_test(ref_vals, cur_vals, feature_type=ftype, bins=bins)
            result["column"] = col
            column_results.append(result)
            if result["drift_detected"]:
                n_drift += 1
            if result["severity"] == "critical":
                critical_cols.append(col)
        except Exception as exc:
            column_results.append(
                {
                    "column": col,
                    "status": "error",
                    "reason": str(exc),
                    "drift_detected": False,
                    "severity": "ok",
                }
            )

    return {
        "columns_checked": len(columns),
        "columns_with_drift": n_drift,
        "critical_columns": critical_cols,
        "overall_drift": n_drift > 0,
        "details": column_results,
    }
