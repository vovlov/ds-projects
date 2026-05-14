"""Qini-кривая и AUUC метрики для оценки uplift-моделей.

Qini curve and AUUC metrics for uplift model evaluation.

Qini-кривая измеряет инкрементальный выигрыш от таргетирования
клиентов в порядке убывания предсказанного uplift (CATE).

The Qini curve measures the incremental gain from targeting customers
in descending order of predicted uplift (CATE).

Интерпретация:
- Random curve: прямая от (0,0) до (n, q_random) — baseline
- Perfect model: резко выходит вверх (все persuadables в начале)
- Our model: между ними; чем выше — тем лучше

Interpretation:
- Random curve: line from (0,0) to (n, q_random) — baseline
- Perfect model: steep rise (all persuadables ranked first)
- Our model: between them; higher is better

Источники:
- Radcliffe & Surry 2011 "Real-World Uplift Modelling with Significance-Based
  Uplift Trees" (Stochastic Solutions Working Paper)
- Devriendt et al. 2020 "uplift: An R package for the uplift modeling"
- Gutierrez & Gerardy 2017 JMLR Workshop
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QiniResult:
    """Результат вычисления Qini-кривой.

    Qini curve computation result.
    """

    targeting_rates: list[float]
    """Доля охваченных клиентов (x-ось), от 0 до 1."""

    qini_gains: list[float]
    """Инкрементальный выигрыш при каждом targeting_rate."""

    random_gains: list[float]
    """Ожидаемый выигрыш при случайном выборе (baseline)."""

    auuc: float
    """Area Under Uplift Curve — площадь между Qini и random curves."""

    qini_coefficient: float
    """Нормированный AUUC: AUUC(model) / AUUC(perfect model).
    Normalized AUUC: AUUC(model) / AUUC(perfect model).
    Диапазон примерно [-1, 1]; > 0 лучше случайного.
    """

    n_samples: int
    n_treated: int
    n_control: int


def compute_qini_curve(
    y_true: np.ndarray,
    y_treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 20,
) -> QiniResult:
    """Вычислить Qini-кривую и AUUC для uplift-модели.

    Compute the Qini curve and AUUC for an uplift model.

    Логика (Radcliffe & Surry 2011):
    Сортируем клиентов по убыванию predicted CATE. При таргетировании
    top-t доли клиентов: Qini gain = #{Y=1, T=1 in top-t} − #{Y=1, T=1 in top-t}
    скорректированный на пропорцию treated (без этой коррекции кривая смещена).

    Algorithm (Radcliffe & Surry 2011):
    Sort customers by descending predicted CATE. When targeting top-t fraction:
    Qini gain = #{Y=1, T=1 in top-t} - n_t_top * (n_c1 / n_c)
    where n_c1 = treated positives in full set, n_c = total treated.

    Args:
        y_true: (n,) истинные исходы {0, 1}. 1 = позитивный ответ.
        y_treatment: (n,) treatment indicator {0=control, 1=treated}
        uplift_scores: (n,) предсказанный CATE — сортировка по этому полю
        n_bins: количество точек на кривой

    Returns:
        QiniResult с кривой, AUUC и Qini-коэффициентом.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_treatment = np.asarray(y_treatment, dtype=int)
    uplift_scores = np.asarray(uplift_scores, dtype=float)

    n = len(y_true)
    if n == 0:
        raise ValueError("Empty arrays")

    n_treated_total = int(y_treatment.sum())
    n_control_total = n - n_treated_total
    n_treated_positive = int((y_true[y_treatment == 1]).sum())

    # Сортируем по убыванию CATE
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = y_treatment[order]

    targeting_rates = []
    qini_gains = []
    random_gains = []

    # Вычисляем кривую в n_bins точках
    steps = np.linspace(0, n, n_bins + 1, dtype=int)

    for k in steps:
        if k == 0:
            targeting_rates.append(0.0)
            qini_gains.append(0.0)
            random_gains.append(0.0)
            continue

        top_k = min(k, n)
        y_top = y_sorted[:top_k]
        t_top = t_sorted[:top_k]

        treated_pos = int((y_top[t_top == 1]).sum())
        n_treated_top = int((t_top == 1).sum())

        # Qini gain = treated positives in top-k − expected treated positives
        # at this rate from random targeting
        # expected = n_treated_top * (overall treated positive rate)
        if n_treated_total > 0:
            expected = n_treated_top * n_treated_positive / n_treated_total
        else:
            expected = 0.0

        targeting_rates.append(top_k / n)
        qini_gains.append(float(treated_pos - expected))
        # Random: каждый шаг по прямой от 0 до max_gain
        random_gains.append(0.0)  # random = 0 по определению Qini

    # AUUC = площадь под кривой (trapezoid)
    rates_arr = np.array(targeting_rates)
    gains_arr = np.array(qini_gains)
    auuc = float(np.trapezoid(gains_arr, rates_arr))

    # Perfect model AUUC: сначала таргетируем всех persuadables (treated+positive)
    # затем остальных. Perfect = n_treated_positive / n (нормировка на 1)
    # Используем упрощённую нормировку на максимально достижимый AUUC
    perfect_auuc = _compute_perfect_auuc(y_true, y_treatment)
    qini_coeff = auuc / perfect_auuc if abs(perfect_auuc) > 1e-8 else 0.0

    return QiniResult(
        targeting_rates=targeting_rates,
        qini_gains=qini_gains,
        random_gains=random_gains,
        auuc=round(auuc, 6),
        qini_coefficient=round(float(np.clip(qini_coeff, -2.0, 2.0)), 4),
        n_samples=n,
        n_treated=n_treated_total,
        n_control=n_control_total,
    )


def compute_auuc(
    y_true: np.ndarray,
    y_treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Вычислить AUUC (Area Under Uplift Curve).

    Compute AUUC (Area Under Uplift Curve).

    Args:
        y_true: (n,) истинные исходы {0, 1}
        y_treatment: (n,) treatment {0, 1}
        uplift_scores: (n,) predicted CATE
        n_bins: точек на кривой

    Returns:
        AUUC как скалярное число. Выше 0 = лучше случайного.
    """
    return compute_qini_curve(y_true, y_treatment, uplift_scores, n_bins).auuc


def qini_coefficient(
    y_true: np.ndarray,
    y_treatment: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Вычислить нормированный Qini-коэффициент (≈ нормированный AUUC).

    Compute normalized Qini coefficient (≈ normalized AUUC).

    Диапазон: [-1, 1]; > 0 означает лучше случайного таргетирования.
    Range: [-1, 1]; > 0 means better than random targeting.
    """
    return compute_qini_curve(y_true, y_treatment, uplift_scores, n_bins).qini_coefficient


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_perfect_auuc(y_true: np.ndarray, y_treatment: np.ndarray) -> float:
    """AUUC идеального классификатора (upper bound для нормировки).

    AUUC of perfect classifier (upper bound for normalization).

    Идеальная модель сначала ставит всех persuadables (T=1, Y=1),
    затем sure things (T=0, Y=0), затем остальных.

    Perfect model ranks persuadables (T=1, Y=1) first,
    then sure things (T=0, Y=0), then others.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    n_treated_total = int(y_treatment.sum())
    n_treated_positive = int((y_true[y_treatment == 1]).sum())

    if n_treated_total == 0 or n_treated_positive == 0:
        return 0.0

    # Идеальный порядок: persuadables (Y=1, T=1) сначала
    perfect_score = np.where((y_true == 1) & (y_treatment == 1), 2.0, 1.0)
    perfect_score = np.where(y_treatment == 0, 0.5, perfect_score)

    n_bins = 20
    steps = np.linspace(0, n, n_bins + 1, dtype=int)
    order = np.argsort(-perfect_score)
    y_s = y_true[order]
    t_s = y_treatment[order]

    rates = [0.0]
    gains = [0.0]
    for k in steps[1:]:
        top_k = min(k, n)
        y_top = y_s[:top_k]
        t_top = t_s[:top_k]
        treated_pos = int((y_top[t_top == 1]).sum())
        n_treated_top = int((t_top == 1).sum())
        expected = n_treated_top * n_treated_positive / n_treated_total
        rates.append(top_k / n)
        gains.append(float(treated_pos - expected))

    return float(np.trapezoid(np.array(gains), np.array(rates)))
