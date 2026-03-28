"""
Профилирование данных / Data profiling module.

Собирает статистику по каждому столбцу DataFrame:
количество значений, пропуски, среднее, стандартное отклонение и т.д.

Collects per-column statistics: count, nulls, mean, std, min, max,
unique values, dtype, and tries to detect the distribution shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy import stats


def profile_column(series: pl.Series) -> dict[str, Any]:
    """
    Профилировать один столбец / Profile a single column.

    Возвращает словарь со статистиками:
    - count, null_count, null_pct
    - unique_count
    - dtype
    - Для числовых: mean, std, min, max, median, q25, q75
    - distribution_type (если числовой)
    """
    total = len(series)
    null_count = series.null_count()
    unique_count = series.n_unique()

    result: dict[str, Any] = {
        "name": series.name,
        "dtype": str(series.dtype),
        "count": total,
        "null_count": null_count,
        "null_pct": round(null_count / total * 100, 2) if total > 0 else 0.0,
        "unique_count": unique_count,
    }

    # Числовые столбцы — считаем полную статистику
    if series.dtype.is_numeric():
        clean = series.drop_nulls()
        if len(clean) > 0:
            result["mean"] = round(float(clean.mean()), 4)  # type: ignore[arg-type]
            result["std"] = round(float(clean.std()), 4)  # type: ignore[arg-type]
            result["min"] = float(clean.min())  # type: ignore[arg-type]
            result["max"] = float(clean.max())  # type: ignore[arg-type]
            result["median"] = float(clean.median())  # type: ignore[arg-type]
            result["q25"] = float(clean.quantile(0.25))  # type: ignore[arg-type]
            result["q75"] = float(clean.quantile(0.75))  # type: ignore[arg-type]
            result["distribution"] = detect_distribution_type(clean)
        else:
            result["mean"] = None
            result["std"] = None
            result["min"] = None
            result["max"] = None
            result["median"] = None
            result["distribution"] = "empty"

    # Строковые / категориальные — топ значений
    elif series.dtype == pl.Utf8 or series.dtype == pl.Categorical:
        clean = series.drop_nulls()
        if len(clean) > 0:
            vc = clean.value_counts().sort("count", descending=True).head(10)
            result["top_values"] = {
                str(row[series.name]): int(row["count"])
                for row in vc.iter_rows(named=True)
            }
        result["distribution"] = "categorical"

    return result


def profile_dataframe(df: pl.DataFrame) -> dict[str, Any]:
    """
    Профилировать весь DataFrame / Profile the entire DataFrame.

    Возвращает:
    - overview: общая информация (строки, столбцы, размер в памяти)
    - columns: словарь {column_name: profile_dict}
    """
    overview = {
        "row_count": df.height,
        "column_count": df.width,
        "column_names": df.columns,
        "estimated_memory_mb": round(df.estimated_size("mb"), 2),
    }

    columns: dict[str, dict[str, Any]] = {}
    for col_name in df.columns:
        columns[col_name] = profile_column(df[col_name])

    return {
        "overview": overview,
        "columns": columns,
    }


def detect_distribution_type(series: pl.Series) -> str:
    """
    Определить тип распределения числового столбца.
    Detect distribution type of a numeric column.

    Применяем несколько статистических тестов:
    - Шапиро-Уилк для нормальности
    - Проверка лог-нормальности (логарифм + Шапиро-Уилк)
    - Если мало уникальных значений — "categorical"
    - Иначе — "other"

    We use Shapiro-Wilk for normality, log-transform + Shapiro-Wilk
    for lognormality, and fall back to "other" if nothing fits.
    """
    clean = series.drop_nulls()
    values = clean.to_numpy().astype(float)

    if len(values) < 8:
        return "too_few_values"

    n_unique = len(np.unique(values))

    # Мало уникальных — скорее категориальный признак, закодированный числами
    if n_unique <= 10:
        return "categorical"

    # Ограничиваем выборку для Shapiro-Wilk (макс 5000)
    sample = values[:5000] if len(values) > 5000 else values
    alpha = 0.05

    # Тест на нормальность / Normality test
    try:
        _, p_normal = stats.shapiro(sample)
        if p_normal > alpha:
            return "normal"
    except ValueError:
        pass

    # Тест на лог-нормальность / Lognormality test
    if np.all(values > 0):
        try:
            _, p_lognormal = stats.shapiro(np.log(sample))
            if p_lognormal > alpha:
                return "lognormal"
        except ValueError:
            pass

    # Тест на равномерность / Uniformity test
    try:
        _, p_uniform = stats.kstest(
            sample,
            "uniform",
            args=(sample.min(), sample.max() - sample.min()),
        )
        if p_uniform > alpha:
            return "uniform"
    except ValueError:
        pass

    return "other"
