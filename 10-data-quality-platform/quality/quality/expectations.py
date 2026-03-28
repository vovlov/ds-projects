"""
Проверки качества данных / Data quality expectations.

Вдохновлено Great Expectations, но реализация своя — легковесная,
на Polars, без лишних зависимостей.

Inspired by Great Expectations but built from scratch on top of Polars.
Each check returns a standardized result dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# Результат проверки / Check result structure
# ---------------------------------------------------------------------------


def _result(
    check: str,
    column: str,
    passed: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Стандартный формат результата / Standard result format."""
    return {
        "check": check,
        "column": column,
        "passed": passed,
        "details": details or {},
    }


# ---------------------------------------------------------------------------
# Отдельные проверки / Individual expectations
# ---------------------------------------------------------------------------


def expect_not_null(df: pl.DataFrame, column: str) -> dict[str, Any]:
    """
    Проверить, что в столбце нет пропусков / Assert zero nulls in column.
    """
    if column not in df.columns:
        return _result(
            "expect_not_null",
            column,
            False,
            {"error": f"Столбец '{column}' не найден / Column not found"},
        )

    null_count = df[column].null_count()
    return _result(
        "expect_not_null",
        column,
        passed=(null_count == 0),
        details={"null_count": null_count, "total": df.height},
    )


def expect_unique(df: pl.DataFrame, column: str) -> dict[str, Any]:
    """
    Проверить уникальность значений / Assert all values are unique.
    """
    if column not in df.columns:
        return _result(
            "expect_unique",
            column,
            False,
            {"error": f"Столбец '{column}' не найден / Column not found"},
        )

    n_unique = df[column].n_unique()
    # null тоже считается одним уникальным значением в Polars,
    # поэтому корректируем — вычитаем 1, если есть null
    has_nulls = df[column].null_count() > 0
    effective_unique = n_unique - (1 if has_nulls else 0)
    non_null_count = df.height - df[column].null_count()

    return _result(
        "expect_unique",
        column,
        passed=(effective_unique == non_null_count),
        details={
            "unique_count": effective_unique,
            "non_null_count": non_null_count,
            "duplicate_count": non_null_count - effective_unique,
        },
    )


def expect_values_in_range(
    df: pl.DataFrame,
    column: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
) -> dict[str, Any]:
    """
    Проверить, что все значения лежат в заданном диапазоне.
    Assert all values fall within [min_value, max_value].
    None означает "без ограничения" / None means unbounded.
    """
    if column not in df.columns:
        return _result(
            "expect_values_in_range",
            column,
            False,
            {"error": f"Столбец '{column}' не найден / Column not found"},
        )

    series = df[column].drop_nulls()
    out_of_range = 0

    if min_value is not None:
        out_of_range += series.filter(series < min_value).len()
    if max_value is not None:
        out_of_range += series.filter(series > max_value).len()

    return _result(
        "expect_values_in_range",
        column,
        passed=(out_of_range == 0),
        details={
            "out_of_range_count": out_of_range,
            "min_value": min_value,
            "max_value": max_value,
            "actual_min": float(series.min()) if len(series) > 0 else None,
            "actual_max": float(series.max()) if len(series) > 0 else None,
        },
    )


def expect_column_exists(df: pl.DataFrame, column: str) -> dict[str, Any]:
    """
    Проверить, что столбец существует в DataFrame / Assert column exists.
    """
    exists = column in df.columns
    return _result(
        "expect_column_exists",
        column,
        passed=exists,
        details={"available_columns": df.columns},
    )


def expect_values_in_set(
    df: pl.DataFrame,
    column: str,
    allowed_values: list[Any],
) -> dict[str, Any]:
    """
    Проверить, что все значения входят в допустимое множество.
    Assert all non-null values belong to the allowed set.
    """
    if column not in df.columns:
        return _result(
            "expect_values_in_set",
            column,
            False,
            {"error": f"Столбец '{column}' не найден / Column not found"},
        )

    series = df[column].drop_nulls()
    invalid = series.filter(~series.is_in(allowed_values))
    invalid_examples = invalid.head(10).to_list()

    return _result(
        "expect_values_in_set",
        column,
        passed=(len(invalid) == 0),
        details={
            "invalid_count": len(invalid),
            "invalid_examples": invalid_examples,
            "allowed_values": allowed_values,
        },
    )


# ---------------------------------------------------------------------------
# Реестр проверок / Check registry
# ---------------------------------------------------------------------------

# Маппинг имя_проверки -> функция, чтобы вызывать по строке из YAML
CHECK_REGISTRY: dict[str, Any] = {
    "expect_not_null": expect_not_null,
    "expect_unique": expect_unique,
    "expect_values_in_range": expect_values_in_range,
    "expect_column_exists": expect_column_exists,
    "expect_values_in_set": expect_values_in_set,
}


# ---------------------------------------------------------------------------
# Запуск набора проверок / Run a suite from config
# ---------------------------------------------------------------------------


def run_suite(
    df: pl.DataFrame,
    suite_config: dict[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """
    Запустить набор проверок из конфигурации (dict или путь к YAML).
    Run all expectations defined in a suite config (dict or YAML path).

    Формат YAML / YAML format:
    ```yaml
    suite_name: "my_checks"
    expectations:
      - check: expect_not_null
        column: user_id
      - check: expect_values_in_range
        column: age
        kwargs:
          min_value: 0
          max_value: 150
    ```
    """
    if isinstance(suite_config, (str, Path)):
        config_path = Path(suite_config)
        if not config_path.exists():
            raise FileNotFoundError(f"Конфиг не найден / Config not found: {config_path}")
        with open(config_path) as f:
            suite_config = yaml.safe_load(f)

    expectations = suite_config.get("expectations", [])  # type: ignore[union-attr]
    results: list[dict[str, Any]] = []

    for spec in expectations:
        check_name = spec["check"]
        column = spec["column"]
        kwargs = spec.get("kwargs", {})

        check_fn = CHECK_REGISTRY.get(check_name)
        if check_fn is None:
            results.append(
                _result(
                    check_name,
                    column,
                    False,
                    {"error": f"Неизвестная проверка / Unknown check: {check_name}"},
                )
            )
            continue

        result = check_fn(df, column, **kwargs)
        results.append(result)

    return results
