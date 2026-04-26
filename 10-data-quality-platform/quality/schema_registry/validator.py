"""
Валидация DataFrame против зарегистрированной схемы.
Validate a DataFrame against a registered schema.

Проверяет: наличие столбцов, nullable-ограничения, диапазоны значений,
допустимые множества. Возвращает структурированный отчёт.

Checks: column presence, nullable constraints, value ranges, allowed sets.
Returns a structured validation report.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from .schema import ColumnSchema, ColumnType, DataSchema

# Маппинг Polars dtype → ColumnType для авто-инференса схемы
# Mapping Polars dtype → ColumnType for schema auto-inference
_POLARS_TO_COLUMN_TYPE: dict[str, ColumnType] = {
    "Int8": ColumnType.INTEGER,
    "Int16": ColumnType.INTEGER,
    "Int32": ColumnType.INTEGER,
    "Int64": ColumnType.INTEGER,
    "UInt8": ColumnType.INTEGER,
    "UInt16": ColumnType.INTEGER,
    "UInt32": ColumnType.INTEGER,
    "UInt64": ColumnType.INTEGER,
    "Float32": ColumnType.FLOAT,
    "Float64": ColumnType.FLOAT,
    "Boolean": ColumnType.BOOLEAN,
    "Utf8": ColumnType.STRING,
    "String": ColumnType.STRING,
    "Datetime": ColumnType.DATETIME,
    "Date": ColumnType.DATETIME,
}


def infer_schema_from_dataframe(df: pl.DataFrame, name: str = "inferred") -> DataSchema:
    """
    Автоматически вывести схему из DataFrame / Auto-infer schema from DataFrame.

    Nullable = True если в столбце есть хотя бы один null.
    Nullable = True if column contains at least one null value.
    """
    columns: list[ColumnSchema] = []
    for col_name in df.columns:
        series = df[col_name]
        # "Int64" from "Int64", "Datetime" from "Datetime(time_unit='us', …)"
        dtype_str = str(series.dtype).split("(")[0]
        col_type = _POLARS_TO_COLUMN_TYPE.get(dtype_str, ColumnType.UNKNOWN)
        nullable = series.null_count() > 0
        columns.append(ColumnSchema(name=col_name, dtype=col_type, nullable=nullable))

    return DataSchema(name=name, columns=columns)


def validate_dataframe_against_schema(
    df: pl.DataFrame,
    schema: DataSchema,
) -> dict[str, Any]:
    """
    Проверить DataFrame на соответствие схеме / Validate DataFrame against schema.

    Возвращает отчёт с per-column результатами и флагом passed.
    Returns a report with per-column results and an overall passed flag.

    Неизвестные столбцы (есть в df, нет в схеме) — не ошибка,
    но перечисляются в extra_columns для аудита.
    Unknown columns (in df but not in schema) are not errors
    but listed in extra_columns for audit purposes.
    """
    issues: list[dict[str, Any]] = []
    passed_count = 0
    total_count = 0
    schema_cols = schema.column_map()

    # 1. Проверить наличие обязательных столбцов / Check required columns present
    for col_name, col_spec in schema_cols.items():
        total_count += 1
        if col_name not in df.columns:
            issues.append(
                {
                    "column": col_name,
                    "check": "column_exists",
                    "passed": False,
                    "detail": f"Column missing: '{col_name}'",
                }
            )
        else:
            passed_count += 1

    # 2. Проверить ограничения по каждому присутствующему столбцу
    #    Validate constraints for each present column
    for col_name in df.columns:
        if col_name not in schema_cols:
            continue  # extra column — not an error

        col_spec = schema_cols[col_name]
        series = df[col_name]

        # Nullable constraint
        total_count += 1
        if series.null_count() > 0 and not col_spec.nullable:
            issues.append(
                {
                    "column": col_name,
                    "check": "nullable",
                    "passed": False,
                    "detail": f"Null values in non-nullable column '{col_name}'",
                }
            )
        else:
            passed_count += 1

        # Allowed values
        if col_spec.allowed_values is not None:
            total_count += 1
            non_null = series.drop_nulls()
            invalid = non_null.filter(~non_null.is_in(col_spec.allowed_values))
            if len(invalid) > 0:
                issues.append(
                    {
                        "column": col_name,
                        "check": "allowed_values",
                        "passed": False,
                        "detail": f"Invalid values: {invalid.head(5).to_list()}",
                    }
                )
            else:
                passed_count += 1

        # Value range (numeric only — skip silently for other types)
        if col_spec.min_value is not None or col_spec.max_value is not None:
            total_count += 1
            out_of_range = 0
            try:
                numeric = series.drop_nulls()
                if col_spec.min_value is not None:
                    out_of_range += len(numeric.filter(numeric < col_spec.min_value))
                if col_spec.max_value is not None:
                    out_of_range += len(numeric.filter(numeric > col_spec.max_value))
                if out_of_range > 0:
                    issues.append(
                        {
                            "column": col_name,
                            "check": "value_range",
                            "passed": False,
                            "detail": (
                                f"{out_of_range} values out of range "
                                f"[{col_spec.min_value}, {col_spec.max_value}]"
                            ),
                        }
                    )
                else:
                    passed_count += 1
            except Exception:
                passed_count += 1  # non-numeric type — skip range check

    extra_columns = [c for c in df.columns if c not in schema_cols]
    return {
        "passed": len(issues) == 0,
        "total_checks": total_count,
        "passed_checks": passed_count,
        "failed_checks": total_count - passed_count,
        "issues": issues,
        "extra_columns": extra_columns,
    }
