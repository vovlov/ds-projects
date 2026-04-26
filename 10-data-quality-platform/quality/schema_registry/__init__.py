"""Schema Registry for data contracts / Реестр схем для контрактов данных."""

from .registry import SchemaRegistry, get_registry
from .schema import ColumnSchema, ColumnType, Compatibility, DataSchema, SchemaVersion
from .validator import infer_schema_from_dataframe, validate_dataframe_against_schema

__all__ = [
    "ColumnSchema",
    "ColumnType",
    "Compatibility",
    "DataSchema",
    "SchemaRegistry",
    "SchemaVersion",
    "get_registry",
    "infer_schema_from_dataframe",
    "validate_dataframe_against_schema",
]
