"""
FastAPI-приложение для Data Quality Platform / FastAPI app.

Предоставляет HTTP-эндпоинты для профилирования, валидации качества,
проверки дрифта и управления реестром схем (data contracts).

Exposes endpoints for profiling, quality validation, drift detection,
and schema registry management (data contracts).
"""

from __future__ import annotations

import io
import json
from typing import Any

import polars as pl
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quality.alerts.alerting import AlertManager, LogAlertChannel, WebhookAlertChannel
from quality.data.profiler import profile_dataframe
from quality.quality.drift import detect_drift
from quality.quality.expectations import run_suite
from quality.schema_registry.registry import get_registry
from quality.schema_registry.schema import ColumnSchema, ColumnType, Compatibility, DataSchema
from quality.schema_registry.validator import (
    infer_schema_from_dataframe,
    validate_dataframe_against_schema,
)

app = FastAPI(
    title="Data Quality Platform",
    description=("Платформа мониторинга качества данных / Data quality monitoring platform"),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Вспомогательные функции / Helpers
# ---------------------------------------------------------------------------


def _read_upload_csv(upload: UploadFile) -> pl.DataFrame:
    """Прочитать загруженный CSV в Polars DataFrame."""
    content = upload.file.read()
    return pl.read_csv(io.BytesIO(content))


def _make_serializable(obj: Any) -> Any:
    """
    Рекурсивно приводим к JSON-сериализуемым типам.
    numpy int64/float64 и прочие не сериализуются по умолчанию.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Эндпоинты / Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Проверка здоровья сервиса / Health check."""
    return {"status": "ok"}


@app.post("/profile")
async def profile_endpoint(
    file: UploadFile = File(..., description="CSV-файл для профилирования"),
) -> JSONResponse:
    """
    Профилирование данных / Data profiling.
    Загрузите CSV — получите полный профиль каждого столбца.
    Upload a CSV to get a full column-level profile.
    """
    df = _read_upload_csv(file)
    report = profile_dataframe(df)
    return JSONResponse(content=_make_serializable(report))


@app.post("/validate")
async def validate_endpoint(
    file: UploadFile = File(..., description="CSV-файл для проверки"),
    suite: str = Form(..., description="YAML-конфиг набора проверок (как строка)"),
) -> JSONResponse:
    """
    Валидация качества данных / Data quality validation.
    Загрузите CSV + YAML-конфиг проверок — получите отчёт.
    Upload CSV + a YAML suite config string to get a quality report.
    """
    df = _read_upload_csv(file)
    suite_config = yaml.safe_load(suite)
    results = run_suite(df, suite_config)
    passed = sum(1 for r in results if r["passed"])

    report = {
        "total_checks": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    return JSONResponse(content=_make_serializable(report))


@app.post("/drift")
async def drift_endpoint(
    reference: UploadFile = File(..., description="Эталонный CSV / Reference CSV"),
    current: UploadFile = File(..., description="Текущий CSV / Current CSV"),
    columns: str | None = Form(
        None,
        description=(
            "JSON-список столбцов для проверки / JSON list of columns to check (optional)"
        ),
    ),
) -> JSONResponse:
    """
    Проверка дрифта распределений / Distribution drift detection.
    Загрузите два CSV (эталон и текущий) — получите отчёт о дрифте.
    Upload reference + current CSV to get a drift report.
    """
    ref_df = _read_upload_csv(reference)
    cur_df = _read_upload_csv(current)

    col_list = json.loads(columns) if columns else None
    report = detect_drift(ref_df, cur_df, columns=col_list)
    return JSONResponse(content=_make_serializable(report))


@app.post("/drift/alert")
async def drift_alert_endpoint(
    reference: UploadFile = File(..., description="Эталонный CSV / Reference CSV"),
    current: UploadFile = File(..., description="Текущий CSV / Current CSV"),
    columns: str | None = Form(
        None,
        description="JSON-список столбцов / JSON list of columns to check (optional)",
    ),
    webhook_url: str | None = Form(
        None,
        description=(
            "URL webhook для алертинга (например, http://churn-svc/retraining/notify). "
            "Webhook URL to notify on drift (e.g. churn service retraining endpoint)."
        ),
    ),
    severity_threshold: str = Form(
        "warning",
        description="Минимальная серьёзность / Min severity: ok|warning|critical",
    ),
) -> JSONResponse:
    """
    Детекция дрейфа + алертинг / Drift detection with alerting.

    Запускает проверку дрейфа и при обнаружении значимого отклонения
    (severity >= severity_threshold) отправляет структурированный алерт.

    Runs drift detection and sends a structured alert when drift severity
    exceeds severity_threshold. If webhook_url is provided, POSTs the alert
    to that URL (e.g. Project 01's /retraining/notify endpoint).

    Returns:
        drift_report: полный отчёт детекции дрейфа
        alert: алерт (если отправлен) или null
    """
    ref_df = _read_upload_csv(reference)
    cur_df = _read_upload_csv(current)

    col_list = json.loads(columns) if columns else None
    drift_report = detect_drift(ref_df, cur_df, columns=col_list)

    # Настраиваем каналы: всегда лог + опциональный webhook
    # Always log + optional webhook (typically Project 01's retraining endpoint)
    channels = [LogAlertChannel()]
    if webhook_url:
        channels.append(WebhookAlertChannel(url=webhook_url))

    manager = AlertManager(channels=channels, severity_threshold=severity_threshold)
    alert = manager.process_drift_report(drift_report)

    response_body = {
        "drift_report": _make_serializable(drift_report),
        "alert": _make_serializable(alert.to_dict()) if alert else None,
        "alert_sent": alert is not None,
    }
    return JSONResponse(content=response_body)


# ---------------------------------------------------------------------------
# Schema Registry — Pydantic request/response models
# ---------------------------------------------------------------------------


class ColumnSchemaRequest(BaseModel):
    """Описание столбца в теле запроса / Column descriptor in request body."""

    name: str
    dtype: str = "string"
    nullable: bool = True
    description: str = ""
    allowed_values: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None


class RegisterSchemaRequest(BaseModel):
    """Запрос регистрации схемы / Schema registration request."""

    schema_name: str
    description: str = ""
    compatibility: str = "BACKWARD"
    columns: list[ColumnSchemaRequest]
    version: str | None = None
    allow_breaking: bool = False


class CompatibilityCheckRequest(BaseModel):
    """Запрос проверки совместимости / Compatibility check request."""

    schema_name: str
    description: str = ""
    compatibility: str = "BACKWARD"
    columns: list[ColumnSchemaRequest]


# ---------------------------------------------------------------------------
# Schema Registry — helpers
# ---------------------------------------------------------------------------


def _build_data_schema(
    req_columns: list[ColumnSchemaRequest],
    name: str,
    compatibility: str,
    description: str,
) -> DataSchema:
    """Собрать DataSchema из Pydantic-запроса / Build DataSchema from Pydantic request."""
    cols = [
        ColumnSchema(
            name=c.name,
            dtype=ColumnType(c.dtype),
            nullable=c.nullable,
            description=c.description,
            allowed_values=c.allowed_values,
            min_value=c.min_value,
            max_value=c.max_value,
        )
        for c in req_columns
    ]
    compat = Compatibility(compatibility.upper())
    return DataSchema(name=name, columns=cols, compatibility=compat, description=description)


# ---------------------------------------------------------------------------
# Schema Registry — endpoints
# ---------------------------------------------------------------------------


@app.post("/schema/register")
def register_schema_endpoint(req: RegisterSchemaRequest) -> JSONResponse:
    """
    Зарегистрировать схему данных (data contract) / Register a data schema.

    Первая регистрация: версия 1.0.0.
    Последующие: автоматический bump (breaking → major, non-breaking → minor).
    allow_breaking=true обходит BACKWARD-проверку (используйте с осторожностью).

    First registration: version 1.0.0.
    Subsequent: auto-bump (breaking → major, non-breaking → minor).
    allow_breaking=true bypasses BACKWARD check (use with caution).
    """
    try:
        schema = _build_data_schema(
            req.columns, req.schema_name, req.compatibility, req.description
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    registry = get_registry()
    try:
        sv = registry.register(schema, version=req.version, allow_breaking=req.allow_breaking)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return JSONResponse(content=sv.to_dict(), status_code=201)


@app.get("/schema/list")
def list_schemas_endpoint() -> JSONResponse:
    """
    Список всех зарегистрированных схем / List all registered schema names.
    """
    return JSONResponse(content={"schemas": get_registry().list_schemas()})


@app.get("/schema/{name}/versions")
def list_versions_endpoint(name: str) -> JSONResponse:
    """
    Список версий конкретной схемы / List versions of a specific schema.
    """
    versions = get_registry().list_versions(name)
    if not versions:
        raise HTTPException(status_code=404, detail=f"Schema not found: '{name}'")
    return JSONResponse(content={"schema_name": name, "versions": versions})


@app.get("/schema/{name}")
def get_schema_endpoint(name: str, version: str | None = None) -> JSONResponse:
    """
    Получить схему (последнюю или конкретную версию).
    Get a schema — latest or a specific version.
    """
    sv = get_registry().get(name, version)
    if sv is None:
        detail = f"Schema '{name}'" + (f" version '{version}'" if version else "") + " not found"
        raise HTTPException(status_code=404, detail=detail)
    return JSONResponse(content=sv.to_dict())


@app.post("/schema/{name}/validate")
async def validate_against_schema_endpoint(
    name: str,
    file: UploadFile = File(..., description="CSV для проверки / CSV to validate"),
    version: str | None = Form(
        None, description="Версия схемы / Schema version (optional, latest if omitted)"
    ),
) -> JSONResponse:
    """
    Проверить CSV-файл на соответствие зарегистрированной схеме.
    Validate a CSV file against a registered schema.

    Проверяет: наличие столбцов, nullable, диапазоны, допустимые значения.
    Checks: column presence, nullable, value ranges, allowed values.
    """
    sv = get_registry().get(name, version)
    if sv is None:
        raise HTTPException(status_code=404, detail=f"Schema not found: '{name}'")

    df = _read_upload_csv(file)
    report = validate_dataframe_against_schema(df, sv.schema)
    report["schema_name"] = name
    report["schema_version"] = sv.version
    return JSONResponse(content=_make_serializable(report))


@app.post("/schema/compatible")
def check_compatibility_endpoint(req: CompatibilityCheckRequest) -> JSONResponse:
    """
    Проверить, совместима ли новая схема с текущей зарегистрированной версией.
    Check if a candidate schema is backward-compatible with the registered version.

    Возвращает список breaking changes (если есть).
    Returns list of breaking changes (if any).
    """
    try:
        candidate = _build_data_schema(
            req.columns, req.schema_name, req.compatibility, req.description
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = get_registry().check_compatibility(req.schema_name, candidate)
    return JSONResponse(content=result)


@app.post("/schema/infer")
async def infer_schema_endpoint(
    file: UploadFile = File(..., description="CSV для инференса схемы / CSV to infer schema from"),
    schema_name: str = Form("inferred", description="Имя схемы / Schema name"),
) -> JSONResponse:
    """
    Автоматически вывести схему из CSV / Auto-infer schema from a CSV file.

    Удобно для начального onboarding: загрузите данные → получите черновик схемы
    → зарегистрируйте через /schema/register.
    Useful for initial onboarding: upload data → get schema draft → register via /schema/register.
    """
    df = _read_upload_csv(file)
    schema = infer_schema_from_dataframe(df, name=schema_name)
    sv_dict = {
        "schema_name": schema.name,
        "description": schema.description,
        "compatibility": schema.compatibility.value,
        "columns": [c.to_dict() for c in schema.columns],
        "inferred_from_rows": df.height,
    }
    return JSONResponse(content=_make_serializable(sv_dict))
