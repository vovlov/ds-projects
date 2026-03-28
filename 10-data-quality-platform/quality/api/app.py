"""
FastAPI-приложение для Data Quality Platform / FastAPI app.

Предоставляет HTTP-эндпоинты для профилирования, валидации качества
и проверки дрифта. Все данные принимаются как CSV-файлы.

Exposes endpoints for profiling, quality validation, and drift detection.
Data is uploaded as CSV files.
"""

from __future__ import annotations

import io
import json
from typing import Any

import polars as pl
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from quality.data.profiler import profile_dataframe
from quality.quality.drift import detect_drift
from quality.quality.expectations import run_suite

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
