"""
FastAPI-приложение для Data Quality Platform / FastAPI app.

Предоставляет HTTP-эндпоинты для профилирования, валидации качества,
проверки дрифта, управления реестром схем (data contracts),
отслеживания родословной данных (data lineage) и аудита безопасности ML.

Exposes endpoints for profiling, quality validation, drift detection,
schema registry management (data contracts), data lineage tracking,
and OWASP ML security audits.
"""

from __future__ import annotations

import io
import json
from typing import Any

import polars as pl
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from quality.alerts.alerting import AlertManager, LogAlertChannel, WebhookAlertChannel
from quality.data.profiler import profile_dataframe
from quality.label_quality.confid_learn import DecoupledConfidentLearning
from quality.lineage.tracker import RunState, get_tracker
from quality.monitoring.prediction_monitor import (
    get_monitor as get_prediction_monitor,
)
from quality.monitoring.prediction_monitor import (
    reset_prediction_monitor,
)
from quality.quality.drift import detect_drift
from quality.quality.expectations import run_suite
from quality.quality.stat_tests import batch_extended_drift
from quality.schema_registry.registry import get_registry
from quality.schema_registry.schema import ColumnSchema, ColumnType, Compatibility, DataSchema
from quality.schema_registry.validator import (
    infer_schema_from_dataframe,
    validate_dataframe_against_schema,
)
from quality.security.owasp import OWASPMLAudit
from quality.security.pii_detector import detect_pii
from quality.sla.monitor import get_monitor, reset_monitor
from quality.sla.slo import SLIObservation, SLIType, SLODefinition
from quality.synthetic.generator import SyntheticConfig, SyntheticDataGenerator

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
# Extended drift test battery — Pydantic models
# ---------------------------------------------------------------------------


class ExtendedDriftRequest(BaseModel):
    """
    Запрос расширенной батареи тестов дрейфа / Extended drift test battery request.

    reference: эталонные данные {column_name: [values...]}
    current:   текущие данные   {column_name: [values...]}
    feature_types: опциональный маппинг column → "continuous"|"categorical"|"auto"
    bins: число бинов для Wasserstein/JS (по умолчанию 20)
    """

    reference: dict[str, list]
    current: dict[str, list]
    feature_types: dict[str, str] | None = None
    bins: int = 20


@app.post("/drift/extended")
def drift_extended_endpoint(req: ExtendedDriftRequest) -> JSONResponse:
    """
    Расширенная батарея статистических тестов дрейфа / Extended statistical drift battery.

    Запускает по каждому признаку:
      - continuous: Wasserstein distance + Jensen-Shannon divergence
      - categorical: Chi-squared test + Jensen-Shannon divergence
      - auto: автоопределение типа (≤20 уникальных значений → categorical)

    Runs per-feature tests:
      continuous → Wasserstein distance + JS divergence
      categorical → chi-squared + JS divergence
      auto → type auto-detected by n_unique values

    Returns:
      columns_checked: число проверенных столбцов
      columns_with_drift: столбцов с обнаруженным дрейфом
      critical_columns: столбцы с severity="critical"
      overall_drift: True если хотя бы один столбец в дрейфе
      details: результаты по каждому столбцу
    """
    report = batch_extended_drift(
        reference_data=req.reference,
        current_data=req.current,
        feature_types=req.feature_types,
        bins=req.bins,
    )
    return JSONResponse(content=_make_serializable(report))


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


# ---------------------------------------------------------------------------
# Data Lineage — Pydantic request models
# ---------------------------------------------------------------------------


class DatasetRef(BaseModel):
    """Ссылка на датасет в lineage-событии / Dataset reference in a lineage event."""

    namespace: str = "default"
    name: str
    facets: dict[str, Any] = {}


class LineageEventRequest(BaseModel):
    """
    Запрос записи lineage-события / Lineage event record request.

    Соответствует OpenLineage RunEvent формату.
    Corresponds to the OpenLineage RunEvent format.
    """

    job_namespace: str
    job_name: str
    event_type: str = "COMPLETE"
    run_id: str | None = None
    inputs: list[DatasetRef] = []
    outputs: list[DatasetRef] = []
    facets: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Data Lineage — endpoints
# ---------------------------------------------------------------------------


@app.post("/lineage/event")
def record_lineage_event(req: LineageEventRequest) -> JSONResponse:
    """
    Записать событие родословной данных / Record a data lineage event.

    Принимает OpenLineage RunEvent: job + его входные/выходные датасеты.
    Обновляет граф родословной для визуализации.

    Accepts an OpenLineage RunEvent: job + input/output datasets.
    Updates the lineage graph for subsequent visualization queries.

    Пример (Example):
        POST /lineage/event
        {
          "job_namespace": "churn-service",
          "job_name": "train_model",
          "event_type": "COMPLETE",
          "inputs": [{"namespace": "postgres", "name": "customers"}],
          "outputs": [{"namespace": "mlflow", "name": "churn_model_v1"}]
        }
    """
    try:
        event_type = RunState(req.event_type.upper())
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid event_type '{req.event_type}'. Must be one of: {[s.value for s in RunState]}",  # noqa: E501
        ) from exc

    tracker = get_tracker()
    event = tracker.record(
        job_namespace=req.job_namespace,
        job_name=req.job_name,
        event_type=event_type,
        inputs=[{"namespace": d.namespace, "name": d.name, **d.facets} for d in req.inputs],
        outputs=[{"namespace": d.namespace, "name": d.name, **d.facets} for d in req.outputs],
        run_id=req.run_id,
        **req.facets,
    )
    return JSONResponse(content=event.to_dict(), status_code=201)


@app.get("/lineage/graph")
def get_lineage_graph() -> JSONResponse:
    """
    Получить полный граф родословной в формате D3.js / Get the full lineage graph (D3.js format).

    Возвращает все узлы (Dataset + Job) и рёбра потока данных.
    Формат совместим с D3 force-directed simulation.

    Returns all nodes (Dataset + Job) and data-flow edges.
    Format is compatible with D3 force-directed simulation.
    """
    graph = get_tracker().get_graph()
    return JSONResponse(content=graph.to_dict())


@app.get("/lineage/dataset/{namespace}/{name}")
def get_dataset_lineage(namespace: str, name: str) -> JSONResponse:
    """
    Родословная конкретного датасета (вверх и вниз) / Dataset lineage (upstream + downstream).

    Возвращает подграф, связанный с указанным датасетом.
    Returns the subgraph connected to the specified dataset.
    """
    tracker = get_tracker()
    dataset_id = f"{namespace}/{name}"
    graph = tracker.get_graph()

    if dataset_id not in graph.nodes:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found in lineage graph.",
        )

    lineage = graph.lineage_for_dataset(dataset_id)
    return JSONResponse(content=lineage)


@app.get("/lineage/upstream/{namespace}/{name}")
def get_upstream_lineage(namespace: str, name: str, max_depth: int = 10) -> JSONResponse:
    """
    Получить все узлы выше по потоку для датасета / Get upstream lineage for a dataset.

    Полезно для impact analysis: какие источники влияют на этот датасет?
    Useful for impact analysis: which sources feed into this dataset?
    """
    node_id = f"{namespace}/{name}"
    graph = get_tracker().get_graph()
    upstream_ids = graph.upstream(node_id, max_depth=max_depth)
    upstream_nodes = [graph.nodes[nid].to_dict() for nid in upstream_ids if nid in graph.nodes]
    return JSONResponse(  # noqa: E501
        content={"node_id": node_id, "upstream": upstream_nodes, "depth": len(upstream_nodes)}
    )


@app.get("/lineage/downstream/{namespace}/{name}")
def get_downstream_lineage(namespace: str, name: str, max_depth: int = 10) -> JSONResponse:
    """
    Получить все узлы ниже по потоку для датасета / Get downstream lineage for a dataset.

    Полезно для impact analysis: что сломается если изменить этот датасет?
    Useful for impact analysis: what breaks if this dataset changes?
    """
    node_id = f"{namespace}/{name}"
    graph = get_tracker().get_graph()
    downstream_ids = graph.downstream(node_id, max_depth=max_depth)
    downstream_nodes = [graph.nodes[nid].to_dict() for nid in downstream_ids if nid in graph.nodes]
    return JSONResponse(
        content={"node_id": node_id, "downstream": downstream_nodes, "depth": len(downstream_nodes)}
    )


@app.get("/lineage/events")
def list_lineage_events(
    job_name: str | None = None,
    event_type: str | None = None,
    limit: int = 50,
) -> JSONResponse:
    """
    Список последних lineage-событий / List recent lineage events.

    Поддерживает фильтрацию по имени задачи и типу события.
    Supports filtering by job name and event type.
    """
    tracker = get_tracker()
    run_state = None
    if event_type:
        try:
            run_state = RunState(event_type.upper())
        except ValueError as exc:
            raise HTTPException(  # noqa: B904
                status_code=422, detail=f"Invalid event_type: '{event_type}'"
            ) from exc

    events = tracker.get_events(job_name=job_name, event_type=run_state, limit=limit)
    return JSONResponse(content={"events": [e.to_dict() for e in events], "total": len(events)})


@app.get("/lineage/summary")
def get_lineage_summary() -> JSONResponse:
    """
    Краткая статистика lineage-трекера / Lineage tracker summary statistics.

    Возвращает число событий, узлов, рёбер и типов событий.
    Returns event counts, node/edge counts, and event type breakdown.
    """
    return JSONResponse(content=get_tracker().summary())


# ---------------------------------------------------------------------------
# Security endpoints — OWASP ML Top 10 + PII detection
# ---------------------------------------------------------------------------


class SecurityAuditRequest(BaseModel):
    """Request body for OWASP ML security audit / Запрос аудита безопасности."""

    numeric_columns: dict[str, list[float]] = {}
    all_columns: dict[str, list[Any]] = {}
    label_column: list[Any] | None = None
    output_fields: list[str] = []
    has_rate_limiting: bool = False
    exposes_probabilities: bool = True


class PIIRequest(BaseModel):
    """Request body for PII scan / Запрос сканирования PII."""

    columns: dict[str, list[Any]]
    max_examples: int = 3


@app.post("/security/audit", status_code=200)
def security_audit(request: SecurityAuditRequest) -> JSONResponse:
    """
    Аудит безопасности ML по OWASP ML Top 10 / OWASP ML Top 10 security audit.

    Принимает метаданные датасета и конфигурацию API, возвращает отчёт
    с найденными проблемами, оценкой (0-100) и рекомендациями.

    Accepts dataset metadata and API config, returns a prioritised finding
    report with a 0-100 security score and actionable recommendations.

    Checks performed:
    - ML01: Input manipulation (adversarial outliers)
    - ML02: Data poisoning (label imbalance)
    - ML03: Model inversion (exposed internal fields)
    - ML04: Membership inference (high-cardinality columns)
    - ML05: Model theft (missing rate limits)
    - ML08: Model skewing (high missing-value rates)
    - ML09: Output integrity (missing signature)
    """
    auditor = OWASPMLAudit()
    report = auditor.run_audit(
        numeric_columns=request.numeric_columns,
        all_columns=request.all_columns,
        label_column=request.label_column,
        output_fields=request.output_fields,
        has_rate_limiting=request.has_rate_limiting,
        exposes_probabilities=request.exposes_probabilities,
    )
    return JSONResponse(content=report.to_dict())


@app.post("/security/pii", status_code=200)
def pii_scan(request: PIIRequest) -> JSONResponse:
    """
    Сканирование датасета на наличие персональных данных (PII).

    Scan a dataset for Personally Identifiable Information (PII).

    Detects: email, phone, SSN, credit card, IP address, passport,
    IBAN, date of birth, full names.

    Returns GDPR compliance flag and masked examples for each finding.
    """
    report = detect_pii(request.columns, max_examples=request.max_examples)
    return JSONResponse(content=report.to_dict())


@app.post("/security/audit/csv", status_code=200)
async def security_audit_csv(
    file: UploadFile = File(..., description="CSV-файл для аудита безопасности"),
    has_rate_limiting: bool = Form(False),
    exposes_probabilities: bool = Form(True),
    label_col: str | None = Form(None, description="Имя столбца с метками (опционально)"),
) -> JSONResponse:
    """
    Аудит безопасности CSV-датасета / CSV dataset security audit.

    Загрузите CSV — получите полный OWASP ML аудит + PII-сканирование.
    Upload a CSV to run a combined OWASP + PII security audit.
    """
    df = _read_upload_csv(file)

    # Build column dicts from Polars DataFrame
    numeric_cols: dict[str, list[float]] = {}
    all_cols: dict[str, list[Any]] = {}
    label_column: list[Any] | None = None

    for col in df.columns:
        values = df[col].to_list()
        all_cols[col] = values
        if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8):
            numeric_cols[col] = [float(v) for v in values if v is not None]
        if label_col and col == label_col:
            label_column = values

    auditor = OWASPMLAudit()
    audit_report = auditor.run_audit(
        numeric_columns=numeric_cols,
        all_columns=all_cols,
        label_column=label_column,
        has_rate_limiting=has_rate_limiting,
        exposes_probabilities=exposes_probabilities,
    )

    pii_report = detect_pii(all_cols)

    return JSONResponse(
        content={
            "owasp_audit": audit_report.to_dict(),
            "pii_scan": pii_report.to_dict(),
            "rows": len(df),
            "columns": len(df.columns),
        }
    )


@app.get("/security/checklist")
def security_checklist() -> JSONResponse:
    """
    OWASP ML Top 10 справочник / OWASP ML Top 10 reference checklist.

    Возвращает список всех 10 рисков с описаниями и рекомендациями.
    Returns all 10 OWASP ML risks with descriptions and mitigations.
    """
    checklist = [
        {
            "id": "ML01",
            "title": "Input Manipulation Attack",
            "description": "Adversarial examples crafted to fool model predictions",
            "mitigation": "Input validation, adversarial training, outlier pre-filtering",
        },
        {
            "id": "ML02",
            "title": "Data Poisoning Attack",
            "description": "Injecting malicious samples into training data",
            "mitigation": "Data provenance tracking, anomaly detection on labels, audit trails",
        },
        {
            "id": "ML03",
            "title": "Model Inversion Attack",
            "description": "Reconstructing training data from model outputs/gradients",
            "mitigation": (
                "Limit output to top-1 class, apply output perturbation, avoid exposing logits"
            ),
        },
        {
            "id": "ML04",
            "title": "Membership Inference Attack",
            "description": "Determining if a record was in the training set",
            "mitigation": (
                "Differential privacy, k-anonymity, remove quasi-identifiers from training"
            ),
        },
        {
            "id": "ML05",
            "title": "Model Theft",
            "description": "Cloning a model via API queries (model extraction)",
            "mitigation": "Rate limiting, prediction rounding, query budgets, watermarking",
        },
        {
            "id": "ML06",
            "title": "AI Supply Chain Attack",
            "description": "Compromised pretrained models, datasets, or ML libraries",
            "mitigation": (
                "Verify model checksums, pin library versions, use private model registry"
            ),
        },
        {
            "id": "ML07",
            "title": "Transfer Learning Attack",
            "description": "Backdoors injected via poisoned pretrained base models",
            "mitigation": "Fine-tune on trusted data only, audit base model provenance",
        },
        {
            "id": "ML08",
            "title": "Model Skewing",
            "description": "Manipulating inference inputs to bias model behaviour",
            "mitigation": "Enforce feature constraints at inference, monitor feature distributions",
        },
        {
            "id": "ML09",
            "title": "Output Integrity Attack",
            "description": "Tampering with predictions in transit between model and consumer",
            "mitigation": "Sign outputs with HMAC-SHA256, TLS end-to-end, prediction IDs",
        },
        {
            "id": "ML10",
            "title": "Model Poisoning",
            "description": "Poisoning the model itself during training or fine-tuning",
            "mitigation": "Secure training infrastructure, model checksums, reproducible builds",
        },
    ]
    return JSONResponse(content={"owasp_ml_top_10": checklist, "version": "2023"})


# ---------------------------------------------------------------------------
# SLA Monitoring — Pydantic request/response models
# ---------------------------------------------------------------------------


class SLODefineRequest(BaseModel):
    """Запрос регистрации SLO / SLO registration request."""

    service: str
    sli_type: str = "availability"
    target: float
    window_days: int = 30
    latency_threshold_ms: float | None = None
    description: str = ""


class SLIObserveRequest(BaseModel):
    """Запрос записи SLI-наблюдения / SLI observation record request."""

    service: str
    sli_type: str = "availability"
    good: int
    total: int
    metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# SLA Monitoring — endpoints
# ---------------------------------------------------------------------------


@app.post("/sla/define", status_code=201)
def sla_define(req: SLODefineRequest) -> JSONResponse:
    """
    Зарегистрировать SLO для сервиса / Register an SLO for a service.

    Первая регистрация создаёт трекер бюджета ошибок.
    Повторная регистрация перезаписывает существующий SLO и сбрасывает трекер.

    First registration creates an error budget tracker.
    Re-registration overwrites the existing SLO and resets its tracker.

    Example:
        POST /sla/define
        {"service": "churn-api", "sli_type": "availability", "target": 0.999}
    """
    try:
        sli_type = SLIType(req.sli_type)
        slo = SLODefinition(
            service=req.service,
            sli_type=sli_type,
            target=req.target,
            window_days=req.window_days,
            latency_threshold_ms=req.latency_threshold_ms,
            description=req.description,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = get_monitor().define_slo(slo)
    return JSONResponse(content=result.to_dict(), status_code=201)


@app.post("/sla/observe", status_code=201)
def sla_observe(req: SLIObserveRequest) -> JSONResponse:
    """
    Записать SLI-наблюдение / Record an SLI observation batch.

    Обновляет трекер бюджета ошибок для соответствующего SLO.
    Updates the error budget tracker for the matching SLO.

    Args:
        good: количество успешных событий (запросов, ответов < порога latency)
        total: общее количество событий

    Example:
        POST /sla/observe
        {"service": "churn-api", "sli_type": "availability", "good": 998, "total": 1000}
    """
    try:
        sli_type = SLIType(req.sli_type)
        obs = SLIObservation(
            service=req.service,
            sli_type=sli_type,
            good=req.good,
            total=req.total,
            metadata=req.metadata,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = get_monitor().observe(obs)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return JSONResponse(content=result.to_dict(), status_code=201)


@app.get("/sla/status")
def sla_status_all() -> JSONResponse:
    """
    Статус бюджета ошибок для всех сервисов / Error budget status for all services.

    Возвращает burn rates (1h / 6h / 3d), оставшийся бюджет и активные алерты.
    Returns burn rates (1h / 6h / 3d), remaining budget, and active alerts.
    """
    statuses = get_monitor().get_all_statuses()
    return JSONResponse(  # noqa: E501
        content={"services": [s.to_dict() for s in statuses], "total": len(statuses)}
    )


@app.get("/sla/status/{service}")
def sla_status_service(service: str, sli_type: str | None = None) -> JSONResponse:
    """
    Статус бюджета для конкретного сервиса / Error budget status for a specific service.

    Args:
        service: имя сервиса (как зарегистрировано в /sla/define)
        sli_type: фильтр по типу SLI (опционально)
    """
    try:
        sli = SLIType(sli_type) if sli_type else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    statuses = get_monitor().get_status(service, sli_type=sli)
    if not statuses:
        raise HTTPException(status_code=404, detail=f"No SLO found for service '{service}'")

    return JSONResponse(content={"service": service, "slos": [s.to_dict() for s in statuses]})


@app.get("/sla/burn-rate/{service}")
def sla_burn_rate(service: str) -> JSONResponse:
    """
    Burn rate и прогноз исчерпания бюджета / Burn rate and budget exhaustion forecast.

    Возвращает multi-window burn rates и прогнозируемое время до исчерпания бюджета.
    Returns multi-window burn rates and projected time until budget exhaustion.
    Используется для Grafana-дашбордов и alerting-интеграций.
    """
    statuses = get_monitor().get_status(service)
    if not statuses:
        raise HTTPException(status_code=404, detail=f"No SLO found for service '{service}'")

    result = []
    for s in statuses:
        result.append(
            {
                "service": s.service,
                "sli_type": s.slo.sli_type.value,
                "burn_rates": s.to_dict()["burn_rates"],
                "budget_remaining_pct": round(s.budget_remaining_pct, 2),
                "budget_consumed_pct": round(s.budget_consumed_pct, 2),
                "projected_exhaustion_hours": s.projected_exhaustion_hours,
                "active_alerts": [a.to_dict() for a in s.active_alerts],
            }
        )
    return JSONResponse(content={"service": service, "burn_rates": result})


@app.post("/sla/report")
def sla_report() -> JSONResponse:
    """
    Сводный отчёт о соответствии SLA / Aggregate SLA compliance report.

    Охватывает все зарегистрированные SLO: статус compliance, число нарушений,
    критических и высоких алертов.

    Covers all registered SLOs: compliance status, violation count,
    critical and high alert counts.
    """
    report = get_monitor().generate_report()
    return JSONResponse(content=report.to_dict())


@app.get("/sla/slos")
def sla_list_slos() -> JSONResponse:
    """
    Список всех зарегистрированных SLO / List all registered SLO definitions.
    """
    slos = get_monitor().list_slos()
    return JSONResponse(content={"slos": slos, "total": len(slos)})


@app.get("/sla/observations")
def sla_observations(service: str | None = None, limit: int = 50) -> JSONResponse:
    """
    Последние SLI-наблюдения / Recent SLI observations.

    Полезно для отладки и аудита измерений.
    Useful for debugging and auditing SLI measurements.
    """
    obs = get_monitor().get_recent_observations(service=service, limit=limit)
    return JSONResponse(content={"observations": obs, "total": len(obs)})


@app.post("/sla/reset")
def sla_reset() -> JSONResponse:
    """
    Сбросить все SLO и наблюдения / Reset all SLOs and observations.

    Полезно для тестирования / Useful for testing and demos.
    """
    reset_monitor()
    return JSONResponse(content={"status": "reset", "message": "All SLOs and observations cleared"})


# ---------------------------------------------------------------------------
# Label Quality — Decoupled Confident Learning (DeCoLe)
# ---------------------------------------------------------------------------


class LabelQualityRequest(BaseModel):
    """Запрос на аудит качества разметки / Label quality audit request."""

    labels: list[int] = Field(
        ...,
        description="Noisy labels ỹ_i ∈ {0,...,K-1}, shape [n]",
    )
    pred_probs: list[list[float]] = Field(
        ...,
        description=(
            "Out-of-fold predicted probabilities P̂(Y=j|x_i), shape [n, K]. "
            "Must come from cross-validation — not predictions on the full train set."
        ),
    )
    groups: list[str] | None = Field(
        default=None,
        description=(
            "Optional subgroup membership labels, shape [n]. "
            "E.g. annotator_id, data_source, demographic_group. "
            "When provided, DeCoLe builds a separate noise model per group."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fallback threshold when a group has no examples of a given class.",
    )


@app.post("/label_quality/check", status_code=200)
def label_quality_check(request: LabelQualityRequest) -> JSONResponse:
    """
    Обнаружение ошибок разметки методом Decoupled Confident Learning.
    Detect label errors using Decoupled Confident Learning (DeCoLe).

    Реализует DeCoLe (arXiv:2507.07216, 2025) — расширение стандартного Confident Learning
    (Northcutt et al. 2021 JAIR), которое строит отдельную матрицу перехода шума
    для каждой подгруппы данных. Это позволяет обнаруживать систематические ошибки
    аннотации, характерные для конкретных слоёв данных (аннотатор, источник, демография).

    Implements DeCoLe (arXiv:2507.07216, 2025) — an extension of standard Confident Learning
    (Northcutt et al. 2021 JAIR) that builds a separate noise transition matrix for each
    data subgroup. Detects systematic annotation errors specific to particular data slices
    (annotator, data source, demographic group).

    Важно / Important: pred_probs должны быть получены через cross-validation на train set.
    pred_probs must come from cross-validation on the train set — not from a model
    trained on all data (this would produce false negatives on memorized examples).

    Returns:
        LabelQualityReport с подозрительными индексами, типами ошибок и матрицами шума.
        LabelQualityReport with suspect indices, error types, and per-group noise matrices.
    """
    import numpy as np

    labels = request.labels
    pred_probs = request.pred_probs

    if len(labels) == 0:
        raise HTTPException(status_code=422, detail="labels must not be empty")

    if len(labels) != len(pred_probs):
        raise HTTPException(
            status_code=422,
            detail=(
                f"labels length ({len(labels)}) must match pred_probs length ({len(pred_probs)})"
            ),
        )

    probs_arr = np.asarray(pred_probs, dtype=float)
    n_classes = probs_arr.shape[1] if probs_arr.ndim == 2 else 0
    if n_classes < 2:
        raise HTTPException(
            status_code=422,
            detail="pred_probs must have at least 2 columns (classes)",
        )

    if request.groups is not None and len(request.groups) != len(labels):
        raise HTTPException(
            status_code=422,
            detail=(
                f"groups length ({len(request.groups)}) must match labels length ({len(labels)})"
            ),
        )

    dcl = DecoupledConfidentLearning(confidence_threshold=request.confidence_threshold)
    report = dcl.find_label_errors(
        labels=labels,
        pred_probs=pred_probs,
        groups=request.groups,
    )
    return JSONResponse(content=report.to_dict())


@app.get("/label_quality/info")
def label_quality_info() -> JSONResponse:
    """
    Справка о методе DeCoLe / Reference info about the DeCoLe method.

    Возвращает краткое описание алгоритма, рекомендуемый workflow и ссылки.
    Returns a brief algorithm description, recommended workflow, and references.
    """
    return JSONResponse(
        content={
            "method": "Decoupled Confident Learning (DeCoLe)",
            "paper": "arXiv:2507.07216 (2025)",
            "base_method": "Confident Learning — Northcutt et al. 2021 JAIR 74:1-65",
            "key_idea": (
                "Builds a separate noise transition matrix Q_g[s,y] per data subgroup g, "
                "so high-noise groups don't inflate thresholds for clean groups. "
                "Detects systematic annotation errors correlated with subgroup membership."
            ),
            "workflow": [
                "1. Train classifier with K-fold cross-validation, save out-of-fold pred_probs.",
                "2. POST /label_quality/check with labels + pred_probs + optional groups.",
                "3. Inspect errors sorted by confidence (descending).",
                "4. Prioritize 'confident_disagreement' errors (confidence ≥ 0.9).",
                "5. Re-annotate or remove flagged examples, retrain, repeat.",
            ],
            "error_types": {
                "confident_disagreement": (
                    "Model confidence ≥ 0.9 in a different class — almost certainly mislabeled."
                ),
                "high_noise_group": "Error in a subgroup with estimated noise_rate > 10%.",
                "off_diagonal": "Standard off-diagonal confident joint entry — probable mislabel.",
            },
            "when_to_use": [
                "Before training: audit annotation quality by data source or annotator.",
                "After training: investigate systematic underperformance on a subgroup.",
                (
                    "Compliance: EU AI Act Article 10 requires data quality processes "
                    "including label error checks."
                ),
            ],
        }
    )


# ---------------------------------------------------------------------------
# Synthetic Data Generation / Генерация синтетических данных
# ---------------------------------------------------------------------------


class SyntheticGenerateRequest(BaseModel):
    """Запрос на генерацию синтетических данных."""

    data: dict[str, list[Any]] = Field(
        ...,
        description="Исходные данные: column_name → list of values (обучающая выборка).",
        examples=[{"age": [25, 30, 45, 50, 22], "income": [50000, 60000, 80000, 90000, 40000]}],
    )
    n_samples: int = Field(100, ge=1, le=10000, description="Число синтетических строк.")
    epsilon: float | None = Field(
        None,
        gt=0,
        description=(
            "Бюджет ε-дифференциальной приватности. None = без шума. "
            "Малые значения (ε<1) = сильная приватность, большие (ε>10) ≈ без шума."
        ),
    )
    seed: int | None = Field(42, description="Seed для воспроизводимости.")
    categorical_threshold: int = Field(
        10,
        ge=2,
        le=100,
        description="Порог уникальных значений для auto-detect categorical типа.",
    )


@app.post("/synthetic/generate", status_code=200)
def synthetic_generate(request: SyntheticGenerateRequest) -> JSONResponse:
    """
    Генерация синтетических данных (Gaussian Copula).

    Принимает реальные данные (dict column→values), возвращает синтетические
    строки с сохранёнными статистиками и корреляциями.

    **Алгоритм:**
    - Continuous: Gaussian fit → Cholesky correlated sampling → clip to [min, max].
    - Categorical: эмпирическое мультиномиальное распределение.
    - ε-DP: Laplace mechanism для mean estimates (Dwork et al. 2006).

    **Применение:** GDPR/EU AI Act тестирование без реальных данных, data augmentation,
    sharing datasets without exposing individual records.

    Generates synthetic tabular data preserving distributions and correlations.
    Optional ε-differential privacy via Laplace mechanism on mean estimates.
    """
    if not request.data:
        raise HTTPException(
            status_code=422, detail="data не может быть пустым / data must not be empty"
        )

    # Проверяем, что хотя бы один столбец непустой
    if all(len(v) == 0 for v in request.data.values()):
        raise HTTPException(status_code=422, detail="Все столбцы пустые / All columns are empty")

    config = SyntheticConfig(
        n_samples=request.n_samples,
        epsilon=request.epsilon,
        seed=request.seed,
        categorical_threshold=request.categorical_threshold,
    )
    gen = SyntheticDataGenerator(config)
    result = gen.fit_generate(request.data, n_samples=request.n_samples)

    response = result.to_dict()
    response["data"] = result.data
    return JSONResponse(content=_make_serializable(response))


@app.get("/synthetic/info")
def synthetic_info() -> JSONResponse:
    """
    Описание алгоритма генерации синтетических данных и EU AI Act compliance.
    Algorithm description and compliance information.
    """
    return JSONResponse(
        content={
            "algorithm": "Gaussian Copula с ε-дифференциальной приватностью",
            "description": (
                "Генерирует синтетические табличные данные, сохраняя маргинальные "
                "распределения (Gaussian для continuous, эмпирические частоты для categorical) "
                "и линейные корреляции через разложение Холецкого."
            ),
            "continuous_columns": (
                "Fit: μ (mean) + σ (std). Sampling: N(μ, σ) → Cholesky transform → clip [min, max]."
            ),
            "categorical_columns": (
                "Empirical multinomial: sample proportionally to observed frequencies."
            ),
            "differential_privacy": {
                "mechanism": "Laplace (additive noise on mean estimates)",
                "guarantee": "(ε, 0)-DP для mean queries (Dwork et al. 2006 TCC)",
                "sensitivity": "Δf = (max - min) / n — global sensitivity для mean query",
                "note": "Малые ε = сильнее приватность, но ниже fidelity_score.",
            },
            "fidelity_score": (
                "Нормализованная близость μ и σ: 1 - mean(|Δμ|/σ + |Δσ|/σ) / 2. "
                "Диапазон [0, 1], выше = лучше."
            ),
            "compliance": {
                "GDPR_Article_5": (
                    "Анонимизация: синтетические данные не привязаны к реальным субъектам."
                ),
                "EU_AI_Act_Article_10": (
                    "Data augmentation без использования персональных данных в production."
                ),
                "NIST_AI_RMF": "GOVERN 1.6 — privacy risk mitigation via synthetic data.",
            },
            "references": [
                "Nelsen 2006 'An Introduction to Copulas', Springer (Gaussian Copula)",
                "Dwork et al. 2006 TCC (ε-DP Laplace Mechanism)",
                "Gentle 2009 'Computational Statistics' §6.2 (Cholesky sampling)",
                "Jordon et al. 2022 NeurIPS (synthetic data evaluation)",
            ],
            "usage": [
                "1. POST /synthetic/generate с реальными данными и n_samples.",
                "2. Укажите epsilon для ε-DP (рекомендуется ε ∈ [1, 10]).",
                "3. Используйте fidelity_score для оценки качества (>0.8 = хорошо).",
                "4. Верифицируйте через /drift или /drift/extended: реальные vs синтетические.",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Prediction Distribution Monitoring — concept drift на выходе модели
# ---------------------------------------------------------------------------


class ObserveRequest(BaseModel):
    """Запрос на добавление предсказаний в монитор."""

    predictions: list[float] = Field(
        ...,
        description="Вероятности или метки [0, 1]. Probabilities or binary labels.",
        min_length=1,
    )
    window_size: int = Field(
        default=1000, ge=10, description="Ёмкость скользящего окна / Sliding window capacity."
    )
    min_reference_size: int = Field(
        default=200, ge=10, description="Минимум наблюдений для установки reference."
    )


class SetReferenceRequest(BaseModel):
    """Явная установка референсного окна."""

    predictions: list[float] = Field(
        ..., min_length=2, description="Базовые предсказания модели (после деплоя)."
    )


@app.post("/predictions/observe", status_code=201, tags=["prediction_monitor"])
def predictions_observe(req: ObserveRequest) -> JSONResponse:
    """
    Добавить предсказания в скользящее окно монитора.
    Submit predictions to the sliding monitoring window.

    Первые min_reference_size наблюдений автоматически формируют
    reference window. Последующие наблюдения заполняют current window.
    First min_reference_size observations auto-establish the reference.
    Subsequent observations populate the current window.
    """
    monitor = get_prediction_monitor(
        window_size=req.window_size,
        min_reference_size=req.min_reference_size,
    )
    n_added = monitor.observe(req.predictions)
    status = monitor.get_status()
    return JSONResponse(
        status_code=201,
        content={
            "n_added": n_added,
            "status": status.to_dict(),
        },
    )


@app.post("/predictions/reference", status_code=201, tags=["prediction_monitor"])
def predictions_set_reference(req: SetReferenceRequest) -> JSONResponse:
    """
    Явно установить референсное окно (например, при ротации версии модели).
    Explicitly set reference window (e.g., on model version rotation).

    Сбрасывает current window и устанавливает новый reference.
    Resets current window and sets new reference (champion/challenger swap).
    """
    monitor = get_prediction_monitor()
    ref_stats = monitor.set_reference(req.predictions)
    return JSONResponse(
        status_code=201,
        content={
            "message": "Reference window set.",
            "reference_stats": ref_stats.to_dict(),
        },
    )


@app.get("/predictions/drift", tags=["prediction_monitor"])
def predictions_detect_drift() -> JSONResponse:
    """
    Сравнить текущее окно с референсным и вернуть результат дрейфа.
    Compare current window with reference and return drift result.

    PSI ≥ 0.2 → critical drift, PSI ≥ 0.1 → warning, < 0.1 → ok.
    Также вычисляется Welch z-test на разность средних и delta positive_rate.
    """
    monitor = get_prediction_monitor()
    try:
        result = monitor.detect_drift()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=result.to_dict())


@app.get("/predictions/status", tags=["prediction_monitor"])
def predictions_status() -> JSONResponse:
    """
    Текущее состояние монитора (размеры окон, is_ready).
    Current monitor state (window sizes, is_ready flag).
    """
    monitor = get_prediction_monitor()
    return JSONResponse(content=monitor.get_status().to_dict())


@app.post("/predictions/reset", tags=["prediction_monitor"])
def predictions_reset() -> JSONResponse:
    """
    Сбросить монитор предсказаний (новая версия модели или тест).
    Reset prediction monitor (new model version or test isolation).
    """
    reset_prediction_monitor()
    return JSONResponse(content={"message": "Prediction monitor reset."})
