"""FastAPI endpoint for anomaly detection with Prometheus metrics and MMD drift detection."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..metrics.prometheus_exporter import AnomalyMetrics, is_available
from ..models.cusum import CUSUMConfig, CUSUMDetector
from ..models.detector import MultiMetricDetector
from ..models.ensemble import STRATEGIES, AnomalyEnsemble, DetectorVote, EnsembleConfig
from ..models.isolation import IsolationConfig, IsolationForestDetector
from ..models.isolation import is_available as isolation_available
from ..models.kalman import KalmanConfig, KalmanDetector
from ..models.lstm_autoencoder import LSTMConfig, create_autoencoder
from ..models.stl import STLConfig, STLDetector

app = FastAPI(
    title="Realtime Anomaly Detection API",
    description="Detect anomalies in metric time series with Prometheus observability, "
    "MMD drift detection, LSTM/ESN autoencoder serving, Isolation Forest with "
    "feature-level explainability, and CUSUM sequential change detection",
    version="6.0.0",
)

detector = MultiMetricDetector(window_size=50, threshold_sigma=3.0)

# Инициализируем метрики только если prometheus_client доступен.
# Graceful degradation: без либы API работает в прежнем режиме.
_metrics: AnomalyMetrics | None = None
if is_available():
    _metrics = AnomalyMetrics()
    _metrics.set_model_config(
        threshold_sigma=detector.detector.threshold_sigma,
        window_size=detector.detector.window_size,
    )

# Глобальный drift trigger — инициализируется lazy при первом вызове /drift/check.
# Lazy init нужен чтобы bootstrap не замедлял старт сервиса.
_drift_trigger = None
_last_drift_result: dict | None = None
_DRIFT_FEATURES = ["cpu", "latency", "requests"]

# LSTM/ESN автоэнкодер — инициализируется при /lstm/train.
# Глобальный стейт: один model per service (production pattern).
_lstm_model = create_autoencoder()
_lstm_train_result: dict | None = None

# Isolation Forest — инициализируется при /isolation/train.
_isolation_model = IsolationForestDetector()
_isolation_train_result: dict | None = None

# CUSUM детектор — инициализируется при /cusum/calibrate.
# Один детектор на сервис (production singleton pattern).
_cusum_detector = CUSUMDetector()


def _reset_cusum() -> None:
    """Пересоздать детектор — используется в тестах для изоляции."""
    global _cusum_detector
    _cusum_detector = CUSUMDetector()


# Kalman Filter детектор — инициализируется при /kalman/calibrate.
_kalman_detector = KalmanDetector()


def _reset_kalman() -> None:
    """Пересоздать Kalman детектор — используется в тестах для изоляции."""
    global _kalman_detector
    _kalman_detector = KalmanDetector()


# STL Seasonal Decomposition детектор — инициализируется при /stl/calibrate.
_stl_detector = STLDetector()


def _reset_stl() -> None:
    """Пересоздать STL детектор — используется в тестах для изоляции."""
    global _stl_detector
    _stl_detector = STLDetector()


class MetricPoint(BaseModel):
    timestamp: float
    cpu: float = Field(..., ge=0, le=100)
    latency: float = Field(..., ge=0)
    requests: float = Field(..., ge=0)


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float


class DriftCheckRequest(BaseModel):
    """Запрос на проверку дрейфа данных через MMD.

    reference: Эталонные данные — матрица [cpu, latency, requests] за «нормальный» период.
    current: Текущие данные — тот же формат за мониторируемый период.

    Формат: список точек, каждая точка = [cpu, latency, requests].
    """

    reference: list[list[float]] = Field(
        ...,
        description="Reference (training) data: list of [cpu, latency, requests] points",
        min_length=10,
    )
    current: list[list[float]] = Field(
        ...,
        description="Current (production) data: list of [cpu, latency, requests] points",
        min_length=5,
    )
    alpha: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Significance level for MMD bootstrap threshold",
    )
    n_bootstrap: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Bootstrap iterations (more = more accurate threshold)",
    )


class DriftCheckResponse(BaseModel):
    """Ответ MMD drift detection."""

    mmd_statistic: float
    threshold: float
    is_drift: bool
    p_value: float
    gamma: float
    reference_size: int
    current_size: int
    features: list[str]
    audit_id: str
    timestamp: str
    reason: str
    should_retrain: bool
    triggered_by: str


@app.get("/health")
def health() -> dict:
    """Health check + краткая статистика метрик."""
    response: dict = {"status": "healthy", "prometheus": is_available()}
    if _metrics is not None:
        response["stats"] = _metrics.get_summary()
    if _last_drift_result is not None:
        response["last_drift"] = {
            "is_drift": _last_drift_result["is_drift"],
            "mmd_statistic": _last_drift_result["mmd_statistic"],
            "timestamp": _last_drift_result["timestamp"],
        }
    response["isolation_fitted"] = _isolation_model.is_fitted
    return response


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """
    Prometheus /metrics endpoint.

    Возвращает метрики в формате text exposition (совместим с Prometheus scrape).
    Если prometheus_client не установлен — возвращает пустую строку.
    """
    if _metrics is None:
        return "# prometheus_client not available\n"
    return _metrics.get_metrics_text()


@app.post("/detect", response_model=list[AnomalyResponse])
def detect_anomalies(points: list[MetricPoint]) -> list[AnomalyResponse]:
    """Detect anomalies in a batch of metric points."""
    data = {
        "cpu": np.array([p.cpu for p in points]),
        "latency": np.array([p.latency for p in points]),
        "requests": np.array([p.requests for p in points]),
    }

    # Измеряем latency детекции и обновляем Prometheus-метрики
    if _metrics is not None:
        with _metrics.track_detection():
            result = detector.detect(data)
    else:
        result = detector.detect(data)

    responses = [
        AnomalyResponse(
            is_anomaly=bool(result.predictions[i]),
            score=float(result.scores[i]),
            threshold=result.threshold,
        )
        for i in range(len(points))
    ]

    # Записываем метрики запроса: кол-во аномалий + их Z-score
    if _metrics is not None:
        anomaly_scores = [
            float(result.scores[i]) for i in range(len(points)) if result.predictions[i]
        ]
        _metrics.record_request(
            n_points=len(points),
            n_anomalies=len(anomaly_scores),
            scores=anomaly_scores,
        )

    return responses


@app.post("/drift/check", response_model=DriftCheckResponse)
def check_drift(request: DriftCheckRequest) -> DriftCheckResponse:
    """Проверить дрейф данных между эталонной и текущей выборкой через MMD.

    MMD (Maximum Mean Discrepancy) — многомерный непараметрический тест.
    Анализирует совместное распределение [cpu, latency, requests], ловит
    дрейф в корреляциях между метриками (чего PSI не умеет).

    Bootstrap-порог гарантирует контроль ошибки I рода на уровне alpha.

    Возвращает audit_id для сквозной трассировки (EU AI Act compliance).
    """
    global _drift_trigger, _last_drift_result

    from ..retraining.trigger import AnomalyRetrainingTrigger

    reference = np.array(request.reference)
    current = np.array(request.current)

    # Lazy-инициализация триггера с переданными эталонными данными.
    # Пересоздаём если эталон изменился (разный размер = новый reference период).
    # Это правильно: в production reference обновляется после каждого retraining.
    _drift_trigger = AnomalyRetrainingTrigger(
        reference_data=reference,
        features=_DRIFT_FEATURES,
        alpha=request.alpha,
        n_bootstrap=request.n_bootstrap,
    )

    retrain_result = _drift_trigger.evaluate(current)
    drift = retrain_result.drift_result

    # Сохраняем последний результат для /health endpoint
    _last_drift_result = {
        "is_drift": drift.is_drift,
        "mmd_statistic": drift.mmd_statistic,
        "timestamp": drift.timestamp,
        "audit_id": drift.audit_id,
    }

    return DriftCheckResponse(
        mmd_statistic=drift.mmd_statistic,
        threshold=drift.threshold,
        is_drift=drift.is_drift,
        p_value=drift.p_value,
        gamma=drift.gamma,
        reference_size=drift.reference_size,
        current_size=drift.current_size,
        features=drift.features,
        audit_id=drift.audit_id,
        timestamp=drift.timestamp,
        reason=drift.reason,
        should_retrain=retrain_result.should_retrain,
        triggered_by=retrain_result.triggered_by,
    )


@app.get("/drift/status")
def drift_status() -> dict:
    """Вернуть статус последней проверки дрейфа.

    Полезно для Grafana/Prometheus alerting: опросить статус
    без повторного вычисления MMD.
    """
    if _last_drift_result is None:
        return {
            "status": "no_check_performed",
            "message": "Call POST /drift/check first to initialize drift monitoring",
        }

    return {
        "status": "drift_detected" if _last_drift_result["is_drift"] else "stable",
        **_last_drift_result,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LSTM / ESN Autoencoder endpoints
# ──────────────────────────────────────────────────────────────────────────────


class LSTMTrainRequest(BaseModel):
    """Обучить ESN-автоэнкодер на нормальных данных.

    data: временной ряд нормального поведения — список точек [cpu, latency, requests].
    Рекомендуется минимум 200 точек (~30 мин при 10s-интервале).
    """

    data: list[list[float]] = Field(
        ...,
        description="Normal time series: list of [cpu, latency, requests] points",
        min_length=50,
    )
    reservoir_size: int = Field(default=100, ge=20, le=500)
    window_size: int = Field(default=30, ge=10, le=100)
    anomaly_percentile: float = Field(default=95.0, ge=80.0, le=99.9)


class LSTMTrainResponse(BaseModel):
    n_samples: int
    n_windows: int
    train_mse: float
    threshold: float
    model_config_: dict = Field(alias="model_config")

    model_config = {"populate_by_name": True}


class LSTMDetectRequest(BaseModel):
    """Обнаружить аномалии через ESN-автоэнкодер.

    data: список точек [cpu, latency, requests] для анализа.
    """

    data: list[list[float]] = Field(
        ...,
        description="Time series to analyze: list of [cpu, latency, requests] points",
        min_length=2,
    )


class LSTMDetectResponse(BaseModel):
    scores: list[float]
    predictions: list[int]
    threshold: float
    n_anomalies: int
    anomaly_rate: float
    model: str


@app.post("/lstm/train", response_model=LSTMTrainResponse)
def lstm_train(request: LSTMTrainRequest) -> LSTMTrainResponse:
    """Обучить ESN-автоэнкодер на нормальных данных.

    Echo State Network захватывает сложные временные паттерны (кросс-метрические
    корреляции, burst-recovery циклы), которые Z-score пропускает.

    Результат: порог аномальности (percentile reconstruction error на train data).
    """
    global _lstm_model, _lstm_train_result

    cfg = LSTMConfig(
        reservoir_size=request.reservoir_size,
        window_size=request.window_size,
        anomaly_percentile=request.anomaly_percentile,
    )
    _lstm_model = create_autoencoder(cfg)

    X = np.array(request.data, dtype=float)
    result = _lstm_model.fit(X)

    _lstm_train_result = {
        "n_samples": result.n_samples,
        "n_windows": result.n_windows,
        "train_mse": result.train_mse,
        "threshold": result.threshold,
    }

    return LSTMTrainResponse(
        n_samples=result.n_samples,
        n_windows=result.n_windows,
        train_mse=result.train_mse,
        threshold=result.threshold,
        model_config=_lstm_model.get_config(),
    )


@app.post("/lstm/detect", response_model=LSTMDetectResponse)
def lstm_detect(request: LSTMDetectRequest) -> LSTMDetectResponse:
    """Обнаружить аномалии через ESN-автоэнкодер.

    Требует предварительного обучения: POST /lstm/train.
    Возвращает reconstruction error score для каждой точки + бинарные предсказания.
    """
    if not _lstm_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Call POST /lstm/train with normal data first.",
        )

    X = np.array(request.data, dtype=float)
    result = _lstm_model.detect(X)

    n_anomalies = int(result.predictions.sum())
    anomaly_rate = n_anomalies / len(result.predictions) if len(result.predictions) > 0 else 0.0

    return LSTMDetectResponse(
        scores=[float(s) for s in result.scores],
        predictions=[int(p) for p in result.predictions],
        threshold=result.threshold,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        model=result.model,
    )


@app.get("/lstm/status")
def lstm_status() -> dict:
    """Статус ESN-модели: обучена / не обучена, конфигурация, метрики.

    Полезно для health-check перед inference.
    """
    config = _lstm_model.get_config()
    status = {
        "fitted": _lstm_model.is_fitted,
        "model_config": config,
    }
    if _lstm_train_result is not None:
        status["train_metrics"] = _lstm_train_result
    else:
        status["message"] = "Model not trained. Call POST /lstm/train to initialize."
    return status


# ──────────────────────────────────────────────────────────────────────────────
# Isolation Forest endpoints
# ──────────────────────────────────────────────────────────────────────────────


class IsolationTrainRequest(BaseModel):
    """Обучить Isolation Forest на нормальных данных.

    data: список точек [cpu, latency, requests] — нормальное поведение инфраструктуры.
    Рекомендуется минимум 100 точек для надёжного построения деревьев.
    contamination: ожидаемая доля аномалий в production (0.01–0.20).
    n_estimators: количество деревьев изоляции (больше = стабильнее, медленнее).
    """

    data: list[list[float]] = Field(
        ...,
        description="Normal time series: list of [cpu, latency, requests] points",
        min_length=20,
    )
    contamination: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Expected fraction of anomalies in production data",
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of isolation trees",
    )


class IsolationTrainResponse(BaseModel):
    n_samples: int
    n_features: int
    contamination: float
    n_trees: int
    avg_path_length_normal: float
    sklearn_available: bool


class IsolationPointResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    path_length: float
    feature_contributions: dict[str, float]
    top_feature: str


class IsolationDetectRequest(BaseModel):
    """Обнаружить аномалии через Isolation Forest.

    data: список точек [cpu, latency, requests] для анализа.
    Требует предварительного обучения: POST /isolation/train.
    """

    data: list[list[float]] = Field(
        ...,
        description="Metric points to analyze: list of [cpu, latency, requests]",
        min_length=1,
    )


class IsolationDetectResponse(BaseModel):
    results: list[IsolationPointResult]
    n_anomalies: int
    anomaly_rate: float
    top_anomalous_feature: str | None


@app.post("/isolation/train", response_model=IsolationTrainResponse)
def isolation_train(request: IsolationTrainRequest) -> IsolationTrainResponse:
    """Обучить Isolation Forest на нормальных метриках инфраструктуры.

    Isolation Forest строит ансамбль деревьев случайных разбиений.
    Аномальные точки изолируются быстрее (меньше разбиений = короче путь).
    Преимущество над Z-score: ловит многомерные аномалии (CPU spike + latency
    spike одновременно), где унивариатный Z-score пропускает паттерн.

    Требуется sklearn; graceful degradation: 503 если недоступен.
    """
    global _isolation_model, _isolation_train_result

    if not isolation_available():
        raise HTTPException(
            status_code=503,
            detail="scikit-learn not available; cannot train IsolationForest",
        )

    cfg = IsolationConfig(
        n_estimators=request.n_estimators,
        contamination=request.contamination,
    )
    _isolation_model = IsolationForestDetector(cfg)

    X = np.array(request.data, dtype=float)
    result = _isolation_model.fit(X)

    _isolation_train_result = {
        "n_samples": result.n_samples,
        "n_features": result.n_features,
        "contamination": result.contamination,
        "n_trees": result.n_trees,
        "avg_path_length_normal": result.avg_path_length_normal,
    }

    return IsolationTrainResponse(
        n_samples=result.n_samples,
        n_features=result.n_features,
        contamination=result.contamination,
        n_trees=result.n_trees,
        avg_path_length_normal=result.avg_path_length_normal,
        sklearn_available=True,
    )


@app.post("/isolation/detect", response_model=IsolationDetectResponse)
def isolation_detect(request: IsolationDetectRequest) -> IsolationDetectResponse:
    """Обнаружить аномалии и объяснить вклад каждого признака.

    Возвращает для каждой точки:
    - is_anomaly: флаг аномалии
    - anomaly_score: нормализованная оценка [0, 1]
    - feature_contributions: вклад каждого признака (сумма = 1)
    - top_feature: главная причина аномалии (важно для SRE diagonistics)

    Требует предварительного обучения: POST /isolation/train.
    """
    if not _isolation_model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Call POST /isolation/train with normal data first.",
        )

    X = np.array(request.data, dtype=float)
    results = _isolation_model.detect(X)

    point_results = [
        IsolationPointResult(
            is_anomaly=r.is_anomaly,
            anomaly_score=r.anomaly_score,
            path_length=r.path_length,
            feature_contributions=r.feature_contributions,
            top_feature=r.top_feature,
        )
        for r in results
    ]

    n_anomalies = sum(1 for r in results if r.is_anomaly)
    anomaly_rate = n_anomalies / len(results) if results else 0.0

    # Наиболее часто встречающийся top_feature среди аномалий
    anomaly_features = [r.top_feature for r in results if r.is_anomaly]
    top_anomalous_feature: str | None = None
    if anomaly_features:
        top_anomalous_feature = max(set(anomaly_features), key=anomaly_features.count)

    return IsolationDetectResponse(
        results=point_results,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
        top_anomalous_feature=top_anomalous_feature,
    )


@app.get("/isolation/status")
def isolation_status() -> dict:
    """Статус Isolation Forest: обучена / не обучена, sklearn доступен.

    Полезно для health-check и мониторинга состояния модели.
    """
    status: dict = {
        "fitted": _isolation_model.is_fitted,
        "sklearn_available": isolation_available(),
    }
    if _isolation_train_result is not None:
        status["train_metrics"] = _isolation_train_result
    else:
        status["message"] = "Model not trained. Call POST /isolation/train to initialize."
    return status


# ──────────────────────────────────────────────────────────────────────────────
# CUSUM Change Detection endpoints
# ──────────────────────────────────────────────────────────────────────────────


class CUSUMCalibrateRequest(BaseModel):
    """Откалибровать CUSUM на нормальных данных (оценить μ₀ и σ₀).

    data: одномерный ряд значений в нормальном режиме (без аномалий).
    Рекомендуется: не менее 50 точек для стабильной оценки.
    k: slack параметр — 0.5σ для обнаружения сдвига в 1σ (стандарт).
    h: порог решения — 5 (ARL₀ ≈ 465, т.е. одна ложная тревога на ~500 норм. точек).
    """

    data: list[float] = Field(
        ...,
        description="Univariate normal-regime time series for calibration",
        min_length=10,
    )
    k: float = Field(default=0.5, ge=0.1, le=2.0, description="Slack parameter")
    h: float = Field(default=5.0, ge=1.0, le=20.0, description="Decision threshold")


class CUSUMCalibrateResponse(BaseModel):
    mu_ref: float
    sigma_ref: float
    n_calibration: int
    k: float
    h: float


class CUSUMDetectRequest(BaseModel):
    """Батч-детекция CUSUM по временному ряду.

    data: одномерный ряд значений (может содержать аномалии).
    Требует предварительной калибровки: POST /cusum/calibrate.
    """

    data: list[float] = Field(
        ...,
        description="Univariate time series to scan for change points",
        min_length=1,
    )


class CUSUMDetectResponse(BaseModel):
    s_pos: list[float]
    s_neg: list[float]
    predictions: list[int]
    change_points: list[int]
    mu_ref: float
    sigma_ref: float
    threshold_k: float
    threshold_h: float
    n_alerts: int


class CUSUMUpdateRequest(BaseModel):
    """Онлайн-обновление CUSUM одной точкой (streaming mode).

    value: новое наблюдение метрики.
    Возвращает текущие S⁺/S⁻ и флаг тревоги. Состояние сохраняется между вызовами.
    """

    value: float = Field(..., description="Single new observation")


class CUSUMUpdateResponse(BaseModel):
    s_pos: float
    s_neg: float
    is_alert: bool
    n_updates: int
    n_alerts: int


@app.post("/cusum/calibrate", response_model=CUSUMCalibrateResponse)
def cusum_calibrate(request: CUSUMCalibrateRequest) -> CUSUMCalibrateResponse:
    """Откалибровать CUSUM детектор на нормальных данных.

    Оценивает μ₀ и σ₀ по выборке нормального режима.
    Сбрасывает накопленные S⁺ и S⁻ в 0 (готов к новому мониторингу).

    CUSUM улавливает персистентный сдвиг, который Z-score игнорирует:
    если CPU каждые 5 минут растёт на 0.5σ, через 10 точек S⁺ > h.
    Z-score в этот момент показывает лишь «нормальные» 0.5σ всплески.
    """
    _reset_cusum()
    cfg = CUSUMConfig(k=request.k, h=request.h)
    global _cusum_detector
    _cusum_detector = CUSUMDetector(cfg)

    data = np.array(request.data, dtype=float)
    result = _cusum_detector.calibrate(data)

    return CUSUMCalibrateResponse(
        mu_ref=result.mu_ref,
        sigma_ref=result.sigma_ref,
        n_calibration=result.n_calibration,
        k=result.k,
        h=result.h,
    )


@app.post("/cusum/detect", response_model=CUSUMDetectResponse)
def cusum_detect(request: CUSUMDetectRequest) -> CUSUMDetectResponse:
    """Батч-детекция точек смены режима в временном ряду.

    Self-resetting: после каждой тревоги CUSUM сбрасывается,
    что позволяет обнаружить несколько смен в одном вызове.

    change_points: индексы, где произошло обнаружение смены.
    s_pos / s_neg: полный путь статистик — полезно для визуализации в Grafana.

    Требует предварительной калибровки: POST /cusum/calibrate.
    """
    if not _cusum_detector.is_calibrated:
        raise HTTPException(
            status_code=400,
            detail="CUSUM not calibrated. Call POST /cusum/calibrate with normal data first.",
        )

    data = np.array(request.data, dtype=float)
    result = _cusum_detector.detect(data)

    return CUSUMDetectResponse(
        s_pos=result.s_pos,
        s_neg=result.s_neg,
        predictions=result.predictions,
        change_points=result.change_points,
        mu_ref=result.mu_ref,
        sigma_ref=result.sigma_ref,
        threshold_k=result.threshold_k,
        threshold_h=result.threshold_h,
        n_alerts=result.n_alerts,
    )


@app.post("/cusum/update", response_model=CUSUMUpdateResponse)
def cusum_update(request: CUSUMUpdateRequest) -> CUSUMUpdateResponse:
    """Онлайн-обновление CUSUM одной точкой (режим стриминга).

    Предназначен для real-time мониторинга: каждые N секунд новая метрика →
    немедленный ответ is_alert. Не требует хранения истории — весь контекст
    в S⁺ и S⁻. Идеален как Prometheus AlertManager hook.

    Требует предварительной калибровки: POST /cusum/calibrate.
    """
    if not _cusum_detector.is_calibrated:
        raise HTTPException(
            status_code=400,
            detail="CUSUM not calibrated. Call POST /cusum/calibrate with normal data first.",
        )

    result = _cusum_detector.update(request.value)

    return CUSUMUpdateResponse(
        s_pos=result.s_pos,
        s_neg=result.s_neg,
        is_alert=result.is_alert,
        n_updates=result.n_updates,
        n_alerts=result.n_alerts,
    )


@app.get("/cusum/status")
def cusum_status() -> dict:
    """Состояние CUSUM детектора: откалиброван ли, текущие S⁺/S⁻, счётчики.

    Полезно для Grafana dashboard: построить gauge S⁺ и S⁻,
    нарисовать горизонтальную линию h — визуальный proximity-to-alert.
    """
    state = _cusum_detector.get_state()
    result: dict = {
        "is_calibrated": state.is_calibrated,
        "s_pos": state.s_pos,
        "s_neg": state.s_neg,
        "n_updates": state.n_updates,
        "n_alerts": state.n_alerts,
    }
    if state.is_calibrated:
        result["mu_ref"] = state.mu_ref
        result["sigma_ref"] = state.sigma_ref
    else:
        result["message"] = "Not calibrated. Call POST /cusum/calibrate first."
    return result


# ---------------------------------------------------------------------------
# Kalman Filter endpoints
# ---------------------------------------------------------------------------


class KalmanCalibrateRequest(BaseModel):
    """Запрос на калибровку Kalman Filter из нормальных данных."""

    data: list[float] = Field(..., min_length=10, description="Normal (anomaly-free) time series")
    process_noise_level: float = Field(default=1e-3, gt=0)
    process_noise_trend: float = Field(default=1e-5, gt=0)
    measurement_noise: float | None = Field(default=None, gt=0)
    anomaly_alpha: float = Field(default=0.01, ge=0.001, le=0.10)


class KalmanCalibrateResponse(BaseModel):
    estimated_r: float
    n_samples: int
    initial_level: float
    initial_trend: float
    threshold_nis: float
    anomaly_alpha: float


class KalmanDetectRequest(BaseModel):
    data: list[float] = Field(..., min_length=1, description="Time series to evaluate")


class KalmanDetectResponse(BaseModel):
    levels: list[float]
    trends: list[float]
    predicted: list[float]
    innovations: list[float]
    nis_scores: list[float]
    predictions: list[bool]
    threshold: float
    anomaly_indices: list[int]
    n_anomalies: int


class KalmanUpdateRequest(BaseModel):
    value: float


class KalmanUpdateResponse(BaseModel):
    level: float
    trend: float
    predicted: float
    innovation: float
    nis: float
    threshold: float
    is_anomaly: bool
    n_updates: int


@app.post("/kalman/calibrate", response_model=KalmanCalibrateResponse)
def kalman_calibrate(req: KalmanCalibrateRequest) -> KalmanCalibrateResponse:
    """Откалибровать Kalman Filter на нормальных данных.

    Оценивает дисперсию шума наблюдения R из детрендированного ряда,
    инициализирует state [level, trend] и устанавливает NIS-порог
    из χ²-таблицы (df=1) для заданного уровня значимости alpha.

    После калибровки используйте POST /kalman/detect или /kalman/update.
    """
    config = KalmanConfig(
        process_noise_level=req.process_noise_level,
        process_noise_trend=req.process_noise_trend,
        measurement_noise=req.measurement_noise,
        anomaly_alpha=req.anomaly_alpha,
    )
    _kalman_detector.config = config
    result = _kalman_detector.calibrate(req.data)
    return KalmanCalibrateResponse(
        estimated_r=result.estimated_R,
        n_samples=result.n_samples,
        initial_level=result.initial_level,
        initial_trend=result.initial_trend,
        threshold_nis=result.threshold_nis,
        anomaly_alpha=req.anomaly_alpha,
    )


@app.post("/kalman/detect", response_model=KalmanDetectResponse)
def kalman_detect(req: KalmanDetectRequest) -> KalmanDetectResponse:
    """Батч-детекция аномалий через Kalman Filter.

    Процессирует ряд последовательно, обновляя внутреннее состояние фильтра.
    NIS > threshold → аномалия (статистически значимое отклонение от предсказания).

    Возвращает сглаженные level/trend для каждой точки — полезно для визуализации
    на дашборде рядом с raw данными (Grafana overlay).

    400 если фильтр не откалиброван.
    """
    if not _kalman_detector._is_calibrated:
        raise HTTPException(
            status_code=400,
            detail="Kalman filter not calibrated. Call POST /kalman/calibrate first.",
        )
    result = _kalman_detector.detect(req.data)
    return KalmanDetectResponse(
        levels=result.levels,
        trends=result.trends,
        predicted=result.predicted,
        innovations=result.innovations,
        nis_scores=result.nis_scores,
        predictions=result.predictions,
        threshold=result.threshold,
        anomaly_indices=result.anomaly_indices,
        n_anomalies=result.n_anomalies,
    )


@app.post("/kalman/update", response_model=KalmanUpdateResponse)
def kalman_update(req: KalmanUpdateRequest) -> KalmanUpdateResponse:
    """Онлайн-обновление Kalman Filter одной точкой (streaming mode).

    Идеально для интеграции с Prometheus: вызывать при каждом scrape (каждые 15 с).
    Возвращает NIS-score и флаг аномалии для немедленного алертинга.

    400 если фильтр не откалиброван.
    """
    if not _kalman_detector._is_calibrated:
        raise HTTPException(
            status_code=400,
            detail="Kalman filter not calibrated. Call POST /kalman/calibrate first.",
        )
    result = _kalman_detector.update(req.value)
    return KalmanUpdateResponse(
        level=result.level,
        trend=result.trend,
        predicted=result.predicted,
        innovation=result.innovation,
        nis=result.nis,
        threshold=result.threshold,
        is_anomaly=result.is_anomaly,
        n_updates=result.n_updates,
    )


@app.get("/kalman/status")
def kalman_status() -> dict:
    """Текущее состояние Kalman Filter для Grafana/health-check.

    level и trend — сглаженные оценки для overlay на метриках.
    threshold_nis — порог χ²(1) для аномалии.
    """
    state = _kalman_detector.get_state()
    result: dict = {
        "is_calibrated": state["is_calibrated"],
        "n_updates": state["n_updates"],
    }
    if state["is_calibrated"]:
        result["level"] = state["level"]
        result["trend"] = state["trend"]
        result["measurement_noise_R"] = state["measurement_noise_R"]
        result["threshold_nis"] = state["threshold_nis"]
    else:
        result["message"] = "Not calibrated. Call POST /kalman/calibrate first."
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble Voting endpoints
# ──────────────────────────────────────────────────────────────────────────────

# Глобальный ансамбль — конфигурируется через /ensemble/configure.
_ensemble = AnomalyEnsemble()


def _reset_ensemble() -> None:
    """Пересоздать ансамбль с дефолтной конфигурацией — используется в тестах."""
    global _ensemble
    _ensemble = AnomalyEnsemble()


class VoteInput(BaseModel):
    """Голос одного детектора для передачи в ансамбль."""

    name: str = Field(..., description="Detector name, e.g. 'cusum', 'kalman'")
    is_anomaly: bool = Field(..., description="Binary anomaly decision")
    score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Anomaly score in [0, 1] (used by weighted strategy)",
    )


class EnsembleVoteRequest(BaseModel):
    """Запрос на ансамблевое голосование.

    votes: список голосов от отдельных детекторов
    strategy: стратегия агрегации (переопределяет глобальную конфигурацию)
    weights: веса детекторов для weighted-стратегии (опционально)
    min_agreement: порог для majority/weighted (опционально)
    """

    votes: list[VoteInput] = Field(..., min_length=1, description="Detector votes to aggregate")
    strategy: str = Field(default="majority", description="majority|weighted|any|all")
    weights: dict[str, float] | None = Field(
        default=None,
        description="Per-detector weights for weighted strategy",
    )
    min_agreement: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agreement threshold for majority/weighted strategies",
    )


class EnsembleVoteResponse(BaseModel):
    """Результат ансамблевого голосования."""

    is_anomaly: bool
    confidence: float
    strategy: str
    agreement_ratio: float
    n_votes: int
    n_anomaly_votes: int
    votes: list[dict]


@app.post("/ensemble/vote", response_model=EnsembleVoteResponse)
def ensemble_vote(request: EnsembleVoteRequest) -> EnsembleVoteResponse:
    """Агрегировать голоса детекторов в итоговое решение об аномалии.

    Принимает список бинарных решений и скоров от нескольких детекторов
    (CUSUM, Kalman, Isolation Forest, ESN) и возвращает консолидированный вердикт.

    Стратегии:
    - majority: аномалия если > min_agreement доля детекторов согласна (default 50%)
    - weighted: взвешенное среднее score по weights; порог min_agreement
    - any: аномалия если хотя бы один детектор сигнализирует (высокая чувствительность)
    - all: аномалия только при консенсусе всех детекторов (низкий false alarm rate)

    422 если votes пуст или стратегия неизвестна.
    """
    config = EnsembleConfig(
        strategy=request.strategy,
        weights=request.weights,
        min_agreement=request.min_agreement,
    )
    ensemble = AnomalyEnsemble(config)

    detector_votes = [
        DetectorVote(name=v.name, is_anomaly=v.is_anomaly, score=v.score) for v in request.votes
    ]

    try:
        result = ensemble.aggregate(detector_votes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return EnsembleVoteResponse(
        is_anomaly=result.is_anomaly,
        confidence=result.confidence,
        strategy=result.strategy,
        agreement_ratio=result.agreement_ratio,
        n_votes=result.n_votes,
        n_anomaly_votes=result.n_anomaly_votes,
        votes=result.to_dict()["votes"],
    )


@app.get("/ensemble/strategies")
def ensemble_strategies() -> dict:
    """Описание доступных стратегий ансамблевого голосования.

    Помогает выбрать стратегию под требования системы:
    - safety-critical (не пропускать аномалии) → "any"
    - low false-alarm (не беспокоить SRE без причины) → "all"
    - balanced (production default) → "majority"
    - calibrated detectors (знаем AUC каждого) → "weighted"
    """
    return {
        "strategies": STRATEGIES,
        "recommendation": {
            "safety_critical": "any — минимальный miss rate",
            "low_false_alarm": "all — минимальный false alarm rate",
            "balanced_default": "majority — баланс precision/recall",
            "calibrated_detectors": "weighted — используйте AUC как веса",
        },
    }


# ---------------------------------------------------------------------------
# STL Seasonal Decomposition endpoints
# ---------------------------------------------------------------------------


class STLCalibrateRequest(BaseModel):
    """Калибровка STL-детектора на нормальных данных.

    normal_data: временной ряд одной метрики
        (например, CPU % за 1 неделю с шагом 1 час = 168 точек).
    period: длина сезонного цикла. 24 = часовые данные, суточный цикл.
    threshold_z: порог аномалии в MAD-sigma единицах.
    robust: True = MAD (устойчив к аномалиям в train), False = std.
    """

    normal_data: list[float] = Field(
        ...,
        min_length=2,
        description="Normal (non-anomalous) time series for calibration",
    )
    period: int = Field(
        default=24,
        ge=2,
        le=365,
        description="Seasonal period length in observations",
    )
    threshold_z: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Anomaly threshold in robust sigma units",
    )
    robust: bool = Field(default=True, description="Use MAD instead of std")


class STLCalibrateResponse(BaseModel):
    """Результат калибровки STL."""

    period: int
    n_samples: int
    n_complete_cycles: int
    mu_residual: float
    sigma_residual: float
    seasonal_pattern: list[float]


class STLDetectRequest(BaseModel):
    """Батч-детекция аномалий через STL-декомпозицию."""

    data: list[float] = Field(
        ...,
        min_length=2,
        description="Time series to decompose and score",
    )


class STLDetectResponse(BaseModel):
    """Результат STL батч-декомпозиции."""

    trend: list[float]
    seasonal: list[float]
    residual: list[float]
    anomaly_score: list[float]
    predictions: list[int]
    anomaly_indices: list[int]
    n_anomalies: int
    period: int
    threshold_z: float


class STLUpdateRequest(BaseModel):
    """Онлайн-обновление: одна новая точка."""

    value: float = Field(..., description="New observed metric value")


class STLUpdateResponse(BaseModel):
    """Результат онлайн-обновления одной точки."""

    value: float
    trend_estimate: float
    seasonal_estimate: float
    residual: float
    anomaly_score: float
    is_anomaly: bool
    n_updates: int


@app.post("/stl/calibrate", response_model=STLCalibrateResponse)
def stl_calibrate(request: STLCalibrateRequest) -> STLCalibrateResponse:
    """Откалибровать STL-детектор на нормальных данных.

    Декомпозирует временной ряд через Centered Moving Average и оценивает:
    - seasonal_pattern — усреднённый сезонный профиль (длина = period)
    - mu_residual, sigma_residual — статистики остатка для порога аномалий

    Сезонный паттерн используется как при батч-детекции, так и при онлайн-обновлении.

    Требует ≥ min_periods × period точек (по умолчанию ≥ 48 для period=24).
    422 если данных недостаточно.
    """
    global _stl_detector

    cfg = STLConfig(
        period=request.period,
        threshold_z=request.threshold_z,
        robust=request.robust,
    )
    _stl_detector = STLDetector(cfg)

    try:
        result = _stl_detector.calibrate(request.normal_data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return STLCalibrateResponse(
        period=result.period,
        n_samples=result.n_samples,
        n_complete_cycles=result.n_complete_cycles,
        mu_residual=result.mu_residual,
        sigma_residual=result.sigma_residual,
        seasonal_pattern=result.seasonal_pattern,
    )


@app.post("/stl/detect", response_model=STLDetectResponse)
def stl_detect(request: STLDetectRequest) -> STLDetectResponse:
    """Батч-декомпозиция и аномальные метки.

    Разбивает временной ряд на компоненты:
    - trend: низкочастотное движение (CMA сглаживание)
    - seasonal: повторяющийся паттерн (средний профиль по фазам)
    - residual: остаток = data - trend - seasonal

    Аномалии детектируются в residual (робастный Z-score > threshold_z).
    Сезонные пики и трендовый рост не вызывают ложных тревог.

    Требует calibrate() для корректных порогов; без калибровки оценивает паттерн
    из текущих данных (менее надёжно). Требует ≥ period точек.
    400 если данных меньше period.
    """
    try:
        result = _stl_detector.detect(request.data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return STLDetectResponse(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.residual,
        anomaly_score=result.anomaly_score,
        predictions=result.predictions,
        anomaly_indices=result.anomaly_indices,
        n_anomalies=result.n_anomalies,
        period=result.period,
        threshold_z=result.threshold_z,
    )


@app.post("/stl/update", response_model=STLUpdateResponse)
def stl_update(request: STLUpdateRequest) -> STLUpdateResponse:
    """Онлайн-обновление: одна новая метрика-точка → аномалия / норма.

    O(period) памяти — хранит только последний скользящий буфер.
    Сезонный паттерн берётся из калибровки (не пересчитывается).

    Использует локальный moving average последних period точек для оценки
    тренда, устраняя необходимость хранить всю историю.

    Требует предварительной калибровки: POST /stl/calibrate.
    400 если calibrate() не был вызван.
    """
    try:
        result = _stl_detector.update(request.value)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return STLUpdateResponse(
        value=result.value,
        trend_estimate=result.trend_estimate,
        seasonal_estimate=result.seasonal_estimate,
        residual=result.residual,
        anomaly_score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
        n_updates=result.n_updates,
    )


@app.get("/stl/status")
def stl_status() -> dict:
    """Состояние STL-детектора для health-check и Grafana gauge.

    Возвращает:
    - is_calibrated: готов ли детектор к обнаружению
    - period, threshold_z: параметры конфигурации
    - n_calibration: сколько точек использовалось для калибровки
    - n_updates: сколько онлайн-обновлений выполнено после калибровки
    - mu_residual, sigma_residual: порог в натуральных единицах метрики
    - seasonal_pattern: извлечённый сезонный профиль (для Grafana/visualization)
    """
    state = _stl_detector.get_state()
    return {
        "is_calibrated": state.is_calibrated,
        "period": state.period,
        "threshold_z": state.threshold_z,
        "n_calibration": state.n_calibration,
        "n_updates": state.n_updates,
        "mu_residual": state.mu_residual,
        "sigma_residual": state.sigma_residual,
        "seasonal_pattern": state.seasonal_pattern,
    }
