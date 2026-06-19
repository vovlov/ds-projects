"""FastAPI-сервис оценки стоимости недвижимости.

Эндпоинт /estimate принимает характеристики квартиры и возвращает:
- estimated_price — точечная оценка
- confidence_interval — диапазон (модель не идеальна, честно об этом говорим)
- top_factors — топ-5 признаков, повлиявших на цену (из SHAP)
- shap_waterfall — полный SHAP waterfall: базовое значение + вклад каждого признака

Эндпоинт /estimate/intervals возвращает статистически обоснованные интервалы:
- LightGBM quantile regression (pinball loss) для прямых квантильных оценок
- CQR (Romano et al. 2019): гарантированное покрытие ≥ 1-α без допущений о распределении
- 90% и 95% интервалы + метрики калибровки

SHAP waterfall — ключевое улучшение над классическими feature importances:
глобальная важность говорит "площадь важна в среднем", а SHAP говорит
"для ЭТОЙ квартиры район добавил +3.2М, а год постройки -400К".
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..forecast.price_forecast import (
    NEIGHBORHOOD_BASE_PRICES,
    ForecastConfig,
    HoltWintersForecaster,
    generate_price_history,
)
from ..models.comps import ComparableSearch, CompsConfig
from ..models.mortgage import (
    DEFAULT_EXPENSE_RATIO,
    MORTGAGE_PROGRAMS,
    NEIGHBORHOOD_RENT_RATES,
    MortgageCalculator,
    MortgageConfig,
    MortgageResult,
    RentalYieldResult,
)
from ..models.quantile import QuantileRegressionModel
from ..models.quantile import is_available as lgbm_available

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Price Estimation API",
    description="Оценка стоимости московской недвижимости на основе CatBoost",
    version="2.0.0",
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

_model = None
_results = None
_quantile_model: QuantileRegressionModel | None = None
_quantile_calibration: dict | None = None
_comps_search: ComparableSearch | None = None


def _load_artifacts() -> tuple:
    global _model, _results
    if _model is None:
        model_path = ARTIFACTS_DIR / "model.pkl"
        results_path = ARTIFACTS_DIR / "results.pkl"
        if not model_path.exists():
            raise HTTPException(500, "Model not found. Run train.py first.")
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        if results_path.exists():
            with open(results_path, "rb") as f:
                _results = pickle.load(f)
    return _model, _results


def _load_quantile_artifacts() -> QuantileRegressionModel:
    """Загрузить quantile модель из артефактов или поднять 503."""
    global _quantile_model, _quantile_calibration
    if _quantile_model is not None:
        return _quantile_model

    qmodel_path = ARTIFACTS_DIR / "quantile_model.pkl"
    if not qmodel_path.exists():
        raise HTTPException(
            503,
            detail="Quantile model not found. Run: python train.py --quantile",
        )
    with open(qmodel_path, "rb") as f:
        data = pickle.load(f)
    _quantile_model = data["model"]
    _quantile_calibration = data.get("calibration")
    return _quantile_model


class PropertyInput(BaseModel):
    """Характеристики квартиры для оценки."""

    sqft: int = Field(..., ge=15, le=500, description="Площадь, кв.м", examples=[65])
    bedrooms: int = Field(..., ge=1, le=10, description="Комнат", examples=[2])
    bathrooms: int = Field(..., ge=1, le=5, description="Санузлов", examples=[1])
    year_built: int = Field(..., ge=1900, le=2026, description="Год постройки", examples=[2015])
    lot_size: int = Field(..., ge=0, le=2000, description="Площадь участка, кв.м", examples=[0])
    garage: int = Field(..., ge=0, le=1, description="Гараж (0/1)", examples=[1])
    neighborhood: str = Field(..., description="Район Москвы", examples=["Хамовники"])
    condition: str = Field(..., description="Состояние", examples=["хорошее"])


class SHAPContribution(BaseModel):
    """Вклад одного признака в предсказание (одна полоска waterfall chart)."""

    feature: str = Field(..., description="Название признака")
    value: str | float | int = Field(..., description="Значение признака для данного объекта")
    contribution: float = Field(..., description="Вклад в рублях (+ повышает цену, - понижает)")
    direction: str = Field(..., description="positive | negative")


class SHAPWaterfall(BaseModel):
    """Полный SHAP waterfall: базовое значение + вклад каждого признака.

    base_value — это E[f(x)], средняя цена по обучающей выборке.
    prediction = base_value + sum(contributions).
    """

    base_value: float = Field(..., description="E[f(x)] — средняя оценка по датасету, руб")
    contributions: list[SHAPContribution] = Field(
        ..., description="Вклады признаков, отсортированы по |contribution|"
    )
    prediction: float = Field(..., description="base_value + sum(contributions), руб")


class PriceEstimate(BaseModel):
    """Результат оценки."""

    estimated_price: int = Field(..., description="Оценка стоимости, руб")
    confidence_low: int = Field(..., description="Нижняя граница, руб")
    confidence_high: int = Field(..., description="Верхняя граница, руб")
    top_factors: list[dict[str, str | float]] = Field(
        ..., description="Топ-5 факторов по |SHAP contribution|"
    )
    shap_waterfall: SHAPWaterfall | None = Field(
        None, description="SHAP waterfall для визуализации объяснения (None если SHAP недоступен)"
    )


class PriceIntervalEstimate(BaseModel):
    """Результат оценки с quantile regression интервалами предсказаний."""

    estimated_price: int = Field(..., description="Медианная оценка (q=0.5), руб")
    interval_90_low: int = Field(..., description="Нижняя граница 90% интервала, руб")
    interval_90_high: int = Field(..., description="Верхняя граница 90% интервала, руб")
    interval_95_low: int = Field(..., description="Нижняя граница 95% интервала, руб")
    interval_95_high: int = Field(..., description="Верхняя граница 95% интервала, руб")
    width_90: int = Field(..., description="Ширина 90% интервала — мера неопределённости, руб")
    width_95: int = Field(..., description="Ширина 95% интервала, руб")
    is_cqr_calibrated: bool = Field(..., description="Применена ли CQR калибровка")
    calibration_coverage_90: float | None = Field(
        None, description="Эмпирическое покрытие 90% на тестовой выборке"
    )
    calibration_coverage_95: float | None = Field(
        None, description="Эмпирическое покрытие 95% на тестовой выборке"
    )
    lgbm_available: bool = Field(..., description="Доступен ли LightGBM")


class ForecastRequest(BaseModel):
    """Запрос прогноза цен для района."""

    neighborhood: str = Field(..., description="Район Москвы", examples=["Хамовники"])
    n_months_history: int = Field(36, ge=12, le=120, description="Месяцев истории")
    forecast_periods: int = Field(12, ge=1, le=36, description="Горизонт прогноза, месяцев")
    seed: int | None = Field(None, description="Seed для воспроизводимости")


class ForecastPointResponse(BaseModel):
    """Одна точка прогноза."""

    period: int
    value: float
    lower: float
    upper: float


class PriceForecastResponse(BaseModel):
    """Прогноз цены руб/кв.м для района с доверительными интервалами."""

    neighborhood: str
    trend_direction: str
    trend_slope_monthly_pct: float
    mape: float
    last_known_price: float
    forecast: list[ForecastPointResponse]
    alpha: float
    beta: float


class TrendSummary(BaseModel):
    """Краткая сводка тренда района для дашборда."""

    neighborhood: str
    trend_direction: str
    trend_slope_monthly_pct: float
    current_price: float
    forecast_12m: float


@app.post("/forecast/price", response_model=PriceForecastResponse)
def forecast_price(req: ForecastRequest) -> PriceForecastResponse:
    """Прогноз цены руб/кв.м для района на forecast_periods месяцев вперёд.

    Использует Holt's Double Exponential Smoothing с оптимизацией α и β.
    """
    if req.neighborhood not in NEIGHBORHOOD_BASE_PRICES:
        raise HTTPException(
            400,
            detail=f"Unknown neighborhood '{req.neighborhood}'. "
            f"Available: {sorted(NEIGHBORHOOD_BASE_PRICES)}",
        )

    history = generate_price_history(req.neighborhood, req.n_months_history, req.seed)
    config = ForecastConfig(forecast_periods=req.forecast_periods)
    result = HoltWintersForecaster(config).fit(history).forecast()

    return PriceForecastResponse(
        neighborhood=req.neighborhood,
        trend_direction=result.trend_direction,
        trend_slope_monthly_pct=result.trend_slope_pct,
        mape=result.mape,
        last_known_price=result.last_known_value,
        forecast=[
            ForecastPointResponse(period=p.period, value=p.value, lower=p.lower, upper=p.upper)
            for p in result.forecast
        ],
        alpha=result.alpha,
        beta=result.beta,
    )


@app.get("/forecast/trends", response_model=list[TrendSummary])
def forecast_trends() -> list[TrendSummary]:
    """Сводка трендов по всем районам Москвы (seed=42 для воспроизводимости)."""
    summaries: list[TrendSummary] = []
    for neighborhood in sorted(NEIGHBORHOOD_BASE_PRICES):
        history = generate_price_history(neighborhood, n_months=36, seed=42)
        result = HoltWintersForecaster().fit(history).forecast(steps=12)
        summaries.append(
            TrendSummary(
                neighborhood=neighborhood,
                trend_direction=result.trend_direction,
                trend_slope_monthly_pct=result.trend_slope_pct,
                current_price=result.last_known_value,
                forecast_12m=round(result.forecast[-1].value, 2),
            )
        )
    return summaries


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "quantile_model_loaded": _quantile_model is not None,
        "lgbm_available": lgbm_available(),
    }


@app.post("/estimate", response_model=PriceEstimate)
def estimate(prop: PropertyInput) -> PriceEstimate:
    """Оценить стоимость квартиры с SHAP waterfall объяснением."""
    model, results = _load_artifacts()

    from ..data.load import CURRENT_YEAR

    # Собираем признаки в том же порядке, что при обучении
    age = CURRENT_YEAR - prop.year_built
    has_garage = "yes" if prop.garage == 1 else "no"

    # Значения признаков — нужны для отображения в SHAP waterfall
    feature_values: dict[str, str | float | int] = {
        "sqft": prop.sqft,
        "bedrooms": prop.bedrooms,
        "bathrooms": prop.bathrooms,
        "year_built": prop.year_built,
        "lot_size": prop.lot_size,
        "age": age,
        "neighborhood": prop.neighborhood,
        "condition": prop.condition,
        "has_garage": has_garage,
    }

    # CatBoost ожидает массив признаков в порядке MODEL_FEATURES
    # MODEL_FEATURES = [sqft, bedrooms, bathrooms, year_built, lot_size, age,
    #                   neighborhood, condition, has_garage]
    features = np.array(list(feature_values.values()), dtype=object)

    prediction = float(model.predict(features.reshape(1, -1))[0])
    estimated = max(int(round(prediction, -4)), 1_000_000)  # не ниже 1М

    # Доверительный интервал на основе MAPE модели
    mape = results.get("mape", 0.10) if results else 0.10
    margin = max(mape, 0.05)  # минимум 5%
    confidence_low = int(round(estimated * (1 - margin), -4))
    confidence_high = int(round(estimated * (1 + margin), -4))

    # SHAP waterfall: per-prediction объяснение через CatBoost built-in SHAP.
    # Это лучше глобальных feature importances: показывает вклад для конкретной квартиры.
    shap_waterfall = None
    feature_names: list[str] = results.get("feature_names", []) if results else []

    if feature_names:
        try:
            from ..models.explain import explain_prediction

            shap_result = explain_prediction(model, features, feature_names)
            contributions = [
                SHAPContribution(
                    feature=fname,
                    value=feature_values.get(fname, ""),
                    contribution=contrib,
                    direction="positive" if contrib >= 0 else "negative",
                )
                for fname, contrib in shap_result["contributions"].items()
            ]
            shap_waterfall = SHAPWaterfall(
                base_value=shap_result["bias"],
                contributions=contributions,
                prediction=shap_result["prediction"],
            )
        except Exception as e:
            # SHAP может упасть если модель не CatBoost или Pool несовместим.
            # Не прерываем ответ — waterfall просто будет None.
            logger.warning("SHAP explain failed: %s", e)

    # top_factors: из SHAP (если доступен) или из глобальных importances (fallback)
    top_factors: list[dict[str, str | float]] = []
    if shap_waterfall:
        for c in shap_waterfall.contributions[:5]:
            top_factors.append(
                {
                    "feature": c.feature,
                    "value": c.value,
                    "contribution": c.contribution,
                    "direction": c.direction,
                }
            )
    elif results and "feature_importances" in results:
        sorted_imp = sorted(
            results["feature_importances"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, importance in sorted_imp[:5]:
            top_factors.append({"feature": name, "importance": round(importance, 2)})

    return PriceEstimate(
        estimated_price=estimated,
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        top_factors=top_factors,
        shap_waterfall=shap_waterfall,
    )


def _build_feature_array(prop: PropertyInput) -> np.ndarray:
    """Собрать массив признаков из запроса (для quantile модели, label encoded)."""
    from ..data.load import CONDITION_MAP, CURRENT_YEAR, NEIGHBORHOODS

    age = CURRENT_YEAR - prop.year_built
    has_garage_code = 1 if prop.garage == 1 else 0

    neighborhood_names = sorted(NEIGHBORHOODS.keys())
    condition_names = sorted(CONDITION_MAP.keys())

    neighborhood_code = float(
        neighborhood_names.index(prop.neighborhood)
        if prop.neighborhood in neighborhood_names
        else 0
    )
    condition_code = float(
        condition_names.index(prop.condition) if prop.condition in condition_names else 0
    )

    return np.array(
        [
            float(prop.sqft),
            float(prop.bedrooms),
            float(prop.bathrooms),
            float(prop.year_built),
            float(prop.lot_size),
            float(age),
            neighborhood_code,
            condition_code,
            float(has_garage_code),
        ],
        dtype=np.float64,
    ).reshape(1, -1)


def _get_comps_search() -> ComparableSearch:
    """Lazy-init базы аналогов (1000 объектов, seed=42)."""
    global _comps_search
    if _comps_search is None:
        from ..data.load import generate_dataset

        df = generate_dataset(n_rows=1000, seed=42)
        records = df.to_dicts()
        # Добавляем bedrooms для кодирования (уже есть в датасете)
        _comps_search = ComparableSearch(CompsConfig()).fit(records)
    return _comps_search


def _reset_comps_search() -> None:
    """Сбросить кэш базы аналогов (для тестовой изоляции)."""
    global _comps_search
    _comps_search = None


class CompsRequest(BaseModel):
    """Запрос на поиск аналогов."""

    sqft: int = Field(..., ge=15, le=500, description="Площадь, кв.м")
    bedrooms: int = Field(..., ge=1, le=10, description="Комнат")
    year_built: int = Field(..., ge=1900, le=2026, description="Год постройки")
    neighborhood: str = Field(..., description="Район Москвы")
    condition: str = Field(..., description="Состояние квартиры")
    estimated_price: int | None = Field(
        None, ge=0, description="Оценочная цена для вычисления market_position"
    )
    n_comps: int = Field(5, ge=1, le=20, description="Число аналогов")


class ComparableItem(BaseModel):
    """Один объект-аналог."""

    sqft: int
    bedrooms: int
    year_built: int
    neighborhood: str
    condition: str
    price: int
    price_per_sqft: float
    similarity_score: float
    distance: float


class CompsResponse(BaseModel):
    """Результат поиска аналогов с рыночным позиционированием."""

    comparables: list[ComparableItem]
    subject_price: int | None
    median_comp_price: int
    mean_comp_price: int
    price_deviation_pct: float | None = Field(
        None, description="Отклонение от медианного аналога, % (None если цена не передана)"
    )
    market_position: str | None = Field(
        None,
        description="above_market | at_market | below_market (None если цена не передана)",
    )
    n_comparables: int


@app.post("/estimate/comps", response_model=CompsResponse)
def estimate_comps(req: CompsRequest) -> CompsResponse:
    """Найти K аналогичных объектов для обоснования оценки.

    Возвращает K ближайших объектов из синтетической базы московской недвижимости
    (1000 объектов, seed=42) по взвешенной нормализованной евклидовой дистанции.
    Если передана estimated_price, вычисляет market_position относительно медианы аналогов.
    """
    searcher = _get_comps_search()

    subject = {
        "sqft": req.sqft,
        "bedrooms": req.bedrooms,
        "year_built": req.year_built,
        "neighborhood": req.neighborhood,
        "condition": req.condition,
        "price": req.estimated_price,
    }

    result = searcher.find_comps(subject, n_comps=req.n_comps)

    return CompsResponse(
        comparables=[
            ComparableItem(
                sqft=c.sqft,
                bedrooms=c.bedrooms,
                year_built=c.year_built,
                neighborhood=c.neighborhood,
                condition=c.condition,
                price=c.price,
                price_per_sqft=c.price_per_sqft,
                similarity_score=c.similarity_score,
                distance=c.distance,
            )
            for c in result.comparables
        ],
        subject_price=result.subject_price,
        median_comp_price=result.median_comp_price,
        mean_comp_price=result.mean_comp_price,
        price_deviation_pct=result.price_deviation_pct,
        market_position=result.market_position,
        n_comparables=result.n_comparables,
    )


##############################################################################
# Mortgage & Rental Yield endpoints
##############################################################################


class MortgageRequest(BaseModel):
    """Запрос ипотечного калькулятора."""

    price: float = Field(..., gt=0, description="Цена квартиры, руб", examples=[12_000_000])
    annual_rate: float = Field(0.165, gt=0, lt=1.0, description="Годовая ставка (0.165 = 16.5%)")
    term_years: int = Field(20, ge=1, le=30, description="Срок кредита, лет")
    down_payment_ratio: float = Field(
        0.20, ge=0.10, le=0.90, description="Первоначальный взнос (0.20 = 20%)"
    )
    program: str = Field("standard", description="Название программы")


class MortgageResponse(BaseModel):
    """Результат ипотечного калькулятора."""

    loan_amount: float
    down_payment: float
    monthly_payment: float
    total_payment: float
    total_interest: float
    ltv_ratio: float
    effective_annual_rate: float
    n_payments: int
    program: str


class AffordabilityRequest(BaseModel):
    """Запрос оценки доступности ипотеки."""

    price: float = Field(..., gt=0, description="Цена квартиры, руб")
    annual_income: float = Field(
        ..., gt=0, description="Годовой доход заёмщика, руб", examples=[2_400_000]
    )
    annual_rate: float = Field(0.165, gt=0, lt=1.0, description="Годовая ставка")
    term_years: int = Field(20, ge=1, le=30, description="Срок кредита, лет")
    down_payment_ratio: float = Field(0.20, ge=0.10, le=0.90, description="Первоначальный взнос")


class AffordabilityResponse(BaseModel):
    """Оценка доступности ипотеки (NAR 28% / CFPB 43% правила)."""

    monthly_payment: float
    annual_income: float
    dti_mortgage_only: float
    is_affordable_28: bool
    is_affordable_43: bool
    recommended_income_annual: float
    stress_test_rate: float
    stress_test_payment: float


class RentalYieldRequest(BaseModel):
    """Запрос оценки доходности аренды."""

    price: float = Field(..., gt=0, description="Цена квартиры, руб")
    monthly_rent: float = Field(..., gt=0, description="Ежемесячная аренда, руб", examples=[85_000])
    expense_ratio: float = Field(
        DEFAULT_EXPENSE_RATIO,
        ge=0.0,
        le=0.5,
        description="Доля расходов от годовой аренды (дефолт 0.20 = НПД+простой+ремонт)",
    )


class RentalYieldResponse(BaseModel):
    """Доходность сдачи квартиры в аренду."""

    monthly_rent: float
    annual_rent: float
    gross_yield_pct: float
    net_yield_pct: float
    payback_years: float
    price_to_rent_ratio: float
    annual_expenses_estimated: float


class InvestmentRequest(BaseModel):
    """Запрос сводного инвестиционного анализа (аренда + ипотека)."""

    price: float = Field(..., gt=0, description="Цена квартиры, руб")
    monthly_rent: float = Field(..., gt=0, description="Ожидаемая аренда, руб")
    annual_rate: float = Field(0.165, gt=0, lt=1.0, description="Ставка ипотеки")
    term_years: int = Field(20, ge=1, le=30, description="Срок кредита, лет")
    down_payment_ratio: float = Field(0.20, ge=0.10, le=0.90, description="Первоначальный взнос")
    expense_ratio: float = Field(
        DEFAULT_EXPENSE_RATIO, ge=0.0, le=0.5, description="Доля расходов арендодателя"
    )


class InvestmentResponse(BaseModel):
    """Сводный инвестиционный анализ: вердикт + cashflow."""

    price: float
    mortgage: MortgageResponse
    rental: RentalYieldResponse
    monthly_cashflow: float
    is_cashflow_positive: bool
    down_payment_recovery_months: float
    investment_verdict: str


class NeighborhoodRentInfo(BaseModel):
    """Типичные ставки аренды и доходность для района."""

    neighborhood: str
    rent_rate_per_sqm: float
    example_sqft_65: float
    example_gross_yield_pct: float
    example_net_yield_pct: float


def _mortgage_to_response(r: MortgageResult) -> MortgageResponse:
    return MortgageResponse(
        loan_amount=r.loan_amount,
        down_payment=r.down_payment,
        monthly_payment=r.monthly_payment,
        total_payment=r.total_payment,
        total_interest=r.total_interest,
        ltv_ratio=r.ltv_ratio,
        effective_annual_rate=r.effective_annual_rate,
        n_payments=r.n_payments,
        program=r.program,
    )


def _rental_to_response(r: RentalYieldResult) -> RentalYieldResponse:
    return RentalYieldResponse(
        monthly_rent=r.monthly_rent,
        annual_rent=r.annual_rent,
        gross_yield_pct=r.gross_yield_pct,
        net_yield_pct=r.net_yield_pct,
        payback_years=r.payback_years,
        price_to_rent_ratio=r.price_to_rent_ratio,
        annual_expenses_estimated=r.annual_expenses_estimated,
    )


@app.post("/mortgage/calculate", response_model=MortgageResponse)
def mortgage_calculate(req: MortgageRequest) -> MortgageResponse:
    """Рассчитать ежемесячный платёж и параметры ипотечного кредита (аннуитет).

    Формула: M = P·r(1+r)^n / ((1+r)^n − 1),
    где P — тело кредита, r = annual_rate/12, n = term_years × 12.
    """
    config = MortgageConfig(
        annual_rate=req.annual_rate,
        term_years=req.term_years,
        down_payment_ratio=req.down_payment_ratio,
        program=req.program,
    )
    result = MortgageCalculator.compute_mortgage(req.price, config)
    return _mortgage_to_response(result)


@app.post("/mortgage/affordability", response_model=AffordabilityResponse)
def mortgage_affordability(req: AffordabilityRequest) -> AffordabilityResponse:
    """Оценить доступность ипотеки по доходу заёмщика.

    Правила:
      28% rule (NAR) — ипотека ≤ 28% месячного дохода (консервативный банковский стандарт).
      43% rule (CFPB) — ипотека ≤ 43% дохода (юридический предел Qualified Mortgage).
    Стресс-тест +2% — буфер по требованию ЦБ РФ / Базель III.
    """
    down = req.price * req.down_payment_ratio
    loan = req.price - down
    monthly = MortgageCalculator.compute_monthly_payment(loan, req.annual_rate, req.term_years)
    result = MortgageCalculator.compute_affordability(
        monthly_payment=monthly,
        annual_income=req.annual_income,
        annual_rate=req.annual_rate,
        term_years=req.term_years,
        loan_amount=loan,
    )
    return AffordabilityResponse(
        monthly_payment=result.monthly_payment,
        annual_income=result.annual_income,
        dti_mortgage_only=result.dti_mortgage_only,
        is_affordable_28=result.is_affordable_28,
        is_affordable_43=result.is_affordable_43,
        recommended_income_annual=result.recommended_income_annual,
        stress_test_rate=result.stress_test_rate,
        stress_test_payment=result.stress_test_payment,
    )


@app.get("/mortgage/programs")
def mortgage_programs() -> list[dict]:
    """Список доступных ипотечных программ России с актуальными ставками (2026-06)."""
    return [
        {
            "program_id": prog_id,
            **info,
        }
        for prog_id, info in MORTGAGE_PROGRAMS.items()
    ]


@app.post("/rental/yield", response_model=RentalYieldResponse)
def rental_yield(req: RentalYieldRequest) -> RentalYieldResponse:
    """Рассчитать валовую и чистую доходность сдачи квартиры в аренду.

    gross_yield = annual_rent / price × 100%
    net_yield   = (annual_rent − expenses) / price × 100%
    payback     = price / annual_net_rent (лет до окупаемости)
    P/R ratio   = price / monthly_rent (Shiller: норма ≤ 200 для инвестиционного рынка)
    """
    result = MortgageCalculator.compute_rental_yield(
        price=req.price,
        monthly_rent=req.monthly_rent,
        expense_ratio=req.expense_ratio,
    )
    return _rental_to_response(result)


@app.get("/rental/market", response_model=list[NeighborhoodRentInfo])
def rental_market() -> list[NeighborhoodRentInfo]:
    """Типичные ставки аренды и доходность по районам Москвы (ЦИАН 2026).

    Пример для 65 кв.м — типичная двушка в Москве.
    """
    EXAMPLE_PRICE_PER_SQM = 350_000  # медиана Москвы 2026, руб/кв.м
    EXAMPLE_SQM = 65.0

    infos: list[NeighborhoodRentInfo] = []
    for neighborhood, rate in sorted(NEIGHBORHOOD_RENT_RATES.items()):
        example_rent = rate * EXAMPLE_SQM
        example_price = EXAMPLE_PRICE_PER_SQM * EXAMPLE_SQM
        yield_result = MortgageCalculator.compute_rental_yield(example_price, example_rent)
        infos.append(
            NeighborhoodRentInfo(
                neighborhood=neighborhood,
                rent_rate_per_sqm=rate,
                example_sqft_65=example_rent,
                example_gross_yield_pct=round(yield_result.gross_yield_pct, 2),
                example_net_yield_pct=round(yield_result.net_yield_pct, 2),
            )
        )
    return infos


@app.post("/investment/analyze", response_model=InvestmentResponse)
def investment_analyze(req: InvestmentRequest) -> InvestmentResponse:
    """Сводный инвестиционный анализ: cashflow = аренда − ипотека − расходы.

    Вердикт:
      strong_buy — положительный cashflow + окупаемость ≤ 20 лет
      buy        — окупаемость ≤ 25 лет
      hold       — окупаемость 25–35 лет
      avoid      — окупаемость > 35 лет
    """
    config = MortgageConfig(
        annual_rate=req.annual_rate,
        term_years=req.term_years,
        down_payment_ratio=req.down_payment_ratio,
    )
    analysis = MortgageCalculator.analyze_investment(
        price=req.price,
        mortgage_config=config,
        monthly_rent=req.monthly_rent,
        expense_ratio=req.expense_ratio,
    )
    return InvestmentResponse(
        price=analysis.price,
        mortgage=_mortgage_to_response(analysis.mortgage),
        rental=_rental_to_response(analysis.rental),
        monthly_cashflow=analysis.monthly_cashflow,
        is_cashflow_positive=analysis.is_cashflow_positive,
        down_payment_recovery_months=analysis.down_payment_recovery_months,
        investment_verdict=analysis.investment_verdict,
    )


@app.post("/estimate/intervals", response_model=PriceIntervalEstimate)
def estimate_with_intervals(prop: PropertyInput) -> PriceIntervalEstimate:
    """Оценить стоимость квартиры с quantile regression интервалами предсказаний.

    Возвращает 90% и 95% интервалы, откалиброванные через CQR (Romano et al. 2019).
    Гарантированное покрытие: ≥90% реальных цен попадают в 90%-интервал.
    """
    qmodel = _load_quantile_artifacts()

    X = _build_feature_array(prop)
    intervals = qmodel.predict_interval(X)
    iv = intervals[0]

    calib_90: float | None = None
    calib_95: float | None = None
    if _quantile_calibration is not None:
        calib_90 = _quantile_calibration.get("coverage_90")
        calib_95 = _quantile_calibration.get("coverage_95")

    return PriceIntervalEstimate(
        estimated_price=max(int(round(iv.point_estimate, -4)), 1_000_000),
        interval_90_low=max(int(round(iv.lower_90, -4)), 500_000),
        interval_90_high=int(round(iv.upper_90, -4)),
        interval_95_low=max(int(round(iv.lower_95, -4)), 500_000),
        interval_95_high=int(round(iv.upper_95, -4)),
        width_90=int(round(iv.width_90, -4)),
        width_95=int(round(iv.width_95, -4)),
        is_cqr_calibrated=iv.is_cqr_calibrated,
        calibration_coverage_90=calib_90,
        calibration_coverage_95=calib_95,
        lgbm_available=lgbm_available(),
    )
