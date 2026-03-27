# 07 — Real Estate Price Estimation

**Автоматическая оценка стоимости московской недвижимости** — модель, API и дашборд для маркетплейса. Ключевое требование: модель должна быть объяснимой — клиенту нужно не просто число, а понимание, почему квартира стоит столько.

*Automated Moscow real estate pricing — model, API, and dashboard for a property marketplace. Key requirement: the model must be explainable — clients need to understand WHY a property is priced the way it is.*

> **Эволюция:** В Практикуме я решал две связанные задачи — EDA по недвижимости (проект 3) и предсказание цен на авто с gradient boosting (проект 11). Здесь объединяю оба подхода: EDA-мышление + продвинутые модели + production-обвязка.

## Бизнес-задача / Business Problem

Маркетплейс недвижимости хочет автоматически оценивать квартиры при размещении объявления. Это нужно для:
1. **Продавцов** — подсказать адекватную цену (завышенные объявления не продаются)
2. **Покупателей** — показать, справедлива ли цена (красный/зелёный индикатор)
3. **Модерации** — выявлять подозрительно дешёвые объявления (возможное мошенничество)

Модель должна объяснять оценку: "район +3М, площадь +2М, состояние -500К".

## Архитектура / Architecture

```
  Synthetic data (1000 properties)
        |
        v
  +---------------+     +---------------+     +---------------+
  |  Polars       |---->|  Optuna       |---->|  MLflow       |
  |  Feature      |     |  20 trials    |     |  Tracking     |
  |  Engineering  |     |  5-fold CV    |     |               |
  +---------------+     +---------------+     +------+--------+
                                                      |
                         +----------------------------+
                         v                            v
                  +---------------+     +--------------------+
                  |  FastAPI      |     |  Streamlit         |
                  |  /estimate    |     |  Dashboard         |
                  |  :8000        |     |  EDA + Estimator   |
                  +---------------+     +--------------------+
```

## Данные / Data

Синтетический датасет: 1000 московских квартир с реалистичным ценообразованием.

| Признак | Описание | Диапазон |
|---------|----------|----------|
| price | Стоимость (RUB) | 3M — 30M |
| sqft | Площадь (m2) | 25 — 200 |
| bedrooms | Комнат | 1 — 6 |
| year_built | Год постройки | 1935 — 2025 |
| neighborhood | Район (15 районов) | Хамовники, Арбат, ... |
| condition | Состояние | отличное — требует ремонта |
| garage | Гараж | 0/1 |

**Инженерные признаки:**
- `age` — возраст дома (проще интерпретировать, чем year_built)
- `price_per_sqft` — цена за метр (для EDA, не для модели — это data leakage)
- `has_garage` — бинарный признак для CatBoost

## Быстрый старт

```bash
# Из корня репо
make setup-pricing

# Обучение
cd 07-realestate-pricing
uv run python train.py

# Дашборд
uv run streamlit run src/dashboard/app.py

# API
uv run uvicorn src.api.app:app --reload

# Docker (API + Dashboard + MLflow)
docker compose up
```

## API

```bash
curl -X POST http://localhost:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 65, "bedrooms": 2, "bathrooms": 1,
    "year_built": 2015, "lot_size": 0, "garage": 1,
    "neighborhood": "Хамовники", "condition": "хорошее"
  }'

# -> {"estimated_price": 14500000, "confidence_low": 13000000,
#     "confidence_high": 16000000, "top_factors": [...]}
```

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| Данные | Polars | Быстрее pandas, строгая типизация |
| Модели | CatBoost, LightGBM | CatBoost — нативные категории, LightGBM — baseline |
| Оптимизация | Optuna (20 trials, 5-fold CV) | Автоматический подбор гиперпараметров |
| Объяснимость | CatBoost native SHAP | Без внешнего SHAP (llvmlite проблемы) |
| Трекинг | MLflow | Воспроизводимость экспериментов |
| API | FastAPI | Async, авто-документация, Pydantic validation |
| UI | Streamlit + Plotly | EDA, метрики, интерактивный оценщик |
| Контейнеры | Docker multi-stage | API и Dashboard в отдельных контейнерах |
| Тесты | pytest | Data quality, feature engineering, API |
| Конфиг | YAML | Гиперпараметры не захардкожены |
