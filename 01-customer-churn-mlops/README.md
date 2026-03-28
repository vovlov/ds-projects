# 01 — Customer Churn Prediction with MLOps

**Полный цикл предсказания оттока клиентов** — от сырых данных до работающего API и дашборда. Это мой подход к тому, как должен выглядеть production ML-проект: воспроизводимый, тестируемый, задеплоенный.

*End-to-end customer churn prediction — from raw data to a working API and dashboard. This is how I think a production ML project should look: reproducible, tested, deployed.*

> **Эволюция:** В 2020 году я решал похожую задачу в [Яндекс.Практикуме](https://github.com/vovlov/YandexPraktikum/tree/master/project_7_Training_teacher) — тогда это был один Jupyter-ноутбук с pandas и sklearn. Здесь та же задача, но с Polars, Optuna, MLflow, FastAPI и Docker.

## Бизнес-задача / Business Problem

Телеком-оператор теряет клиентов. Маркетинг хочет знать **кто уйдёт в следующем месяце**, чтобы предложить промокод или специальные условия. Удержать текущего клиента в 5-7 раз дешевле, чем привлечь нового.

**Задача:** Построить модель, которая по 18 признакам клиента (тариф, срок, услуги, способ оплаты) выдаёт вероятность оттока.

## Архитектура / Architecture

```
  CSV (7043 клиента)
        │
        ▼
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Polars     │────▶│  Optuna     │────▶│  MLflow     │
  │  Feature    │     │  30 trials  │     │  Tracking   │
  │  Engineering│     │  5-fold CV  │     │             │
  └─────────────┘     └─────────────┘     └──────┬──────┘
                                                  │
                         ┌────────────────────────┤
                         ▼                        ▼
                  ┌─────────────┐     ┌──────────────────┐
                  │  FastAPI    │     │  Streamlit       │
                  │  /predict   │     │  Dashboard       │
                  │  :8000      │     │  EDA + Predict   │
                  └─────────────┘     └──────────────────┘
```

## Результаты / Results

| Модель | F1 Score | ROC AUC | Время обучения |
|--------|----------|---------|----------------|
| CatBoost | 0.6232 | 0.8401 | ~6 мин (30 trials) |
| **LightGBM** | **0.6372** | **0.8471** | ~30 сек (30 trials) |

LightGBM выбран как лучшая модель по F1.

**Инженерные признаки** (добавлены мной, не из исходных данных):
- `AvgMonthlySpend` — средний месячный расход (TotalCharges / tenure)
- `ExpectedTotalCharges` — ожидаемая сумма (MonthlyCharges × tenure)
- `TenureGroup` — сегмент по сроку: new (≤12м), mid (≤36м), long (>36м)
- `NumServices` — количество подключённых сервисов (0–6)

## Быстрый старт

```bash
# Из корня репо
make setup-churn

# Обучение (загрузит данные, обучит модели, сохранит артефакты)
cd 01-customer-churn-mlops
uv run python train.py

# Дашборд
uv run streamlit run churn/dashboard/app.py

# API
uv run uvicorn churn.api.app:app --reload

# Docker (API + Dashboard + MLflow)
docker compose up
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 844.2
  }'

# → {"churn_probability": 0.73, "churn_prediction": true, "risk_level": "high"}
```

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| Данные | Polars | Быстрее pandas, лаконичнее API |
| Модели | CatBoost, LightGBM | Лучшие для табличных данных |
| Оптимизация | Optuna (30 trials, 5-fold CV) | Автоматический подбор гиперпараметров |
| Трекинг | MLflow | Воспроизводимость экспериментов |
| API | FastAPI | Async, авто-документация, Pydantic |
| UI | Streamlit + Plotly | Интерактивный дашборд с EDA |
| Контейнеры | Docker multi-stage | API и Dashboard в отдельных контейнерах |
| Тесты | pytest (14 тестов) | Data quality, feature engineering, загрузка |
| Конфиг | YAML | Гиперпараметры не захардкожены |

## Датасет

**Telco Customer Churn** (IBM, [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) — 7 043 клиента, 21 признак. Бинарная классификация: уйдёт клиент или нет.
