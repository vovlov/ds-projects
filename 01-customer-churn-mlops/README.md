# 01 — Customer Churn Prediction with MLOps

> **Evolution from:** [Yandex.Praktikum Project 7 (Churn)](https://github.com/vovlov/YandexPraktikum/tree/master/project_7_Training_teacher) + [Project 17 (Graduation)](https://github.com/vovlov/YandexPraktikum/tree/master/project_17_Graduation%20project)

End-to-end customer churn prediction system for a telecom operator — from data pipeline through model training to production-ready API and dashboard.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Raw CSV     │────▶│  Feature Eng.  │────▶│   Training   │
│  (DVC)       │     │  (Polars)      │     │  (Optuna +   │
│              │     │                │     │   MLflow)    │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                    │
                           ┌────────────────────────┤
                           ▼                        ▼
                    ┌──────────────┐     ┌──────────────────┐
                    │  FastAPI     │     │  Streamlit       │
                    │  /predict    │     │  Dashboard       │
                    │  :8000       │     │  :8501           │
                    └──────────────┘     └──────────────────┘
```

## Results

| Model | F1 Score | ROC AUC | Training |
|-------|----------|---------|----------|
| CatBoost | 0.6232 | 0.8401 | Optuna 30 trials |
| **LightGBM** | **0.6372** | **0.8471** | Optuna 30 trials |

> LightGBM selected as the best model based on F1 score.

## Quick Start

```bash
# From repo root
make setup-churn

# Download data + train
cd 01-customer-churn-mlops
uv run python train.py

# Run dashboard
uv run streamlit run src/dashboard/app.py

# Run API
uv run uvicorn src.api.app:app --reload

# Docker
docker compose up
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
  }'
```

## Stack

| Component | Tool |
|-----------|------|
| Data processing | Polars |
| Feature store | Engineered features (tenure groups, service counts, spend ratios) |
| ML models | CatBoost, LightGBM |
| Hyperparameter tuning | Optuna (30 trials, 5-fold CV) |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| API | FastAPI |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker multi-stage build |
| CI/CD | GitHub Actions |
| Data quality | Great Expectations |
| Explainability | Feature importances, CatBoost built-in |

## Dataset

**Telco Customer Churn** (IBM, Kaggle) — 7,043 customers, 21 features.
Binary classification: will the customer churn (leave) in the next period?

**Features engineered:**
- `AvgMonthlySpend` — total charges / tenure
- `ExpectedTotalCharges` — monthly × tenure
- `TenureGroup` — new (≤12m), mid (≤36m), long (>36m)
- `NumServices` — count of subscribed services (0–6)
