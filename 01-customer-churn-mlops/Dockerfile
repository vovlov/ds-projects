FROM python:3.12-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    polars catboost lightgbm scikit-learn optuna mlflow \
    fastapi uvicorn streamlit plotly

COPY churn/ churn/
COPY artifacts/ artifacts/
COPY data/ data/

# ── API ──────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "churn.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Dashboard ────────────────────────────────────
FROM base AS dashboard
EXPOSE 8501
CMD ["streamlit", "run", "churn/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
