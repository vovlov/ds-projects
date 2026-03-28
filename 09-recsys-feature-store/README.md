# 09. Recommendation Engine with Feature Store

## Рекомендательная система для e-commerce с feature store

Персонализированные рекомендации товаров: коллаборативная фильтрация (SVD) для
постоянных пользователей, контентный подход для новых (cold start).

Personalized product recommendations: collaborative filtering (SVD) for returning
users, content-based fallback for cold-start users.

> **Эволюция:** В Практикуме я [строил классификатор для рекомендации тарифов](https://github.com/vovlov/YandexPraktikum/tree/master/project_6_Intro_ML) — один sklearn-классификатор. Здесь — полноценная рекомендательная система: SVD-факторизация, feature store для консистентных фичей, content-based fallback для cold start.

**Evolves from:** Project 6 (classification for recommendations), Projects 2, 9, 10
(feature engineering).

---

## Результаты / Results

| Метрика | Значение | Комментарий |
|---------|----------|------------|
| Precision@10 | 0.019 | Синтетические данные с рандомными рейтингами |
| Recall@10 | 0.065 | На реальных данных (MovieLens) будет значительно выше |
| NDCG@10 | 0.038 | Пайплайн верифицирован e2e: train → split → evaluate |

Метрики ожидаемо низкие на синтетике — задача демонстрирует **архитектуру** (feature store, SVD, content-based fallback), а не state-of-the-art quality. На MovieLens-25M с нейронным two-tower ожидается NDCG@10 > 0.3.

## Architecture / Архитектура

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Synthetic   │────▶│  Feature Store   │────▶│  Models       │
│  Data (load) │     │  (registry +     │     │  - SVD Collab │
│  500 users   │     │   offline comp)  │     │  - Content CF │
│  200 products│     │  Parquet storage │     │  - Popular    │
│  10K interact│     └──────────────────┘     └───────┬───────┘
└──────────────┘                                      │
                                                      ▼
                               ┌──────────────────────────────────┐
                               │           Serving Layer           │
                               │  ┌──────────┐  ┌──────────────┐  │
                               │  │ FastAPI   │  │  Streamlit   │  │
                               │  │ :8000     │  │  Dashboard   │  │
                               │  │           │  │  :8501       │  │
                               │  └──────────┘  └──────────────┘  │
                               └──────────────────────────────────┘
```

## Business Problem / Бизнес-задача

E-commerce платформа нуждается в персонализированных рекомендациях:
- **Returning users** — collaborative filtering (SVD matrix factorization)
- **New users (cold start)** — content-based filtering + popular items
- **Feature store** — offline-computed features for real-time serving

## Project Structure / Структура проекта

```
09-recsys-feature-store/
├── recsys/
│   ├── data/load.py              # Synthetic data generation (Polars)
│   ├── models/
│   │   ├── collaborative.py      # SVD-based collaborative filtering
│   │   └── content_based.py      # Content-based + popular items
│   ├── feature_store/
│   │   ├── registry.py           # Feature definitions & storage
│   │   └── offline.py            # Batch feature computation
│   ├── api/app.py                # FastAPI recommendation service
│   └── dashboard/app.py          # Streamlit interactive dashboard
├── tests/
│   └── test_recsys_feature_store.py  # 15+ tests
├── notebooks/eda.py              # Exploratory data analysis
├── configs/training.yaml         # Training configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start / Быстрый старт

```bash
# Install dependencies / Установка зависимостей
pip install -r requirements.txt

# Run tests / Запуск тестов
PYTHONPATH=. pytest tests/ -v

# Start API / Запуск API
PYTHONPATH=. uvicorn recsys.api.app:app --reload

# Start dashboard / Запуск дашборда
PYTHONPATH=. streamlit run recsys/dashboard/app.py

# Run EDA / Запуск EDA
PYTHONPATH=. python notebooks/eda.py

# Docker
docker-compose up --build
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health + model status |
| `/recommend/{user_id}?top_k=10` | GET | Personalized recommendations |
| `/popular?top_k=10` | GET | Popular items (cold start) |

## Models / Модели

### Collaborative Filtering (SVD)
- TruncatedSVD from sklearn on user-item rating matrix
- Excludes already-rated items from recommendations
- Evaluated with Precision@K, Recall@K, NDCG@K

### Content-Based Filtering
- One-hot encoded item features (category + price_tier)
- User profile = weighted average of interacted item profiles
- Cosine similarity for ranking

### Popular Items (Baseline)
- Bayesian-smoothed average rating
- Fallback for completely new users

## Tech Stack

- **Data:** Polars, NumPy
- **ML:** scikit-learn (TruncatedSVD, cosine_similarity)
- **API:** FastAPI, Pydantic
- **Dashboard:** Streamlit
- **Testing:** pytest
- **Config:** YAML
- **Deploy:** Docker, docker-compose
