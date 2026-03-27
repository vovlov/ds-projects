# Data Science & ML Projects

![CI](https://github.com/vovlov/ds-projects/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-185-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Десять production-ready проектов машинного обучения — от классической предиктивной аналитики до LLM fine-tuning и real-time обнаружения аномалий. Каждый проект решает конкретную бизнес-задачу и доведён до стадии работающего API с тестами, Docker-контейнерами и CI/CD.

*Ten production-ready ML projects — from classical predictive analytics to LLM fine-tuning and real-time anomaly detection. Each project solves a concrete business problem and ships as a working API with tests, Docker, and CI/CD.*

> Эти проекты — эволюция моего пути в Data Science: от [учебных проектов Яндекс.Практикума (2020)](https://github.com/vovlov/YandexPraktikum) к production-grade ML-системам. Подробнее: [docs/EVOLUTION.md](docs/EVOLUTION.md).

## Проекты / Projects

### Core ML & NLP (01–05)

| # | Проект | Бизнес-задача | Стек |
|---|--------|---------------|------|
| 01 | [Customer Churn MLOps](01-customer-churn-mlops) | Предсказание оттока клиентов | CatBoost, LightGBM, MLflow, Optuna, FastAPI |
| 02 | [RAG Enterprise](02-rag-enterprise) | Q&A по документам через LLM | LangChain, ChromaDB, Claude API, Gradio |
| 03 | [NER Service](03-ner-service) | Извлечение сущностей из текстов | BERT, FastAPI, Streamlit |
| 04 | [Graph Fraud Detection](04-graph-fraud-detection) | Обнаружение мошенничества через граф | PyTorch Geometric, CatBoost, NetworkX |
| 05 | [Realtime Anomaly](05-realtime-anomaly) | Real-time детекция аномалий | Kafka, Grafana, Prometheus, LSTM |

### Computer Vision, Pricing, LLM, RecSys, Data Engineering (06–10)

| # | Проект | Бизнес-задача | Стек |
|---|--------|---------------|------|
| 06 | [CV Document Scanner](06-cv-document-scanner) | Классификация документов по фото | EfficientNet, ONNX, Albumentations |
| 07 | [Real Estate Pricing](07-realestate-pricing) | Оценка стоимости недвижимости | CatBoost, Optuna, SHAP, Polars |
| 08 | [LLM Code Review](08-llm-code-review) | AI-ассистент для код-ревью | Claude API, LoRA fine-tuning, Gradio |
| 09 | [RecSys Feature Store](09-recsys-feature-store) | Персонализированные рекомендации | SVD, FAISS, Feature Store, Redis |
| 10 | [Data Quality Platform](10-data-quality-platform) | Мониторинг качества данных | DuckDB, PSI/KS drift detection, Polars |

**185 тестов** | **10 API endpoints** | **10 Docker Compose stacks** | **GitHub Actions CI/CD**

## Покрытие направлений / Coverage

| Направление | Проекты |
|-------------|---------|
| Табличный ML / MLOps | 01, 07 |
| NLP / LLM / RAG | 02, 03, 08 |
| Computer Vision | 06 |
| Графы / GNN | 04 |
| Временные ряды / Streaming | 05 |
| Рекомендательные системы | 09 |
| Data Engineering / Quality | 10 |
| Feature Engineering | 01, 07, 09, 10 |
| Explainability (SHAP) | 07 |
| Drift Detection | 10 |

## Стек / Tech Stack

```
ML/DL           CatBoost · LightGBM · PyTorch · PyTorch Geometric · scikit-learn · Optuna
NLP/LLM         LangChain · HuggingFace · ChromaDB · Claude API · LoRA/QLoRA
CV              EfficientNet · ONNX Runtime · Albumentations
RecSys          SVD · FAISS · Feature Store · Redis
Data            Polars · DuckDB · DVC · Great Expectations
MLOps           MLflow · GitHub Actions · Docker · Makefile
API             FastAPI · Streamlit · Gradio
Streaming       Apache Kafka · Prometheus · Grafana
Quality         PSI · KS-test · Distribution profiling
Infra           Docker Compose · uv · ruff · mypy · pytest
```

## Быстрый старт / Quick Start

```bash
git clone https://github.com/vovlov/ds-projects.git
cd ds-projects

make setup              # Все зависимости
make test               # 184 теста по всем 10 проектам
make lint               # ruff check + format

# Запуск любого проекта
make run-churn          # 01: Streamlit
make run-pricing        # 07: Streamlit
make run-recsys         # 09: Streamlit
make run-quality        # 10: Streamlit
```

## Автор / Author

**Владимир Ловцов** — Enterprise Architect & AI Practitioner

- [lovtsov.dev](https://lovtsov.dev) | [GitHub](https://github.com/vovlov) | [Telegram](https://t.me/it_underside)
