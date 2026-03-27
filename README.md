# Data Science & ML Projects

Пять production-ready проектов машинного обучения — от классической предиктивной аналитики до real-time обнаружения аномалий. Каждый проект решает конкретную бизнес-задачу и доведён до стадии работающего API с тестами, Docker-контейнерами и CI/CD.

*Five production-ready ML projects — from classical predictive analytics to real-time anomaly detection. Each project solves a concrete business problem and is delivered as a working API with tests, Docker containers, and CI/CD.*

> Эти проекты — эволюция моего пути в Data Science: от [учебных проектов Яндекс.Практикума (2020)](https://github.com/vovlov/YandexPraktikum) к production-grade ML-системам с современным стеком. Подробнее об этом пути — в [docs/EVOLUTION.md](docs/EVOLUTION.md).

## Проекты / Projects

| # | Проект | Бизнес-задача | Стек | Тесты |
|---|--------|---------------|------|-------|
| 01 | [Customer Churn MLOps](01-customer-churn-mlops) | Предсказание оттока клиентов телеком-оператора | CatBoost, LightGBM, MLflow, Optuna, FastAPI, Streamlit | 14 |
| 02 | [RAG Enterprise](02-rag-enterprise) | Q&A по корпоративным документам с помощью LLM | LangChain, ChromaDB, Claude API, Gradio | 11 |
| 03 | [NER Service](03-ner-service) | Извлечение именованных сущностей из русскоязычных текстов | PyTorch, HuggingFace Transformers, FastAPI, Streamlit | 14 |
| 04 | [Graph Fraud Detection](04-graph-fraud-detection) | Обнаружение мошеннических транзакций через граф связей | PyTorch Geometric, CatBoost, NetworkX | 9 |
| 05 | [Realtime Anomaly Detection](05-realtime-anomaly) | Real-time обнаружение аномалий в инфраструктурных метриках | Kafka, Grafana, Prometheus, FastAPI | 14 |

**62 теста** | **5 API endpoints** | **5 Docker Compose stacks** | **GitHub Actions CI/CD**

## Связь между проектами / How Projects Connect

Проекты не изолированы — они отражают реальный ML-стек enterprise-компании:

```
                    ┌──────────────────────┐
                    │  01 Customer Churn   │ ← Классический ML + полный MLOps
                    │  (Predict & Retain)  │
                    └──────────┬───────────┘
                               │ Данные о клиентах
                    ┌──────────▼───────────┐
                    │  04 Graph Fraud      │ ← Графовый анализ транзакций
                    │  (Protect Revenue)   │
                    └──────────┬───────────┘
                               │ Мониторинг систем
                    ┌──────────▼───────────┐
                    │  05 Realtime Anomaly │ ← Потоковая обработка метрик
                    │  (Ops Reliability)   │
                    └──────────────────────┘

┌──────────────────────┐     ┌──────────────────────┐
│  02 RAG Enterprise   │     │  03 NER Service      │
│  (Knowledge Access)  │────▶│  (Text Understanding)│
│  Ответы на вопросы   │     │  Извлечение сущностей│
└──────────────────────┘     └──────────────────────┘
```

- **01 → 04:** Табличные данные клиентов обогащаются графовыми признаками транзакций
- **04 → 05:** Fraud-детекция генерирует метрики, за которыми следит система мониторинга
- **02 → 03:** RAG-система может использовать NER для извлечения сущностей из документов

## Стек / Tech Stack

```
ML/DL           CatBoost · LightGBM · PyTorch · PyTorch Geometric · scikit-learn
NLP             LangChain · HuggingFace Transformers · ChromaDB · Claude API
Data            Polars · NumPy · DVC (data versioning)
MLOps           MLflow · GitHub Actions · Docker · Optuna (HPO) · Makefile
API             FastAPI · Streamlit · Gradio
Streaming       Apache Kafka · Prometheus · Grafana
Infra           Docker Compose · uv · ruff · mypy · pytest
```

## Быстрый старт / Quick Start

```bash
git clone https://github.com/vovlov/ds-projects.git
cd ds-projects

# Установка окружения
make setup

# Запуск любого проекта
make run-churn        # 01: Streamlit dashboard
make run-rag          # 02: Gradio чат
make run-ner          # 03: NER демо
make run-fraud        # 04: API для скоринга
make run-anomaly      # 05: docker-compose (Kafka + Grafana)

# Тесты и линтинг
make test             # Все проекты (62 теста)
make lint             # ruff check + format
```

## Структура проекта / Project Structure

Каждый проект следует единой архитектуре:

```
project/
├── README.md             # Бизнес-контекст, результаты, архитектура
├── Dockerfile            # Production-ready образ
├── docker-compose.yml    # Полный стек с зависимостями
├── configs/              # YAML-конфигурация (без хардкода)
├── src/
│   ├── data/             # Загрузка, валидация, feature engineering
│   ├── models/           # Обучение, evaluation
│   ├── api/              # FastAPI endpoints
│   └── dashboard/        # Streamlit / Gradio UI
├── tests/                # pytest (unit + integration)
└── notebooks/            # EDA и эксперименты (Plotly)
```

**CI/CD pipeline:**
```
git push → GitHub Actions → ruff lint → pytest (per project) → Docker build
```

## Автор / Author

**Владимир Ловцов** — Enterprise Architect & AI Practitioner

8+ лет в IT: системный аналитик → техлид → директор разработки → архитектор.
VTB (real-time системы, <0.3s latency) → T1 (enterprise architecture) → Digital Artel (IT-кооператив).

- [lovtsov.dev](https://lovtsov.dev) — персональный сайт
- [GitHub](https://github.com/vovlov) — все проекты
- [Telegram](https://t.me/it_underside) — канал об архитектуре и AI
