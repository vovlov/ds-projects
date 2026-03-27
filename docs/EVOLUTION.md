# Эволюция: от учебных проектов к production / Evolution: From Learning to Production

Эта страница — не просто changelog. Это история моего пути в Data Science и машинном обучении.

## 2020: Фундамент — Яндекс.Практикум

В 2020 году я прошёл курс Data Science в Яндекс.Практикуме. 15 проектов, от предобработки данных до компьютерного зрения. Каждый проект — один Jupyter-ноутбук. Никаких тестов, никакого деплоя, никакого version control для данных.

Но именно там я понял, что ML — это не про модели. Это про **пайплайны**: данные → фичи → обучение → валидация → deploy → мониторинг. Ноутбук — это proof of concept, а не продукт.

> [Архив учебных проектов](https://github.com/vovlov/YandexPraktikum) — 15 проектов, сохранены как есть.

**Стек тогда:** pandas, scikit-learn, TensorFlow 1.x, nltk, matplotlib

## 2025: Production — ds-projects

Каждый учебный топик стал production-системой. Не потому что нужно было «переделать», а потому что за 5 лет я понял, чего не хватало.

### Что изменилось конкретно

| Учебный проект (2020) | Production-версия (2025) | Что я добавил |
|-----------------------|--------------------------|---------------|
| Отток клиентов (sklearn notebook) | [01-customer-churn-mlops](../01-customer-churn-mlops) | Polars вместо pandas, Optuna HPO, MLflow tracking, FastAPI, Streamlit dashboard, Docker, 14 тестов |
| Классификация текстов (TF-IDF) | [02-rag-enterprise](../02-rag-enterprise) | RAG-пайплайн с LangChain, ChromaDB, Claude API, Gradio UI, 11 тестов |
| То же NLP-задание | [03-ner-service](../03-ner-service) | Token-level NER вместо document-level classification, BERT fine-tuning, FastAPI с batch endpoint, Streamlit demo |
| ML в бизнесе (табличный fraud) | [04-graph-fraud-detection](../04-graph-fraud-detection) | Graph Neural Networks (GCN, GraphSAGE), NetworkX для анализа графа, сравнение с табличным baseline |
| Временные ряды (batch forecasting) | [05-realtime-anomaly](../05-realtime-anomaly) | Real-time детекция через Kafka, LSTM Autoencoder, Grafana мониторинг, webhook-алерты |

### Принципиальные отличия

| Аспект | 2020 | 2025 |
|--------|------|------|
| Код | Jupyter notebook | Python-модули с type hints |
| Данные | Ручной CSV | DVC pipeline, автоматическая загрузка |
| Модели | Обучил один раз, записал метрики | MLflow tracking, Optuna HPO, reproducibility |
| Деплой | Нет | FastAPI + Docker + docker-compose |
| Мониторинг | Нет | Grafana + Prometheus |
| Тесты | Нет | pytest (62 теста суммарно) |
| CI/CD | Нет | GitHub Actions (lint + test per project) |
| Конфигурация | Хардкод | YAML-файлы |
| Воспроизводимость | Запустить ноутбук руками | `make setup && make run-{project}` |

## Что дальше

- **Реальные датасеты** вместо синтетических для fraud и anomaly
- **DVC pipelines** для полной воспроизводимости data → train → evaluate
- **A/B testing framework** для сравнения моделей в production
- **Feature store** — общее хранилище признаков между проектами
