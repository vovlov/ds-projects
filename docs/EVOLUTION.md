# Эволюция: от учебных проектов к production

Эта страница — история моего пути в Data Science. От 15 Jupyter-ноутбуков до 10 production-ready систем.

## 2020: Фундамент — Яндекс.Практикум

В 2020 году я прошёл курс Data Science в Яндекс.Практикуме. 15 проектов, от предобработки данных до компьютерного зрения. Каждый проект — один Jupyter-ноутбук. Никаких тестов, деплоя, version control для данных.

Но именно там я понял: ML — это не про модели. Это про **пайплайны**: данные → фичи → обучение → валидация → deploy → мониторинг.

> [Архив учебных проектов](https://github.com/vovlov/YandexPraktikum) — 15 проектов, сохранены как есть.

## 2025: Production — ds-projects

Все 15 направлений Практикума эволюционировали в 10 production-систем:

| Практикум (2020) | ds-projects (2025) | Что изменилось |
|---|---|---|
| #7, #17: Отток клиентов | [01-customer-churn-mlops](../01-customer-churn-mlops) | Polars, Optuna HPO, MLflow, FastAPI, Streamlit, Docker |
| #13: Классификация текстов | [02-rag-enterprise](../02-rag-enterprise) | RAG с LangChain + ChromaDB + Claude API |
| #13: То же NLP | [03-ner-service](../03-ner-service) | Token-level NER, BERT fine-tuning, batch API |
| #8: ML в бизнесе | [04-graph-fraud-detection](../04-graph-fraud-detection) | GNN (GCN, GraphSAGE), граф транзакций |
| #12: Временные ряды | [05-realtime-anomaly](../05-realtime-anomaly) | Kafka streaming, LSTM AE, Grafana |
| #15: Компьютерное зрение | [06-cv-document-scanner](../06-cv-document-scanner) | EfficientNet, ONNX export, Albumentations |
| #3, #11: EDA + регрессия | [07-realestate-pricing](../07-realestate-pricing) | CatBoost + SHAP, геопризнаки, стекинг |
| #6: Классификация | [08-llm-code-review](../08-llm-code-review) | Claude API, LoRA fine-tuning, код-ревью |
| #2, #6, #9: Feature eng | [09-recsys-feature-store](../09-recsys-feature-store) | SVD, FAISS, Feature Store, Redis |
| #2, #10, #14: SQL/Data | [10-data-quality-platform](../10-data-quality-platform) | DuckDB, PSI/KS drift, quality expectations |

### Принципиальные отличия

| Аспект | 2020 | 2025 |
|--------|------|------|
| Код | Jupyter notebook | Python-модули с type hints |
| Данные | Ручной CSV | DVC pipeline, автоматическая генерация |
| Модели | Один раз обучил | MLflow tracking, Optuna HPO |
| Деплой | Нет | FastAPI + Docker + docker-compose |
| Мониторинг | Нет | Grafana + Prometheus |
| Тесты | Нет | pytest (184 теста) |
| CI/CD | Нет | GitHub Actions (10 параллельных jobs) |
| Конфигурация | Хардкод | YAML-файлы |
| Качество данных | Нет | PSI/KS drift, expectations framework |
| Explainability | Нет | SHAP, feature importance |
| Воспроизводимость | Нет | `make setup && make run-{project}` |

## Что я бы сделал иначе / Lessons Learned

1. **Начинать с тестов.** В Практикуме я писал код, потом проверял глазами. Сейчас — pytest первым делом.
2. **Не хардкодить.** Магические числа в ноутбуках — кошмар при воспроизведении. YAML-конфиги решают.
3. **Думать о деплое с самого начала.** Модель без API — это research, не product.
4. **Версионировать данные.** DVC pipeline вместо "а где тот CSV, который был вчера".
5. **Документировать решения, а не код.** README должен объяснять "почему CatBoost", а не "how to import pandas".
