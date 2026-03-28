# Архитектурные решения / Architecture Decisions

Здесь описаны ключевые решения — почему выбраны именно эти инструменты, а не альтернативы. Цель — показать осознанный подход, а не «взял первое попавшееся».

## Принципы / Design Principles

1. **Воспроизводимость** — любой проект запускается с `make setup && make run-{name}`. Нет «а у меня на ноутбуке работало».
2. **Production patterns** — каждая модель имеет API, тесты и Docker-образ. Ноутбук — это EDA, а не деливери.
3. **Монорепо** — общий CI, линтинг, Makefile. Но зависимости проектов независимы (extras в pyproject.toml).
4. **Прогрессивная сложность** — проекты 01→05 усложняются: табличный ML → RAG → NER → графы → real-time.

## Выбор технологий / Technology Choices

### Python 3.12 + uv (не poetry, не pip)

**Почему uv?** В 10-50x быстрее pip при резолвинге зависимостей. Lock-файл детерминированный. `uv run` заменяет активацию виртуального окружения.

**Почему не poetry?** Poetry медленный на больших dependency trees (torch + transformers). uv — это Rust-скорость + pip-совместимость.

### Polars (не pandas) в Project 01

**Почему?** DataFrame API лаконичнее, lazy evaluation, нативная многопоточность. На 7K строк разница в скорости не критична, но API чище: `.with_columns()` вместо цепочки `df["col"] = ...`.

**Когда pandas?** Для интеграции с sklearn (`.to_pandas()` на этапе model fitting).

### CatBoost + LightGBM (не XGBoost) в Project 01

**CatBoost:** Нативная обработка категориальных признаков без one-hot encoding. Auto class weights для дисбаланса. Из коробки — SHAP-совместимый feature importance.

**LightGBM:** Быстрее CatBoost на обучение (~10x на этом датасете). Leaf-wise рост дерева лучше работает на малых данных.

**Почему не XGBoost?** CatBoost и LightGBM доминируют в Kaggle-соревнованиях для табличных данных. XGBoost — стандарт 2017 года.

### ChromaDB (не Pinecone, не Qdrant) в Project 02

**Почему?** Встраиваемая, работает локально, persistence из коробки. Для портфолио — не нужен облачный сервис. Production-ready для малых коллекций (<100K документов).

**Когда Qdrant?** Если нужен distributed search, фильтрация по метаданным, или >1M документов.

### FastAPI (не Flask, не Django) для всех API

**Почему?** Async по умолчанию, автогенерация OpenAPI docs, Pydantic валидация на входе. Type hints — не опциональны, а обязательны.

**Когда Flask?** Для быстрого прототипа без валидации (но мы уже прошли этот этап).

### Docker Compose для full stack

**Почему?** Одна команда `docker compose up` поднимает всё: API + UI + MLflow / Grafana / Prometheus. Рекрутер или коллега может проверить проект за 30 секунд.

**Почему не Kubernetes?** Это портфолио, не production-деплой. K8s — overkill для демо.

### Pytest + ruff + mypy

**ruff:** Линтер и форматтер в одном, написан на Rust. Заменяет flake8 + black + isort.

**pytest:** Стандарт индустрии. Fixtures для переиспользования данных между тестами. Параметризация для edge cases.

**mypy:** Статическая типизация ловит баги до runtime. Особенно полезно в ML-пайплайнах, где DataFrame schema меняется неявно.

## Независимость проектов

Каждый проект может быть разработан, протестирован и задеплоен отдельно:

- Своя группа зависимостей (`--extra churn`, `--extra rag`, ...)
- Свой `Dockerfile` и `docker-compose.yml`
- Свой CI job в GitHub Actions
- Общее только: правила линтинга, структура Makefile, pyproject.toml

Это осознанный выбор: монорепо даёт единообразие, но не создаёт связанности между проектами.

### Почему уникальные имена пакетов (churn/, rag/), а не src/

Изначально все проекты использовали `src/`. Это стандарт для одного проекта, но в монорепо каждый `src/` конфликтует с остальными — Python кеширует первый найденный `src` и не видит остальные.

Решение: каждый проект имеет уникальное имя пакета (`churn/`, `rag/`, `fraud/`, ...). Это позволяет CI тестировать все 10 проектов в одном воркфлоу без конфликтов.

## Новые проекты (06–10)

### DuckDB (не PostgreSQL/Spark) в Project 10

**Почему?** DuckDB — встраиваемая аналитическая СУБД. Один файл, zero dependencies. Для профилирования и drift-детекции не нужен сервер. В production — заменяется на Spark/Trino.

### SVD + FAISS (не нейросеть) в Project 09

**Почему SVD?** Работает без GPU, интерпретируем, быстрый baseline. На MovieLens-уровне данных SVD не уступает нейросетям. Two-tower модель (PyTorch) — следующий этап, для Docker-обучения.

**Почему FAISS?** Facebook AI Similarity Search — sub-millisecond ANN для embeddings. Альтернатива: Annoy, ScaNN. FAISS — стандарт индустрии.

### Claude API (не fine-tuned модель) в Project 08

**Почему?** Для демо code review API-based подход быстрее и дешевле. Fine-tuning (LoRA/QLoRA) готов как Docker-скрипт для production. Показываем оба подхода: prompting для быстрого старта, fine-tuning для качества.

### sklearn RandomForest (не CNN) в Project 06

**Почему?** Baseline на синтетических фичах документов. Работает нативно на macOS x86_64. CNN (EfficientNet-V2) — для Docker-обучения с реальными изображениями. Показываем прогрессию: baseline → production.

## Полная coverage matrix

| Технология / Подход | Проект(ы) |
|---|---|
| Табличный ML (классификация) | 01, 04 |
| Табличный ML (регрессия) | 07 |
| NLP / Text Classification | 03, 08 |
| RAG / LLM | 02, 08 |
| Computer Vision / CNN | 06 |
| Graph Neural Networks | 04 |
| Рекомендательные системы | 09 |
| Временные ряды | 05 |
| Streaming / Real-time | 05 |
| Data Engineering / SQL | 10 |
| Feature Store | 09 |
| Data Quality / Drift | 10 |
| MLOps / Experiment Tracking | 01, 07 |
| Explainability (SHAP) | 07 |
| Docker / Containerization | все 10 |
| FastAPI | все 10 |
| CI/CD (GitHub Actions) | все 10 |
| Тесты (pytest) | все 10 (185) |
