# Data Quality Platform

**Платформа мониторинга качества данных / Data Quality Monitoring Platform**

---

## Бизнес-задача / Business Problem

Команда данных крупной компании ежедневно получает десятки датасетов из разных
источников: CSV-выгрузки, SQL-базы, потоковые данные. Без автоматического
мониторинга качества проблемы обнаруживаются слишком поздно — когда испорченные
данные уже попали в отчёты или ML-модели.

A data platform team receives dozens of datasets daily from various sources.
Without automated quality monitoring, data issues are discovered too late —
after corrupted data has already reached reports or ML models.

**Эта платформа решает три задачи / This platform solves three problems:**

1. **Профилирование** — автоматический сбор статистик по каждому столбцу
   (пропуски, распределение, выбросы) / Automated per-column profiling
2. **Проверки качества** — декларативные expectations (аналог Great Expectations,
   но легковесный) / Declarative quality checks
3. **Детекция дрифта** — сравнение текущих данных с эталоном через PSI и KS-тест /
   Distribution drift detection via PSI and KS test

## Технологии / Tech Stack

| Компонент | Технология |
|-----------|-----------|
| Обработка данных | **Polars** |
| SQL-движок | **DuckDB** |
| Стат. тесты | **scipy** |
| API | **FastAPI** |
| Дашборд | **Streamlit** |
| Тесты | **pytest** |
| Контейнеризация | **Docker** |

## Результаты / Results

| Возможность | Статус |
|-------------|--------|
| Профилирование колонок (count, nulls, mean, std, min, max, unique) | Работает |
| Expectations (not_null, unique, range, exists, values_in_set) | 5 типов проверок |
| PSI drift detection | Работает (detects mean shift + std change) |
| KS-test drift | Работает (p-value для численных колонок) |
| API /profile | CSV upload → полный профиль |
| API /validate | CSV + YAML suite → отчёт качества |
| API /drift | Два CSV → отчёт о дрифте |

27 тестов покрывают: профилирование, все expectations (pass + fail cases), drift detection, API endpoints.

## Структура проекта / Project Structure

```
10-data-quality-platform/
├── quality/
│   ├── data/
│   │   ├── connectors.py    # CSV + DuckDB коннекторы
│   │   └── profiler.py      # Профилирование данных
│   ├── quality/
│   │   ├── expectations.py  # Проверки качества (expectations)
│   │   └── drift.py         # Детекция дрифта (PSI, KS)
│   ├── api/
│   │   └── app.py           # FastAPI-приложение
│   └── dashboard/
│       └── app.py           # Streamlit-дашборд
├── tests/
│   └── test_data_quality_platform.py
├── configs/
│   └── expectations.yaml    # Конфиг проверок
├── notebooks/
│   └── eda.py               # Демо профилирования
├── scripts/
│   └── seed_demo_data.py    # Генерация демо-данных
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Быстрый старт / Quick Start

### Локально / Local

```bash
# Установить зависимости / Install dependencies
pip install -e ".[dev]"

# Сгенерировать демо-данные / Generate demo data
python scripts/seed_demo_data.py

# Запустить тесты / Run tests
pytest tests/ -v

# Запустить API / Start API
uvicorn quality.api.app:app --reload --port 8000

# Запустить дашборд / Start dashboard
streamlit run quality/dashboard/app.py
```

### Docker

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Проверка здоровья / Health check |
| POST | `/profile` | Профилирование CSV / Profile a CSV |
| POST | `/validate` | Проверки качества / Run quality checks |
| POST | `/drift` | Детекция дрифта / Drift detection |

### Пример / Example

```bash
# Профилирование / Profiling
curl -X POST http://localhost:8000/profile \
  -F "file=@data/reference.csv"

# Проверки качества / Quality validation
curl -X POST http://localhost:8000/validate \
  -F "file=@data/reference.csv" \
  -F "suite=$(cat configs/expectations.yaml)"

# Дрифт / Drift detection
curl -X POST http://localhost:8000/drift \
  -F "reference=@data/reference.csv" \
  -F "current=@data/current.csv"
```

## Развитие проекта / Roadmap

Этот проект эволюционирует из Praktikum Project 14 (SQL + PySpark) и покрывает
задачи data engineering, data quality и observability.

- [x] Профилирование данных
- [x] Expectations (проверки качества)
- [x] Детекция дрифта (PSI + KS)
- [x] FastAPI
- [x] Streamlit dashboard
- [ ] Планировщик (Airflow / Prefect)
- [ ] Алертинг (Slack / email)
- [ ] Хранение истории проверок (DuckDB)
- [ ] Поддержка Parquet, JSON, S3
