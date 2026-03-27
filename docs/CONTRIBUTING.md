# Contributing / Разработка

## Как запустить проект локально

```bash
# 1. Клонировать
git clone https://github.com/vovlov/ds-projects.git
cd ds-projects

# 2. Установить uv (если нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Установить зависимости
make setup              # все проекты
make setup-churn        # только Project 01
make setup-pricing      # только Project 07

# 4. Запустить тесты
make test               # все 185 тестов
make test-churn         # только Project 01

# 5. Запустить линтер
make lint               # проверка
make lint-fix           # автоисправление
```

## Структура нового проекта

Каждый проект следует шаблону:

```
NN-project-name/
├── README.md             # Бизнес-задача, архитектура, результаты
├── Dockerfile
├── docker-compose.yml
├── configs/
│   └── *.yaml            # Конфигурация (не хардкод!)
├── src/
│   ├── data/             # Загрузка, генерация, feature engineering
│   ├── models/           # Обучение, evaluation
│   ├── api/
│   │   └── app.py        # FastAPI с /health + основной endpoint
│   └── dashboard/
│       └── app.py        # Streamlit / Gradio
├── tests/
│   └── test_*.py         # pytest: TestData, TestModel, TestAPI
├── notebooks/
│   └── eda.py            # Plotly-based EDA (percent-format)
└── train.py              # Основной скрипт обучения
```

## Правила

- Python 3.12, type hints обязательны
- `ruff` для линтинга и форматирования (line-length=100)
- Каждый проект имеет свой extra в корневом `pyproject.toml`
- Тесты запускаются из директории проекта: `cd project && uv run pytest tests/`
- Зависимости от torch — через Docker (macOS x86_64 workaround)
- Комментарии объясняют "почему", а не "что"
- README билингвальный (русский + английский)

## Ограничения платформы

- **macOS x86_64:** PyTorch не устанавливается нативно
- **Workaround:** Docker-контейнер `python:3.12-slim` (Linux x86_64)
- **Скрипт обучения:** `scripts/train_all.Dockerfile`
- **Проекты 06, 09 без torch:** sklearn baseline нативно, PyTorch в Docker
