# 08 — LLM Code Review: AI-Powered Review Assistant

**AI-ассистент для код-ревью** на базе Claude API + sklearn-классификатор категорий. Вставляешь diff — получаешь структурированные замечания с категорией, severity и конкретными рекомендациями.

*AI code review assistant powered by Claude API + sklearn category classifier. Paste a diff, get structured review comments with category, severity, and actionable suggestions.*

> **Эволюция:** В Практикуме я работал с [классификацией текстов (TF-IDF + ML)](https://github.com/vovlov/YandexPraktikum). Здесь — тот же подход (TF-IDF + LogisticRegression) для классификации замечаний, плюс Claude API для генерации ревью, как в [проекте 02 (RAG)](../02-rag-enterprise/).

## Бизнес-задача

Инженерные команды тратят ~30% рабочего времени на код-ревью. AI-ассистент автоматически генерирует первичные замечания (баги, безопасность, стиль, перформанс, документация), сокращая цикл ревью и позволяя ревьюерам фокусироваться на архитектурных вопросах.

## Архитектура

```
  Code Diff (unified format)
        │
        ▼
  ┌─────────────────────────┐     ┌─────────────────────────┐
  │  Claude API             │     │  TF-IDF + LogReg        │
  │  "Senior reviewer"      │     │  Category classifier    │
  │  → structured JSON      │     │  sklearn (no torch)     │
  └────────────┬────────────┘     └────────────┬────────────┘
               │                                │
               ▼                                ▼
  ┌─────────────────────────────────────────────────────────┐
  │  [{line, category, comment, severity}, ...]             │
  └────────────┬──────────────────────────┬─────────────────┘
               │                          │
      ┌────────▼─────────┐      ┌────────▼─────────┐
      │  FastAPI :7868   │      │  Gradio Dashboard │
      │  /review         │      │  Paste & review   │
      │  /classify       │      │  Sample diffs     │
      └──────────────────┘      └──────────────────┘
```

## Быстрый старт

```bash
make setup-review

# Установить API-ключ
export ANTHROPIC_API_KEY="sk-ant-..."

# Запуск дашборда
cd 08-llm-code-review
uv run python -m src.dashboard.app

# Открыть http://localhost:7868
```

## API

```bash
# Health check
curl http://localhost:7868/health

# AI review (требует ANTHROPIC_API_KEY)
curl -X POST http://localhost:7868/review \
  -H "Content-Type: application/json" \
  -d '{"diff": "--- a/foo.py\n+++ b/foo.py\n+x = eval(input())"}'

# Classify review comment (sklearn, без API ключа)
curl -X POST http://localhost:7868/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "SQL injection via f-string"}'
# → {"category": "security", "confidence": 0.85, ...}
```

## Как это работает

1. **Review generation:** Diff отправляется в Claude API с системным промптом "senior code reviewer". Модель возвращает JSON-массив замечаний: `{line, category, comment, severity}`
2. **Category classification:** TF-IDF + LogisticRegression (sklearn) классифицирует текст замечания в одну из 5 категорий — работает локально, без API
3. **Sample data:** 12 реалистичных примеров Python-диффов с ревью-комментариями покрывают все категории

## Категории замечаний

| Категория | Примеры |
|-----------|---------|
| **bug** | Off-by-one, None-check, неочищаемый state |
| **security** | SQL injection, path traversal |
| **style** | Naming conventions (PEP 8), неинформативные имена |
| **performance** | O(n^2) вместо O(n), N+1 queries |
| **documentation** | Отсутствие docstring, magic numbers без комментариев |

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| AI Review | Claude API (Anthropic) | Генерация структурированных замечаний |
| Классификатор | TF-IDF + LogisticRegression | Категоризация без GPU и torch |
| API | FastAPI | REST-эндпоинты /review, /classify, /health |
| UI | Gradio | Интерактивный интерфейс для вставки диффов |
| Конфиг | YAML | Модель, параметры TF-IDF, порты |
| Тесты | pytest (14 тестов) | Data, classifier, reviewer, API |

## Развитие: LoRA Fine-Tuning (Docker only)

В будущем — дообучение open-source LLM на реальных ревью-данных:

```bash
# Fine-tuning в Docker (требует GPU, torch, transformers)
docker compose -f docker-compose.finetune.yml up
```

Основной демо работает **без torch** — только sklearn + Claude API. Fine-tuning код изолирован в Docker-контейнере и не нужен для запуска.
