# 03 — NER Service: Named Entity Recognition

**Сервис извлечения именованных сущностей из русскоязычных текстов** — определяет имена людей, организации и географические объекты. Rule-based baseline + fine-tuned BERT (обучен в Docker).

*Named Entity Recognition service for Russian text — extracts persons, organizations, and locations. Rule-based baseline + fine-tuned BERT transformer (trained in Docker).*

> **Эволюция:** В Практикуме я [классифицировал тексты по тональности](https://github.com/vovlov/YandexPraktikum/tree/master/project_13_ML_for_text) (positive/negative). Здесь — token-level задача: каждому слову присваивается метка сущности (BIO-формат).

## Бизнес-задача

Интернет-магазин хочет автоматически извлекать из текстов обзоров и комментариев: упомянутые бренды (ORG), города доставки (LOC), имена менеджеров (PER). Это нужно для аналитики, маршрутизации обращений и CRM-интеграции.

## Пример работы

```
Вход:  "Владимир Путин посетил Москву и встретился с Газпромом."
Выход: [
  {"text": "Владимир Путин", "label": "PER", "start": 0, "end": 14},
  {"text": "Москву",         "label": "LOC", "start": 24, "end": 30},
  {"text": "Газпромом",      "label": "ORG", "start": 47, "end": 56}
]
```

## Модели

| Модель | Подход | Качество | Скорость |
|--------|--------|----------|----------|
| Rule-based | Regex-паттерны для известных сущностей | Высокая precision, низкий recall | <1ms |
| **BERT NER** | Fine-tuned bert-base-multilingual-cased | Хороший recall на новых именах | ~50ms/текст |

BERT-модель обучена в Docker-контейнере (20 эпох на 15 аннотированных примерах). Тест: `"Владимир Путин посетил Москву"` → `[B-PER, I-PER, O, B-LOC, O]` — корректно.

## Архитектура

```
  Входной текст (русский)
        │
        ├──────────────────┐
        ▼                  ▼
  ┌─────────────┐   ┌─────────────┐
  │  Rule-based │   │  BERT NER   │
  │  (Regex)    │   │  (Optional) │
  └──────┬──────┘   └──────┬──────┘
         │                 │
         ▼                 ▼
  ┌──────────────────────────┐
  │  Entity List             │
  │  [{text, label, pos}]    │
  └──────────┬───────────────┘
             │
     ┌───────┤───────┐
     ▼       ▼       ▼
  FastAPI  Streamlit  Batch API
  :8000    :8501      /predict/batch
```

## Быстрый старт

```bash
make setup-ner
cd 03-ner-service

# Демо с подсветкой сущностей
uv run streamlit run src/demo/app.py

# API
uv run uvicorn src.api.app:app --reload
```

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| Baseline | Regex + словари | Быстрый старт без ML-инфраструктуры |
| Transformer | bert-base-multilingual-cased | Multilingual, хорошо работает с русским |
| BIO-теги | O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC | Стандарт для sequence labeling |
| API | FastAPI (single + batch) | Пакетная обработка для нагрузки |
| Demo | Streamlit | Подсветка сущностей цветами: PER=красный, ORG=бирюзовый, LOC=синий |
| Тесты | pytest (14 тестов) | BIO-extraction, rule-based detection, edge cases |
