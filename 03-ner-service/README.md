# 03 — NER Service: Named Entity Recognition for Russian Text

> **Evolution from:** [Yandex.Praktikum Project 13 (ML for Text)](https://github.com/vovlov/YandexPraktikum/tree/master/project_13_ML_for_text) — from text classification to token-level entity extraction

Production NER service extracting persons (PER), organizations (ORG), and locations (LOC) from Russian text.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Input Text  │────▶│  NER Model     │────▶│  Entities    │
│  (Russian)   │     │  (Rule-based / │     │  PER/ORG/LOC │
│              │     │   Transformer) │     │              │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                    │
                           ┌────────────────────────┤
                           ▼                        ▼
                    ┌──────────────┐     ┌──────────────────┐
                    │  FastAPI     │     │  Streamlit       │
                    │  /predict    │     │  Demo            │
                    │  :8000       │     │  :8501           │
                    └──────────────┘     └──────────────────┘
```

## Quick Start

```bash
make setup-ner
cd 03-ner-service

# Run demo
uv run streamlit run src/demo/app.py

# Run API
uv run uvicorn src.api.app:app --reload
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Владимир Путин посетил Москву."}'
```

Response:
```json
{
  "entities": [
    {"text": "Владимир Путин", "label": "PER", "start": 0, "end": 14},
    {"text": "Москву", "label": "LOC", "start": 24, "end": 30}
  ],
  "text": "Владимир Путин посетил Москву."
}
```

## Stack

| Component | Tool |
|-----------|------|
| NER Model (baseline) | Rule-based (regex patterns) |
| NER Model (transformer) | Fine-tuned `bert-base-multilingual-cased` on Russian NER (trained in Docker) |
| API | FastAPI (single + batch endpoints) |
| Demo | Streamlit with entity highlighting |
| Entity Types | PER (persons), ORG (organizations), LOC (locations) |

## Entity Types

| Label | Description | Example |
|-------|------------|---------|
| **PER** | Person names | Владимир Путин, Илон Маск |
| **ORG** | Organizations | Газпром, Яндекс, Сбербанк |
| **LOC** | Locations | Москва, Санкт-Петербург, Россия |
