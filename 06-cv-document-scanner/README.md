# 06 -- CV Document Scanner

Insurance Document Classification / Классификация страховых документов

## Problem / Задача

**EN:** Insurance companies receive thousands of scanned claim documents daily -- receipts, IDs, medical reports, invoices, contracts.  Manual sorting is slow and error-prone.  This project builds an automatic document type classifier that routes each scan to the right processing queue.

**RU:** Страховые компании ежедневно получают тысячи отсканированных документов: чеки, удостоверения личности, медицинские заключения, счета, договоры.  Ручная сортировка медленная и ненадежная.  Проект автоматически определяет тип документа, чтобы направить скан в правильную очередь обработки.

## Two-track approach / Двухуровневый подход

| Track | Model | When to use |
|-------|-------|-------------|
| **Baseline** | sklearn Random Forest on 5 extracted features | Local dev, CI, demo without GPU |
| **Production** | EfficientNet-V2-S fine-tuned on document images | Docker with GPU, real image data |

The baseline works on any machine (no PyTorch needed).  The CNN training pipeline runs inside Docker and exports an ONNX model that can be served without PyTorch at inference time.

**RU:** Базовый вариант (Random Forest) работает на любой машине без PyTorch.  CNN обучается в Docker и экспортируется в ONNX для inference без PyTorch.

## Evolves from / Развитие проекта

Praktikum project 15 (age detection with CNN).  Here we go further: custom fine-tuning, ONNX export, FastAPI serving, Streamlit dashboard.

## Project structure / Структура проекта

```
06-cv-document-scanner/
  configs/training.yaml       # hyperparameters
  notebooks/eda.py            # exploratory analysis (VS Code / Jupyter cells)
  scanner/
    data/dataset.py           # synthetic document features
    models/classifier.py      # sklearn baseline
    models/cnn.py             # EfficientNet CNN (torch optional)
    api/app.py                # FastAPI service
    dashboard/app.py          # Streamlit dashboard
  tests/
    test_cv_document_scanner.py
  Dockerfile                  # multi-stage: train + serve
  docker-compose.yml
```

## Результаты / Results

| Модель | Accuracy | Подход |
|--------|----------|--------|
| **RandomForest (baseline)** | **0.920** | sklearn, 5 синтетических фичей |
| EfficientNet-V2 (CNN) | — | Docker training, реальные изображения |

Baseline достигает 92% accuracy на 5 типах документов (receipt, id_card, medical_report, invoice, contract). CNN-модель — следующий этап для реальных изображений.

## Quick start / Быстрый старт

```bash
# install dependencies (no torch needed)
pip install scikit-learn polars numpy fastapi uvicorn streamlit plotly pydantic httpx

# run tests
pytest tests/ -v

# start the API
uvicorn scanner.api.app:app --reload

# start the dashboard
streamlit run scanner/dashboard/app.py
```

## Docker (CNN training) / Docker (обучение CNN)

```bash
# build and run the API (sklearn baseline)
docker compose up api

# train CNN on GPU (mount your labelled images to ./data/)
docker compose --profile train run --rm trainer
```

## API usage / Использование API

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"aspect_ratio": 0.77, "brightness": 0.82, "text_density": 0.60, "edge_density": 0.20, "file_size_kb": 350}'
```

Response:
```json
{
  "doc_type": "medical_report",
  "confidence": 0.94,
  "probabilities": {"contract": 0.02, "id_card": 0.01, "invoice": 0.02, "medical_report": 0.94, "receipt": 0.01}
}
```

## Key results / Основные результаты

- Baseline accuracy on synthetic data: **~95%** (features are designed to be separable).
- On real document scans the CNN approach would be necessary -- text density and edge density alone can't capture layout and visual patterns.
- The ONNX export path means production inference doesn't need PyTorch installed.
