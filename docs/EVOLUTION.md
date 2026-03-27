# Evolution: From Learning to Production

This document traces the journey from foundational Data Science training to production-grade ML engineering.

## 2020: Foundation — Yandex.Praktikum

Completed the Data Science program covering:
- Data preprocessing, EDA, statistical analysis
- Classical ML (classification, regression, clustering)
- Time series forecasting
- NLP basics (TF-IDF, Word2Vec, BERT)
- Computer Vision (CNNs with TensorFlow)
- SQL and data extraction

**Stack:** pandas, scikit-learn, TensorFlow 1.x, nltk, matplotlib
**Format:** Jupyter notebooks, no version control, no tests, no deployment

> [Archive repository](https://github.com/vovlov/YandexPraktikum)

## 2025–2026: Production — ds-projects

Each foundational topic evolved into a production-ready system:

| Foundation (2020) | Evolution (2025) | What Changed |
|-------------------|------------------|--------------|
| Customer churn notebook | Full MLOps pipeline | MLflow, DVC, CI/CD, FastAPI, Docker |
| TF-IDF text classification | RAG enterprise system | LangChain, vector DBs, LLM evaluation |
| Basic NLP | Production NER service | Fine-tuned transformers, ONNX, API |
| Tabular fraud detection | Graph Neural Networks | PyTorch Geometric, graph construction |
| Time series forecasting | Real-time anomaly detection | Kafka streaming, Grafana monitoring |

## Key Differences

| Aspect | 2020 | 2025 |
|--------|------|------|
| Code | Jupyter notebooks | Python modules + tests |
| Data | Manual CSV files | DVC pipelines, automated download |
| Models | Train once, report metrics | MLflow tracking, hyperparameter optimization |
| Deployment | None | FastAPI + Docker + CI/CD |
| Monitoring | None | Grafana dashboards, data quality checks |
| Reproducibility | Manual | `make setup && make run-{project}` |
