# Data Science & ML Projects

Production-grade machine learning projects demonstrating end-to-end MLOps, deep learning, NLP, graph neural networks, and real-time inference.

> **Evolution:** These projects build upon foundational work from my [Data Science training (2020)](https://github.com/vovlov/YandexPraktikum), modernized with current industry stack and production practices.

## Projects

| # | Project | Domain | Stack | Demo |
|---|---------|--------|-------|------|
| 01 | [Customer Churn MLOps](01-customer-churn-mlops) | Tabular ML, MLOps | CatBoost, MLflow, DVC, FastAPI | Streamlit |
| 02 | [RAG Enterprise](02-rag-enterprise) | NLP, LLM, RAG | LangChain, ChromaDB, Claude API | Gradio |
| 03 | [NER Service](03-ner-service) | NLP, Token Classification | PyTorch, HuggingFace, ONNX | Streamlit |
| 04 | [Graph Fraud Detection](04-graph-fraud-detection) | GNN, Fraud Detection | PyTorch Geometric, NetworkX | Streamlit |
| 05 | [Realtime Anomaly Detection](05-realtime-anomaly) | Time Series, Streaming | PyTorch, Kafka, Grafana | Grafana |

## Tech Stack

```
ML/DL:        PyTorch, scikit-learn, CatBoost, LightGBM, PyTorch Geometric
NLP:          HuggingFace Transformers, LangChain, sentence-transformers
Data:         Polars, pandas, DVC, Great Expectations
MLOps:        MLflow, GitHub Actions, Docker, Makefile
API:          FastAPI, Streamlit, Gradio
Streaming:    Kafka, Redis
Monitoring:   Grafana, Prometheus
Infra:        Docker Compose, uv, ruff, mypy, pytest
```

## Quick Start

```bash
# Clone
git clone https://github.com/vovlov/ds-projects.git
cd ds-projects

# Setup environment
make setup

# Run any project
make run-churn        # Project 01: Streamlit dashboard
make run-rag          # Project 02: Gradio chat
make run-ner          # Project 03: NER demo
make run-fraud        # Project 04: Graph visualization
make run-anomaly      # Project 05: Grafana dashboard

# Run tests
make test             # All projects
make test-churn       # Single project

# Lint
make lint
```

## Architecture

Each project follows the same structure:

```
project/
├── README.md           # Business context, results, architecture diagram
├── Dockerfile          # Multi-stage production build
├── docker-compose.yml  # Full stack with dependencies
├── pyproject.toml      # Project-specific dependencies
├── src/                # Production code (API, models, data pipelines)
├── tests/              # Unit + integration tests
├── notebooks/          # Experiments and EDA
└── configs/            # Hydra/YAML configuration
```

**MLOps Pipeline:**
```
git push → GitHub Actions → lint + test → build Docker → [deploy]
                ↓
         MLflow tracking ← model training ← DVC data pipeline
```

## Project Status

| Project | Code | Tests | Docker | CI/CD | Demo |
|---------|------|-------|--------|-------|------|
| 01 Churn MLOps | ✅ | ✅ 14 | ✅ | ✅ | Streamlit |
| 02 RAG Enterprise | ✅ | ✅ 11 | ✅ | ✅ | Gradio |
| 03 NER Service | ✅ | ✅ 14 | ✅ | ✅ | Streamlit |
| 04 Graph Fraud | ✅ | ✅ 9 | ✅ | ✅ | Streamlit |
| 05 Realtime Anomaly | ✅ | ✅ 14 | ✅ | ✅ | Grafana |

## Author

**Vladimir Lovtsov** — Enterprise Architect & AI Practitioner
- [lovtsov.dev](https://lovtsov.dev) | [GitHub](https://github.com/vovlov) | [Telegram](https://t.me/it_underside)
