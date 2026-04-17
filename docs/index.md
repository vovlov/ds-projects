# DS Projects — ML Portfolio

[![CI](https://github.com/vovlov/ds-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/vovlov/ds-projects/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-185+-brightgreen)](https://github.com/vovlov/ds-projects)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Ten **production-grade** ML projects — from classical predictive analytics to LLM fine-tuning and real-time anomaly detection. Each project solves a concrete business problem and ships as a working API with tests, Docker, and CI/CD.

---

## Projects at a Glance

| # | Project | Business Domain | Stack highlights |
|---|---------|----------------|-----------------|
| [01](projects/01-churn.md) | Customer Churn MLOps | Telecom retention | CatBoost · LightGBM · MLflow · SHAP |
| [02](projects/02-rag.md) | RAG Enterprise | HR document Q&A | LangChain · ChromaDB · Claude API |
| [03](projects/03-ner.md) | NER Service | Legal entity extraction | BERT · Collection5 · seqeval |
| [04](projects/04-fraud.md) | Graph Fraud Detection | FinTech P2P fraud | PyG · Elliptic dataset · VAE |
| [05](projects/05-anomaly.md) | Realtime Anomaly | SRE infrastructure | Prometheus · MMD drift · LSTM |
| [06](projects/06-scanner.md) | CV Document Scanner | Insurance docs | EfficientNet · GradCAM · RVL-CDIP |
| [07](projects/07-pricing.md) | Real Estate Pricing | Marketplace valuation | CatBoost · Optuna · SHAP waterfall |
| [08](projects/08-review.md) | LLM Code Review | DevTools PR assistant | Claude API · multi-model · Semgrep |
| [09](projects/09-recsys.md) | RecSys Feature Store | E-commerce personalisation | Two-tower · LLM re-rank · WAP gate |
| [10](projects/10-quality.md) | Data Quality Platform | Data observability | DuckDB · PSI · drift alerting |

---

## MLOps Maturity

This portfolio demonstrates **MLOps Level 2** capabilities:

```
Level 0  No MLOps             ✅ Graduated (Yandex Practicum 2020)
Level 1  DevOps but no MLOps  ✅ CI/CD · Docker · tests
Level 2  Training automation  ✅ MLflow · Optuna · DVC · drift detection
Level 3  Automated deployment ⬜ Next goal
Level 4  Full MLOps           ⬜ Horizon
```

---

## Quick Start

```bash
git clone https://github.com/vovlov/ds-projects.git
cd ds-projects

make setup        # install all deps via uv
make test         # run 185+ tests across 10 projects
make lint         # ruff check + format

# Run individual Streamlit apps
make run-churn    # Project 01
make run-pricing  # Project 07
make run-anomaly  # Project 05
```

---

## Directory Structure

```
ds-projects/
├── 01-customer-churn-mlops/   churn/      CatBoost MLOps pipeline
├── 02-rag-enterprise/         rag/        LangChain + Claude RAG
├── 03-ner-service/            ner/        BERT NER fine-tuning
├── 04-graph-fraud-detection/  fraud/      GNN + VAE fraud detection
├── 05-realtime-anomaly/       anomaly/    Streaming anomaly + MMD
├── 06-cv-document-scanner/    scanner/    EfficientNet + GradCAM
├── 07-realestate-pricing/     pricing/    CatBoost + SHAP API
├── 08-llm-code-review/        review/     Multi-model code review
├── 09-recsys-feature-store/   recsys/     Two-tower + WAP gate
├── 10-data-quality-platform/  quality/    DuckDB + PSI alerting
├── docs/                                  This documentation site
├── scripts/                               Shared CI/train scripts
├── pyproject.toml                         Monorepo uv config
└── Makefile                               Developer workflow
```

---

## Tech Stack

=== "ML / DL"
    CatBoost · LightGBM · PyTorch · PyTorch Geometric · scikit-learn · Optuna · SHAP

=== "NLP / LLM"
    LangChain · HuggingFace Transformers · ChromaDB · Claude API (Anthropic) · LoRA

=== "Computer Vision"
    EfficientNet · ONNX Runtime · Albumentations · GradCAM · RVL-CDIP

=== "RecSys"
    Two-tower neural · FAISS · Feature Store · Redis · MovieLens-25M

=== "Data Engineering"
    Polars · DuckDB · DVC · Great Expectations · PSI drift detection

=== "MLOps"
    MLflow · GitHub Actions · Docker Compose · Prometheus · Grafana · uv

=== "API / UI"
    FastAPI · Streamlit · Gradio · Pydantic · OpenAPI
