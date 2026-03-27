# 04 — Graph Fraud Detection

> **Evolution from:** [Yandex.Praktikum Project 8 (ML in Business)](https://github.com/vovlov/YandexPraktikum/tree/master/project_8_ML_in_business) — from tabular features to graph neural networks

Fraud detection in transaction networks using Graph Neural Networks (GCN, GraphSAGE) compared against a CatBoost tabular baseline.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Transaction │────▶│  Graph         │────▶│  GNN Model   │
│  Data        │     │  Construction  │     │  (GCN/SAGE)  │
│              │     │  (NetworkX)    │     │              │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                    │
       ┌────────────────────────────────────────────┤
       ▼                        ▼                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  CatBoost    │     │  Comparison  │     │  Streamlit   │
│  Baseline    │     │  Metrics     │     │  Dashboard   │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Results

| Model | F1 Score | ROC AUC | Notes |
|-------|----------|---------|-------|
| CatBoost (baseline) | Synthetic data | Synthetic data | Tabular features only |
| GCN | Synthetic data | Synthetic data | Uses graph structure |
| GraphSAGE | Synthetic data | Synthetic data | Inductive learning |

## Quick Start

```bash
make setup-fraud
cd 04-graph-fraud-detection

# Run tests
uv run pytest tests/ -v

# Run demo (when available)
uv run streamlit run src/demo/app.py
```

## Stack

| Component | Tool |
|-----------|------|
| Baseline | CatBoost |
| GNN | PyTorch Geometric (GCNConv, SAGEConv) |
| Graph ops | NetworkX |
| Visualization | PyVis, Plotly |
| Data | Synthetic transaction graph generator |
| API | FastAPI |
| Demo | Streamlit |
