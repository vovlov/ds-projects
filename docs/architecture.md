# Architecture Decisions

Key design choices — the *why* behind each technology pick.

## Design Principles

1. **Reproducibility** — every project starts with `make setup && make run-{name}`. No "works on my machine".
2. **Production patterns** — every model has an API, tests, and a Docker image. Notebooks are for EDA, not delivery.
3. **Monorepo** — shared CI, linting, Makefile. But project dependencies are independent (`extras` in `pyproject.toml`).
4. **Progressive complexity** — projects 01→05 escalate: tabular ML → RAG → NER → graphs → real-time streaming.

---

## Technology Choices

### Python 3.12 + uv (not poetry, not pip)

**Why uv?** 10–50× faster than pip for dependency resolution. Deterministic lock file. `uv run` replaces venv activation.

**Why not poetry?** Poetry is slow on large dependency trees (torch + transformers). uv = Rust speed + pip compatibility.

### Polars (not pandas) in Projects 01, 09, 10

**Why?** Cleaner DataFrame API, lazy evaluation, native multithreading. `.with_columns()` is more explicit than `df["col"] = ...`.

**When pandas?** For sklearn integration — `.to_pandas()` at model fit time only.

### CatBoost + LightGBM (not XGBoost) in Projects 01, 07

- **CatBoost**: Native categorical handling. Auto class weights. Built-in SHAP.
- **LightGBM**: 10× faster training. Leaf-wise trees work better on small data.
- **XGBoost**: The 2017 standard — still valid, but not differentiated in 2026.

### ChromaDB (not Pinecone, not Qdrant) in Project 02

**Why?** Embeddable, local, persistence out of the box. No cloud service for a portfolio demo. Production-ready up to ~100K docs.

**When Qdrant?** Distributed search, complex metadata filtering, >1M documents.

### FastAPI for all APIs

Async by default, auto OpenAPI docs, Pydantic input validation. Type hints are mandatory, not optional.

### Docker Compose for full stack

One `docker compose up` starts everything: API + UI + MLflow / Grafana / Prometheus. A recruiter can test any project in 30 seconds.

**Why not Kubernetes?** This is a portfolio, not a production deployment. K8s is overhead for demos.

### pytest + ruff + mypy

- **ruff**: linter + formatter in one Rust binary. Replaces flake8 + black + isort.
- **pytest**: industry standard. Fixtures for data reuse. Parametrisation for edge cases.
- **mypy**: static typing catches bugs before runtime. Critical in ML pipelines where DataFrame schemas change implicitly.

---

## Cross-Project Integrations

```
Project 10 (Quality) ──drift alert──▶ Project 01 (Churn retraining)
Project 01 (Churn)   ──PSI metrics──▶ Project 10 (Quality monitoring)
Project 05 (Anomaly) ──Prometheus──▶  Grafana (shared dashboard)
Project 09 (RecSys)  ──WAP gate──▶    Feature Store (drift-gated publish)
```

---

## Why Each Project Exists

| Project | Gap it fills |
|---------|-------------|
| 01 Churn | Canonical MLOps lifecycle: train → register → serve → retrain |
| 02 RAG | LLM integration with faithfulness evaluation — beyond simple chatbots |
| 03 NER | Structured NLP output: entities, not just text generation |
| 04 Fraud | Graph topology as a feature — tabular ML misses network effects |
| 05 Anomaly | Real-time, stateful inference — different from batch predictions |
| 06 Scanner | Computer vision in a non-image-recognition context |
| 07 Pricing | Explainability as a product feature, not an afterthought |
| 08 Review | LLM tool use + multi-pass reasoning + static analysis injection |
| 09 RecSys | Two-tower architecture + data quality gate (WAP) |
| 10 Quality | Monitoring as infrastructure, not an add-on |
