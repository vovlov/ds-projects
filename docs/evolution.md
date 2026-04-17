# Evolution

This portfolio traces a deliberate path from beginner notebooks to production ML systems.

## Timeline

### 2020 — Yandex Practicum
[Yandex Practicum notebooks](https://github.com/vovlov/YandexPraktikum) — EDA, sklearn baselines, pandas. MLOps Level 0.

### 2024 — First production patterns
Added FastAPI, Docker, pytest. Moved from notebooks to proper Python packages.

### Early 2026 — Portfolio upgrade
| Date | What changed |
|------|-------------|
| 2026-03-29 | Pre-commit hooks (ruff, mypy, pytest on push) |
| 2026-03-30 | MLflow Model Registry for Projects 01, 07 |
| 2026-03-31 | SHAP waterfall in API response (Project 07) |
| 2026-04-01 | RAGAS evaluation for RAG (Project 02) |
| 2026-04-02 | Faithfulness gate + confidence score (Project 02) |
| 2026-04-03 | Collection5 NER dataset + batch processing (Project 03) |
| 2026-04-04 | Elliptic Bitcoin dataset + VAE baseline (Project 04) |
| 2026-04-05 | MovieLens-25M + power-law mock (Project 09) |
| 2026-04-06 | RVL-CDIP dataset + GradCAM explainability (Project 06) |
| 2026-04-07 | Two-tower model + LLM re-ranker (Project 09) |
| 2026-04-08 | Multi-model cross-check review (Project 08) |
| 2026-04-09 | Prometheus metrics exporter (Project 05) |
| 2026-04-10 | Automated retraining trigger via PSI (Project 01) |
| 2026-04-13 | Data drift alerting cross-project (Projects 01, 10) |
| 2026-04-14 | Write-Audit-Publish gate for feature store (Project 09) |
| 2026-04-15 | MMD drift detection + retraining trigger (Project 05) |
| 2026-04-16 | Streamlit Cloud deployment (Projects 01, 05, 07) |
| 2026-04-17 | mkdocs documentation site (this site) |

## What Each Iteration Taught

Each daily improvement was driven by a specific gap in the portfolio:

- **Real datasets** (Elliptic, RVL-CDIP, Collection5, MovieLens) → credibility; synthetic data doesn't tell the full story
- **Drift detection before alerts** → the system must know *when* to retrain, not just *how*
- **Cross-project integration** → real ML systems have multiple services talking to each other
- **Explainability in API** → explanation is a user-facing feature, not a debugging tool
