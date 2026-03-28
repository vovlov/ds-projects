# ds-projects — Claude Code Instructions

## Project

ML portfolio: 10 production-grade projects covering tabular ML, NLP/LLM, CV, GNN, time series, recommender systems, and data engineering.

## Known Issues

### CI (GitHub Actions)
- **11/11 test jobs pass** (lint, test-churn, test-rag, test-ner)
- **All pass after src→unique package rename + gitignore fix
- **Root cause:** uv monorepo with shared `src/` namespace across 10 projects. Python caches `src` module from first project, blocking subsequent projects' `src.models` imports.
- **Locally:** All 185 tests pass — `make test` or `./scripts/run_tests.sh <project>`
- **Fix needed:** Rename each project's `src/` to unique package name (e.g., `churn/`, `rag/`, `ner/`). This is a large refactor — do it when explicitly asked.

### Platform
- **macOS x86_64** — PyTorch doesn't install natively
- **Workaround:** Docker training via `scripts/train_all.Dockerfile`
- **Colima** required: `colima start --cpu 4 --memory 8`

## Development

```bash
# Setup
uv sync --extra dev --extra churn --extra rag --extra fraud --extra pricing --extra recsys --extra quality --extra review --extra cv

# Test single project
./scripts/run_tests.sh 01-customer-churn-mlops

# Test all (locally)
make test

# Lint
make lint && make lint-fix
```

## Project Structure

Each of 10 projects follows: `NN-name/{src/,tests/,notebooks/,configs/,Dockerfile,docker-compose.yml,README.md}`

Dependencies managed via extras in root `pyproject.toml`.
