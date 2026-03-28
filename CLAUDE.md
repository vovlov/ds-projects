# ds-projects — Claude Code Instructions

## Project

ML portfolio: 10 production-grade projects. Unique package names per project (churn/, rag/, ner/, etc.).

## CI Status: 11/11 GREEN

All test jobs pass in GitHub Actions.

## Package Layout

Each project uses a unique package name instead of generic `src/`:
- `01-customer-churn-mlops/churn/`
- `02-rag-enterprise/rag/`
- `03-ner-service/ner/`
- `04-graph-fraud-detection/fraud/`
- `05-realtime-anomaly/anomaly/`
- `06-cv-document-scanner/scanner/`
- `07-realestate-pricing/pricing/`
- `08-llm-code-review/review/`
- `09-recsys-feature-store/recsys/`
- `10-data-quality-platform/quality/`

## Platform
- **macOS x86_64** — PyTorch doesn't install natively
- **Workaround:** Docker training via `scripts/train_all.Dockerfile`

## Development

```bash
uv sync --all-extras          # install everything
make test                     # all 185 tests
make lint                     # ruff check + format
./scripts/run_tests.sh 04-graph-fraud-detection  # single project
```

## CI

CI runs `cd project && ../.venv/bin/python -m pytest tests/` which adds CWD to sys.path via `-m` flag.
