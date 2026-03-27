# Architecture Decisions

## Design Principles

1. **Reproducibility first** — Any project runs with `make setup && make run-{name}`
2. **Production patterns** — Every model has an API, tests, and Docker image
3. **Monorepo** — Shared tooling (CI, linting, Makefile) with independent project dependencies
4. **Progressive complexity** — Projects 01→05 increase in architectural complexity

## Technology Choices

### Python 3.12 + uv
- `uv` for fast, deterministic dependency resolution
- Each project has its own extras in root `pyproject.toml`
- No virtualenv juggling — `uv run` handles everything

### MLflow for Experiment Tracking
- Lightweight, self-hosted, no cloud dependency
- Tracks parameters, metrics, artifacts, model versions
- UI for comparing experiments

### DVC for Data Versioning
- Git-like commands for large files
- Reproducible data pipelines (`dvc.yaml`)
- Remote storage agnostic (S3, GCS, local)

### FastAPI for Model Serving
- Async by default, automatic OpenAPI docs
- Type validation with Pydantic
- Easy to containerize and test

### Docker Compose for Full Stack
- One command to start everything (app + DB + monitoring)
- Matches production deployment patterns
- Easy for reviewers to run locally

## Project Independence

Each project can be developed, tested, and deployed independently:
- Own `pyproject.toml` dependencies (via extras)
- Own `Dockerfile` and `docker-compose.yml`
- Own CI job in GitHub Actions
- Shared only: linting rules, Makefile targets, CI workflow structure
