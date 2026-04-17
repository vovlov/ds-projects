# Contributing

## Dev Environment

```bash
git clone https://github.com/vovlov/ds-projects.git
cd ds-projects

# Install all dependencies
uv sync --all-extras

# Install pre-commit hooks
make pre-commit-install
```

## Project Structure Conventions

Each project follows the same layout:

```
NN-project-name/
├── package/           # Python package (unique name, no src/)
│   ├── __init__.py
│   ├── api/app.py     # FastAPI application
│   ├── models/        # ML models
│   ├── data/          # Data loading / preprocessing
│   └── ...
├── tests/             # pytest tests
├── configs/           # YAML configs
├── notebooks/         # EDA only (excluded from lint)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt   # Streamlit Cloud requirements
└── train.py           # Model training entry point
```

## Coding Standards

- **Type hints** on every function signature
- **Docstrings** in Russian + English for public APIs
- **No comments** unless the WHY is non-obvious
- **No magic numbers** — use named constants or config
- `ruff` format: line length 100, Python 3.12 target

## Testing

```bash
# Single project
cd 01-customer-churn-mlops
../.venv/bin/python -m pytest tests/ -v --tb=short

# All projects
make test

# With coverage
cd PROJECT && ../.venv/bin/python -m pytest tests/ --cov=PACKAGE --cov-report=term-missing
```

Tests must:
- Be **independent** (no shared state between tests)
- Work **without API keys** (mock LLM calls)
- Work **without GPU** (graceful `is_available()` fallback)
- Work **without real datasets** (mock generators)

## Adding a New Feature

1. Write failing tests first
2. Implement the feature in the package
3. Run tests: all green
4. Run lint: `uv run ruff check . && uv run ruff format .`
5. Update `docs/ROADMAP.md` — mark done with date
6. Commit with descriptive message

## Pull Request Checklist

- [ ] Tests pass for the modified project
- [ ] No ruff errors
- [ ] No new test skips
- [ ] ROADMAP.md updated
- [ ] Docs updated if public API changed
