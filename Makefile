.PHONY: setup lint test clean

# ── Setup ──────────────────────────────────────────────────────────
setup:
	uv sync --all-extras
	@echo "✅ Environment ready"

setup-churn:
	uv sync --extra dev --extra churn

setup-rag:
	uv sync --extra dev --extra rag

setup-ner:
	uv sync --extra dev --extra ner

setup-fraud:
	uv sync --extra dev --extra fraud

setup-anomaly:
	uv sync --extra dev --extra anomaly

# ── Lint ───────────────────────────────────────────────────────────
lint:
	uv run ruff check .
	uv run ruff format --check .

lint-fix:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy .

# ── Test ───────────────────────────────────────────────────────────
test: test-churn test-rag test-ner test-fraud test-anomaly

test-churn:
	cd 01-customer-churn-mlops && uv run pytest tests/ -v

test-rag:
	cd 02-rag-enterprise && uv run pytest tests/ -v

test-ner:
	cd 03-ner-service && uv run pytest tests/ -v

test-fraud:
	cd 04-graph-fraud-detection && uv run pytest tests/ -v

test-anomaly:
	cd 05-realtime-anomaly && uv run pytest tests/ -v

# ── Run demos ──────────────────────────────────────────────────────
run-churn:
	cd 01-customer-churn-mlops && uv run streamlit run src/dashboard/app.py

run-rag:
	cd 02-rag-enterprise && uv run python -m src.api.app

run-ner:
	cd 03-ner-service && uv run streamlit run src/demo/app.py

run-fraud:
	cd 04-graph-fraud-detection && uv run streamlit run src/demo/app.py

run-anomaly:
	cd 05-realtime-anomaly && docker compose up

# ── Docker ─────────────────────────────────────────────────────────
docker-churn:
	docker compose -f 01-customer-churn-mlops/docker-compose.yml up --build

docker-rag:
	docker compose -f 02-rag-enterprise/docker-compose.yml up --build

docker-ner:
	docker compose -f 03-ner-service/docker-compose.yml up --build

docker-fraud:
	docker compose -f 04-graph-fraud-detection/docker-compose.yml up --build

docker-anomaly:
	docker compose -f 05-realtime-anomaly/docker-compose.yml up --build

# ── Clean ──────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name mlruns -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
