.PHONY: setup lint test clean

# ── Setup ──────────────────────────────────────────────────────────
setup:
	uv sync --all-extras
	@echo "Environment ready"

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
setup-cv:
	uv sync --extra dev --extra fraud  # sklearn baseline, no torch
setup-pricing:
	uv sync --extra dev --extra pricing
setup-review:
	uv sync --extra dev --extra review
setup-recsys:
	uv sync --extra dev --extra recsys
setup-quality:
	uv sync --extra dev --extra quality

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
test: test-churn test-rag test-ner test-fraud test-anomaly test-cv test-pricing test-review test-recsys test-quality

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
test-cv:
	cd 06-cv-document-scanner && uv run pytest tests/ -v
test-pricing:
	cd 07-realestate-pricing && uv run pytest tests/ -v
test-review:
	cd 08-llm-code-review && uv run pytest tests/ -v
test-recsys:
	cd 09-recsys-feature-store && uv run pytest tests/ -v
test-quality:
	cd 10-data-quality-platform && uv run pytest tests/ -v

# ── Run demos ──────────────────────────────────────────────────────
run-churn:
	cd 01-customer-churn-mlops && uv run streamlit run churn/dashboard/app.py
run-rag:
	cd 02-rag-enterprise && uv run python -m rag.api.app
run-ner:
	cd 03-ner-service && uv run streamlit run ner/demo/app.py
run-fraud:
	cd 04-graph-fraud-detection && uv run streamlit run fraud/demo/app.py
run-anomaly:
	cd 05-realtime-anomaly && docker compose up
run-cv:
	cd 06-cv-document-scanner && uv run streamlit run scanner/dashboard/app.py
run-pricing:
	cd 07-realestate-pricing && uv run streamlit run pricing/dashboard/app.py
run-review:
	cd 08-llm-code-review && uv run python -m review.api.app
run-recsys:
	cd 09-recsys-feature-store && uv run streamlit run recsys/dashboard/app.py
run-quality:
	cd 10-data-quality-platform && uv run streamlit run quality/dashboard/app.py

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
docker-cv:
	docker compose -f 06-cv-document-scanner/docker-compose.yml up --build
docker-pricing:
	docker compose -f 07-realestate-pricing/docker-compose.yml up --build
docker-review:
	docker compose -f 08-llm-code-review/docker-compose.yml up --build
docker-recsys:
	docker compose -f 09-recsys-feature-store/docker-compose.yml up --build
docker-quality:
	docker compose -f 10-data-quality-platform/docker-compose.yml up --build

# ── Clean ──────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name mlruns -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
