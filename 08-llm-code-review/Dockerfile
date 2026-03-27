FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    anthropic scikit-learn fastapi uvicorn gradio

COPY src/ src/
COPY configs/ configs/

EXPOSE 7868
CMD ["python", "-m", "src.dashboard.app"]
