FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    langchain langchain-community langchain-text-splitters langchain-anthropic \
    chromadb anthropic fastapi uvicorn gradio

COPY src/ src/
COPY data/ data/

EXPOSE 7860
CMD ["python", "-m", "src.api.app"]
