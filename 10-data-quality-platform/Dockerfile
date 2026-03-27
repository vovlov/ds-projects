FROM python:3.12-slim

WORKDIR /app

# Системные зависимости / System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Зависимости Python / Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Код проекта / Project code
COPY . .

# Генерируем демо-данные при сборке / Generate demo data at build time
RUN python scripts/seed_demo_data.py

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
