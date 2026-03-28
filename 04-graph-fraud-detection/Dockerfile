FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    torch torch-geometric networkx catboost scikit-learn \
    pyvis plotly fastapi uvicorn streamlit

COPY fraud/ fraud/

EXPOSE 8501
CMD ["streamlit", "run", "fraud/demo/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
