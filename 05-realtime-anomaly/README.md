# 05 — Realtime Anomaly Detection

> **Evolution from:** [Yandex.Praktikum Project 12 (Time Series)](https://github.com/vovlov/YandexPraktikum/tree/master/project_12_Time_series) — from batch forecasting to real-time anomaly detection

Real-time anomaly detection in infrastructure metrics (CPU, latency, request count) using statistical methods and LSTM autoencoders.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Metrics     │────▶│  Anomaly       │────▶│  Alerts      │
│  Stream      │     │  Detector      │     │  (Webhook)   │
│  (Kafka)     │     │  (Z-score /    │     │              │
│              │     │   LSTM AE)     │     │              │
└──────────────┘     └────────────────┘     └──────────────┘
       │                                           │
       ▼                                           ▼
┌──────────────┐                          ┌──────────────┐
│  Grafana     │                          │  FastAPI     │
│  Dashboard   │                          │  /detect     │
│  :3000       │                          │  :8000       │
└──────────────┘                          └──────────────┘
```

## Quick Start

```bash
make setup-anomaly
cd 05-realtime-anomaly

# Run tests
uv run pytest tests/ -v

# Run API
uv run uvicorn src.api.app:app --reload

# Full stack with Kafka + Grafana
docker compose up
```

## API

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '[
    {"timestamp": 0, "cpu": 45.2, "latency": 52.1, "requests": 1050},
    {"timestamp": 1, "cpu": 95.8, "latency": 450.3, "requests": 120}
  ]'
```

## Detection Methods

| Method | Type | Description |
|--------|------|-------------|
| Z-Score | Statistical | Rolling window z-score across metrics |
| Multi-Metric | Ensemble | Max anomaly score across CPU, latency, requests |
| **LSTM Autoencoder** | Deep Learning | AUC=0.999, trained in Docker on synthetic data |

## Stack

| Component | Tool |
|-----------|------|
| Detection | Statistical (Z-score), LSTM Autoencoder (PyTorch) |
| Streaming | Kafka (producer/consumer) |
| Monitoring | Grafana + Prometheus |
| Cache | Redis |
| API | FastAPI |
| Data | Synthetic metric generator with anomaly injection |
| Containerization | Docker Compose (full stack) |

## Data Generation

Synthetic time series generator producing:
- **CPU load**: 0-100%, daily seasonality + noise
- **Latency**: Right-skewed with seasonality
- **Request count**: Poisson-distributed with daily patterns

Anomaly types injected:
- **Spike**: Sudden metric increase
- **Level shift**: Sustained change in baseline
- **Drop**: Sudden decrease in traffic
