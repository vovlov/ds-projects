# 05 — Realtime Anomaly Detection

**Обнаружение аномалий в инфраструктурных метриках в реальном времени** — Z-score детектор + LSTM Autoencoder, потоковая обработка через Kafka, визуализация в Grafana.

*Real-time anomaly detection in infrastructure metrics — Z-score detector + LSTM Autoencoder, Kafka streaming, Grafana visualization.*

> **Эволюция:** В Практикуме я [прогнозировал заказы такси](https://github.com/vovlov/YandexPraktikum/tree/master/project_12_Time_series) batch-методами (sklearn + statsmodels). Здесь — потоковая обработка: метрики приходят в Kafka, модель работает в реальном времени, аномалии отображаются в Grafana.

## Бизнес-задача

SRE-команда мониторит сотни серверов. CPU, latency, request count — тысячи метрик в секунду. Нужно автоматически обнаруживать аномалии (спайки нагрузки, деградацию latency, падение трафика) и алертить до того, как пользователи заметят проблему.

## Результаты

| Модель | AUC | Подход | Скорость |
|--------|-----|--------|----------|
| Z-Score (statistical) | — | Rolling window σ-threshold | <1ms/точка |
| Multi-Metric Ensemble | — | Max score по CPU/latency/requests | <1ms/точка |
| **LSTM Autoencoder** | **0.999** | Reconstruction error | ~5ms/окно |

LSTM AE обучен в Docker на нормальных данных (5000 точек, 0% аномалий). На тестовых данных с 3% аномалий — AUC=0.999.

## Архитектура

```
  Метрики (CPU, Latency, Requests)
        │
        ▼
  ┌─────────────┐     ┌─────────────┐
  │  Kafka      │────▶│  Consumer   │
  │  Producer   │     │  (Detector) │
  │  (Simulator)│     │             │
  └─────────────┘     └──────┬──────┘
                             │
                    ┌────────┤────────┐
                    ▼        ▼        ▼
             ┌──────────┐ ┌────────┐ ┌──────────┐
             │ Z-Score  │ │ LSTM   │ │ Webhook  │
             │ Detector │ │ AE     │ │ Alerting │
             └──────────┘ └────────┘ └──────────┘
                    │        │
                    ▼        ▼
             ┌──────────────────┐
             │  Prometheus      │
             │  → Grafana       │
             │  :3000           │
             └──────────────────┘
```

## Быстрый старт

```bash
make setup-anomaly
cd 05-realtime-anomaly

# Тесты
uv run pytest tests/ -v

# API
uv run uvicorn src.api.app:app --reload

# Полный стек (Kafka + Grafana + Prometheus)
docker compose up
```

## API

```bash
# Batch-детекция аномалий
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '[
    {"timestamp": 0, "cpu": 45.2, "latency": 52.1, "requests": 1050},
    {"timestamp": 1, "cpu": 95.8, "latency": 450.3, "requests": 120}
  ]'

# → [
#   {"is_anomaly": false, "score": 0.5, "threshold": 3.0},
#   {"is_anomaly": true,  "score": 8.2, "threshold": 3.0}
# ]
```

## Типы аномалий

Генератор создаёт три типа аномалий:
- **Spike:** Резкий рост CPU (+30–50%) и latency (×2–5)
- **Level Shift:** Устойчивое повышение нагрузки, падение трафика (×0.3)
- **Drop:** Обвал request count до ~0, рост latency (×3–8)

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| Детекция | Z-Score + Multi-Metric Ensemble | Быстрый статистический baseline |
| Deep Learning | LSTM Autoencoder (PyTorch) | Ловит нелинейные паттерны |
| Streaming | Kafka (producer + consumer) | Real-time обработка потока метрик |
| Мониторинг | Grafana + Prometheus | Визуализация и дашборды |
| Alerting | Webhook module | JSON-алерты при обнаружении аномалии |
| API | FastAPI (/detect) | Batch-скоринг для интеграции |
| Данные | Синтетический генератор | CPU/latency/requests с сезонностью и шумом |
| Тесты | pytest (14 тестов) | Генерация, windowing, детекция, threshold sensitivity |
