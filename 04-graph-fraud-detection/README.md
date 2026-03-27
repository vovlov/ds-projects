# 04 — Graph Fraud Detection

**Обнаружение мошеннических транзакций через граф связей** — сравнение табличного CatBoost-бейслайна с Graph Neural Networks (GCN, GraphSAGE). Мошенники часто действуют кластерами — граф это ловит.

*Fraud detection in transaction networks — comparing tabular CatBoost baseline with Graph Neural Networks. Fraudsters tend to cluster — graph structure captures this.*

> **Эволюция:** В Практикуме я [выбирал локации для бурения скважин](https://github.com/vovlov/YandexPraktikum/tree/master/project_8_ML_in_business) на табличных данных. Здесь те же принципы (бизнес-метрики, Bootstrap-анализ), но данные — граф транзакций, а модель — GNN.

## Бизнес-задача

Финтех-компания обрабатывает миллионы транзакций. 5-8% из них — мошеннические. Табличные признаки (сумма, частота) ловят очевидный фрод, но пропускают организованные схемы. Граф транзакций показывает, что мошенники связаны друг с другом — GNN использует эту структуру.

## Результаты

| Модель | F1 Score | ROC AUC | Что использует |
|--------|----------|---------|----------------|
| CatBoost (baseline) | — | — | Только табличные признаки |
| **GCN** | **0.215** | **0.760** | Табличные + структура графа |

GCN обучен на синтетическом графе (1000 узлов, 5000 рёбер, 6.8% fraud). Модель обучена в Docker-контейнере с PyTorch Geometric.

**Почему F1 низкий?** При 6.8% fraud rate и 200 тестовых узлах (из которых ~10 fraud) — recall 70% это хороший результат. На реальных данных (Elliptic, IEEE-CIS) с тысячами fraud-примеров метрики будут выше.

## Архитектура

```
  Транзакции
        │
        ▼
  ┌─────────────┐     ┌─────────────┐
  │  Graph      │────▶│  Node       │
  │  Builder    │     │  Features   │
  │  (NetworkX) │     │  (avg_amt,  │
  │             │     │   n_txn,    │
  └──────┬──────┘     │   age)      │
         │            └──────┬──────┘
         │                   │
         ▼                   ▼
  ┌─────────────┐   ┌──────────────┐
  │  Edge Index │   │  CatBoost    │ ← baseline (табличный)
  │  (src, dst) │   │  Baseline    │
  └──────┬──────┘   └──────────────┘
         │
         ▼
  ┌─────────────┐
  │  GCN / SAGE │ ← использует и фичи, и структуру графа
  │  2 layers   │
  │  + dropout  │
  └──────┬──────┘
         │
         ▼
    Fraud Score
```

## Быстрый старт

```bash
make setup-fraud
cd 04-graph-fraud-detection

# Тесты
uv run pytest tests/ -v

# API (автоматически обучит baseline на первом запросе)
uv run uvicorn src.api.app:app --reload

# Скоринг транзакции
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"avg_amount": 5000, "n_transactions": 15, "account_age_days": 10}'
```

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| Baseline | CatBoost | Сильный бейслайн для табличных данных |
| GNN | PyTorch Geometric (GCNConv, SAGEConv) | Два архитектуры для сравнения |
| Граф | NetworkX | Анализ структуры, метрики графа |
| Данные | Синтетический генератор | Контролируемый fraud rate, кластеризация мошенников |
| API | FastAPI (/score, /score/batch) | Скоринг отдельных транзакций и пакетов |
| Тесты | pytest (9 тестов) | Генерация данных, структура графа, baseline |
