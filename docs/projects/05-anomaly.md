# 05 · Realtime Anomaly Detection

> **Business domain:** SRE team — infrastructure metrics monitoring  
> **Package:** `anomaly/`  
> **Directory:** `05-realtime-anomaly/`

## What it solves

Detects anomalies in streaming infrastructure metrics (CPU, memory, latency, error rate) in real-time. Alerts on drift between current and reference distributions before issues escalate to incidents.

## Architecture

```mermaid
graph LR
    A[Kafka stream] --> B[LSTM detector]
    B --> C{Anomaly?}
    C -->|yes| D[Prometheus alert]
    C --> E[/detect API]
    F[Reference window] --> G[MMD drift check]
    G -->|drift detected| H[Retraining trigger]
    H --> B
    E --> I[Grafana dashboard]
```

## Key components

### LSTM Anomaly Detector (`anomaly/models/`)
- Sliding window LSTM trained on normal metric patterns
- Reconstruction error → anomaly score
- Configurable threshold (default: 3σ)

### Prometheus Metrics Exporter {#prometheus}

`anomaly/metrics/prometheus_exporter.py` — production-grade observability:

```
anomaly_requests_total      Counter   Total detection requests
anomaly_anomalies_total     Counter   Detected anomalies
anomaly_score               Histogram Score distribution (SRE buckets)
anomaly_detection_seconds   Histogram Inference latency
anomaly_threshold           Gauge     Current detection threshold
anomaly_window_size         Gauge     Sliding window size
```

`track_detection()` context manager for automatic latency tracking.

### MMD Drift Detection {#mmd-drift-detection}

`anomaly/drift/mmd.py` — Maximum Mean Discrepancy with RBF kernel (Gretton 2012):

- `compute_mmd_rbf()` — numpy-only, no alibi-detect dependency
- `_median_heuristic_gamma()` — automatic bandwidth selection
- `bootstrap_mmd_threshold()` — permutation bootstrap, controls Type I error (α)
- `MMDDriftDetector` — null distribution → p-value
- `DriftResult` — audit UUID + ISO 8601 timestamp (EU AI Act Article 9)

### Retraining Trigger (`anomaly/retraining/trigger.py`)
- MMD drift → retraining decision
- MLflow audit log (graceful degradation without MLflow)
- `triggered_by` audit trail

### Streamlit Dashboard

Three tabs: **Live Monitor** (real-time metrics) · **Drift MMD** (distribution comparison) · **Architecture** (system overview)

```bash
cd 05-realtime-anomaly
streamlit run streamlit_app.py
```

### API (`anomaly/api/app.py`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | Detect anomalies in metric batch |
| `/drift/check` | POST | MMD drift test reference vs current |
| `/drift/status` | GET | Last drift result (Grafana-compatible) |
| `/metrics` | GET | Prometheus text format exposition |
| `/health` | GET | Service health + last drift info |

## Running Tests

```bash
cd 05-realtime-anomaly
../.venv/bin/python -m pytest tests/ -v --tb=short
```
