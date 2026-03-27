# %% [markdown]
# # Realtime Anomaly Detection — Signal Analysis
#
# Analyze synthetic time series metrics with injected anomalies.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.data.generator import generate_timeseries, to_windows
from src.models.detector import MultiMetricDetector

# %%
data = generate_timeseries(n_points=2000, anomaly_rate=0.03, seed=42)
print(f"Points: {len(data['timestamps'])}")
print(f"Anomaly rate: {data['labels'].mean():.2%}")
print(f"Anomaly points: {data['labels'].sum()}")

# %% [markdown]
# ## 1. Metric Time Series with Anomalies

# %%
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["CPU %", "Latency (ms)", "Requests/min"])

anomaly_mask = data["labels"] == 1
ts = data["timestamps"]

for i, (metric, name) in enumerate([(data["cpu"], "CPU"), (data["latency"], "Latency"), (data["requests"], "Requests")]):
    fig.add_trace(go.Scatter(x=ts, y=metric, mode="lines", name=name, opacity=0.7, line={"width": 1}), row=i+1, col=1)
    fig.add_trace(
        go.Scatter(x=ts[anomaly_mask], y=metric[anomaly_mask], mode="markers", name=f"{name} Anomaly",
                   marker={"color": "red", "size": 5}),
        row=i+1, col=1,
    )

fig.update_layout(height=800, title="Metrics with Injected Anomalies (red dots)")
fig.show()

# %% [markdown]
# ## 2. Statistical Detector Performance

# %%
detector = MultiMetricDetector(window_size=50, threshold_sigma=3.0)
result = detector.detect(data)

print(f"Detected anomalies: {result.predictions.sum()}")
print(f"Actual anomalies: {data['labels'].sum()}")

# Confusion matrix
tp = ((result.predictions == 1) & (data["labels"] == 1)).sum()
fp = ((result.predictions == 1) & (data["labels"] == 0)).sum()
fn = ((result.predictions == 0) & (data["labels"] == 1)).sum()
tn = ((result.predictions == 0) & (data["labels"] == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPrecision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# %% [markdown]
# ## 3. Anomaly Score Distribution

# %%
fig = px.histogram(
    x=result.scores,
    color=data["labels"].astype(str),
    nbins=100,
    title="Anomaly Score Distribution (by actual label)",
    labels={"x": "Anomaly Score", "color": "Is Anomaly"},
    barmode="overlay",
    opacity=0.7,
)
fig.add_vline(x=result.threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
fig.show()

# %% [markdown]
# ## 4. Threshold Sensitivity

# %%
thresholds = np.arange(1.0, 5.0, 0.25)
f1_scores = []
for thresh in thresholds:
    det = MultiMetricDetector(window_size=50, threshold_sigma=thresh)
    res = det.detect(data)
    tp = ((res.predictions == 1) & (data["labels"] == 1)).sum()
    fp = ((res.predictions == 1) & (data["labels"] == 0)).sum()
    fn = ((res.predictions == 0) & (data["labels"] == 1)).sum()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    f1_scores.append(f)

fig = px.line(x=thresholds, y=f1_scores, title="F1 Score vs Threshold (sigma)", labels={"x": "Threshold (σ)", "y": "F1 Score"}, markers=True)
fig.show()

best_idx = np.argmax(f1_scores)
print(f"Best threshold: {thresholds[best_idx]:.2f}σ → F1={f1_scores[best_idx]:.3f}")

# %% [markdown]
# ## 5. Window Analysis

# %%
X_windows, y_windows = to_windows(data, window_size=30, stride=10)
print(f"Windows: {X_windows.shape}")
print(f"Anomalous windows: {y_windows.sum()} ({y_windows.mean():.1%})")

# %% [markdown]
# ## Key Findings
#
# 1. **Anomalies** manifest as spikes in CPU/latency and drops in request count
# 2. **Z-score detector** with σ=3.0 provides reasonable baseline
# 3. **Optimal threshold** depends on precision/recall tradeoff
# 4. **Multi-metric fusion** (max score) improves detection over single metrics
# 5. **Window-based approach** needed for sequence models (LSTM)
