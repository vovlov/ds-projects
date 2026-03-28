"""
Exploratory data analysis for the document scanner project.

Run as a script or open in VS Code / Jupyter with the `# %%` cell markers.
"""

# %%
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import polars as pl
from scanner.data.dataset import (
    DOC_TYPES,
    FEATURE_COLS,
    generate_synthetic_documents,
    get_feature_matrix,
)
from scanner.models.classifier import train_classifier

# %%  generate data
data = generate_synthetic_documents(n=500)
print(data.shape)
data.head(10)

# %%  class distribution
counts = data.group_by("doc_type").len().sort("doc_type")
fig = px.bar(
    x=counts["doc_type"].to_list(),
    y=counts["len"].to_list(),
    color=counts["doc_type"].to_list(),
    labels={"x": "Document type", "y": "Count"},
    title="Document type distribution",
)
fig.show()

# %%  summary statistics per class
for doc_type in DOC_TYPES:
    subset = data.filter(pl.col("doc_type") == doc_type)
    print(f"\n--- {doc_type} ---")
    print(subset.select(FEATURE_COLS).describe())

# %%  feature distributions
pdf = data.to_pandas()
for col in FEATURE_COLS:
    fig = px.histogram(
        pdf,
        x=col,
        color="doc_type",
        barmode="overlay",
        nbins=40,
        opacity=0.6,
        title=f"Distribution of {col} by document type",
    )
    fig.show()

# %%  pairwise scatter (aspect_ratio vs text_density -- two most discriminative features)
fig = px.scatter(
    pdf,
    x="aspect_ratio",
    y="text_density",
    color="doc_type",
    opacity=0.7,
    title="Aspect ratio vs text density",
)
fig.show()

# %%  correlation matrix
corr = data.select(FEATURE_COLS).to_pandas().corr()
fig = ff.create_annotated_heatmap(
    z=corr.values,
    x=FEATURE_COLS,
    y=FEATURE_COLS,
    colorscale="RdBu",
    showscale=True,
)
fig.update_layout(title="Feature correlation matrix")
fig.show()

# %%  train baseline and inspect
X, y, le = get_feature_matrix(data)
result = train_classifier(X, y, label_encoder=le)

print(f"\nAccuracy: {result['accuracy']:.3f}")
print(f"F1 macro: {result['f1_macro']:.3f}")
print(result["classification_report"])

# %%  confusion matrix
cm = result["confusion_matrix"]
labels = list(le.classes_)
fig_cm = ff.create_annotated_heatmap(
    z=cm, x=labels, y=labels, colorscale="Blues", showscale=True,
)
fig_cm.update_layout(
    title="Confusion matrix",
    xaxis_title="Predicted",
    yaxis_title="Actual",
)
fig_cm.update_yaxes(autorange="reversed")
fig_cm.show()

# %%  feature importances from the random forest
importances = result["model"].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig = px.bar(
    x=[FEATURE_COLS[i] for i in sorted_idx],
    y=importances[sorted_idx],
    labels={"x": "Feature", "y": "Importance"},
    title="Random Forest feature importances",
)
fig.show()
