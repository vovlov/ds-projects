"""
Streamlit dashboard for the document scanner project.

Run with:  streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from src.data.dataset import (
    FEATURE_COLS,
    generate_synthetic_documents,
    get_feature_matrix,
)
from src.models.classifier import predict, train_classifier

st.set_page_config(page_title="Document Scanner", layout="wide")
st.title("Insurance Document Classification")


# cache so we don't regenerate / retrain on every widget interaction
@st.cache_data
def load_data():
    return generate_synthetic_documents(n=500)


@st.cache_resource
def load_model():
    data = generate_synthetic_documents(n=500)
    X, y, le = get_feature_matrix(data)
    return train_classifier(X, y, label_encoder=le)


data = load_data()
result = load_model()
model = result["model"]
le = result["label_encoder"]

# ---------- sidebar: classify a new document ----------
st.sidebar.header("Classify a document")
aspect_ratio = st.sidebar.slider("Aspect ratio", 0.1, 3.0, 0.77)
brightness = st.sidebar.slider("Brightness", 0.0, 1.0, 0.80)
text_density = st.sidebar.slider("Text density", 0.0, 1.0, 0.50)
edge_density = st.sidebar.slider("Edge density", 0.0, 1.0, 0.30)
file_size_kb = st.sidebar.slider("File size (KB)", 10.0, 2000.0, 200.0)

if st.sidebar.button("Predict"):
    features = np.array(
        [[aspect_ratio, brightness, text_density, edge_density, file_size_kb]],
        dtype=np.float32,
    )
    preds = predict(model, features, label_encoder=le)
    pred = preds[0]
    st.sidebar.success(f"**{pred['doc_type']}** ({pred['confidence']:.1%})")
    st.sidebar.json(pred["probabilities"])


# ---------- main area ----------
col1, col2 = st.columns(2)

# document type distribution
with col1:
    st.subheader("Document type distribution")
    counts = data.group_by("doc_type").len().sort("doc_type")
    fig = px.bar(
        x=counts["doc_type"].to_list(),
        y=counts["len"].to_list(),
        labels={"x": "Document type", "y": "Count"},
        color=counts["doc_type"].to_list(),
    )
    st.plotly_chart(fig, use_container_width=True)

# confusion matrix
with col2:
    st.subheader("Confusion matrix (test set)")
    cm = result["confusion_matrix"]
    labels = list(le.classes_) if le is not None else [str(i) for i in range(cm.shape[0])]

    # plotly annotated heatmap
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    # flip y axis so row 0 is at top
    fig_cm.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_cm, use_container_width=True)

# feature distributions per class
st.subheader("Feature distributions by document type")
# convert to pandas for plotly (polars support in px is still experimental)
pdf = data.to_pandas()

tabs = st.tabs(FEATURE_COLS)
for tab, col_name in zip(tabs, FEATURE_COLS, strict=True):
    with tab:
        fig = px.histogram(
            pdf,
            x=col_name,
            color="doc_type",
            barmode="overlay",
            nbins=40,
            opacity=0.6,
            labels={"doc_type": "Document type"},
        )
        st.plotly_chart(fig, use_container_width=True)

# metrics summary
st.subheader("Model performance")
st.metric("Accuracy", f"{result['accuracy']:.3f}")
st.metric("F1 (macro)", f"{result['f1_macro']:.3f}")
st.text(result["classification_report"])
