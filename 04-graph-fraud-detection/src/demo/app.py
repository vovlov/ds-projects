"""Streamlit demo for Graph Fraud Detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import generate_synthetic_transactions, get_edge_index, get_feature_matrix
from src.models.baseline.tabular import train_baseline

st.set_page_config(page_title="Graph Fraud Detection", layout="wide")


@st.cache_resource
def get_data_and_model():
    data = generate_synthetic_transactions(n_nodes=500, n_transactions=2000, fraud_rate=0.08)
    X, y = get_feature_matrix(data)
    result = train_baseline(X, y)
    return data, X, y, result


def main():
    st.title("Graph Fraud Detection")
    st.markdown(
        "Обнаружение мошеннических транзакций через анализ графа связей. "
        "Мошенники образуют кластеры — граф это ловит."
    )
    st.markdown("---")

    data, X, y, model_result = get_data_and_model()
    get_edge_index(data)

    tab1, tab2, tab3 = st.tabs(["Graph Analysis", "Model Results", "Score Transaction"])

    with tab1:
        st.header("Transaction Graph")

        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", len(data["nodes"]))
        col2.metric("Edges", len(data["edges"]))
        col3.metric("Fraud Rate", f"{y.mean():.1%}")

        # Feature distributions
        col_left, col_right = st.columns(2)

        with col_left:
            fig = px.histogram(
                x=X[:, 0],
                color=y.astype(str),
                nbins=50,
                title="Average Transaction Amount",
                labels={"x": "Amount", "color": "Is Fraud"},
                barmode="overlay",
                opacity=0.7,
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig = px.histogram(
                x=X[:, 2],
                color=y.astype(str),
                nbins=50,
                title="Account Age (days)",
                labels={"x": "Age (days)", "color": "Is Fraud"},
                barmode="overlay",
                opacity=0.7,
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        fig = px.scatter(
            x=X[:, 0],
            y=X[:, 2],
            color=y.astype(str),
            title="Avg Amount vs Account Age",
            labels={"x": "Avg Amount", "y": "Account Age", "color": "Is Fraud"},
            opacity=0.5,
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("CatBoost Baseline Results")

        col1, col2 = st.columns(2)
        col1.metric("F1 Score", f"{model_result['f1_score']:.4f}")
        col2.metric("ROC AUC", f"{model_result['roc_auc']:.4f}")

        report = model_result["report"]
        st.markdown("### Classification Report")
        st.json(report)

    with tab3:
        st.header("Score a Transaction")

        avg_amount = st.number_input("Average Transaction Amount", 0.0, 100000.0, 500.0)
        n_txn = st.number_input("Number of Transactions", 0, 1000, 10)
        account_age = st.number_input("Account Age (days)", 0.0, 3650.0, 180.0)

        if st.button("Score", type="primary"):
            features = np.array([[avg_amount, n_txn, account_age]])
            proba = float(model_result["model"].predict_proba(features)[0][1])

            if proba >= 0.7:
                risk_color, risk_label = "red", "HIGH RISK"
            elif proba >= 0.3:
                risk_color, risk_label = "orange", "MEDIUM RISK"
            else:
                risk_color, risk_label = "green", "LOW RISK"

            st.markdown(f"### Fraud Probability: `{proba:.1%}`")
            st.markdown(f"### Risk: :{risk_color}[{risk_label}]")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Fraud Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
