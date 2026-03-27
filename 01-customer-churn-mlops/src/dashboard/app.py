"""Streamlit dashboard for Customer Churn analysis and prediction."""

from __future__ import annotations

import pickle
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ..data.load import prepare_dataset

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@st.cache_data
def load_data():
    return prepare_dataset(DATA_DIR / "raw.csv")


@st.cache_resource
def load_model():
    model_path = ARTIFACTS_DIR / "model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def main():
    st.title("Customer Churn Analysis & Prediction")
    st.markdown("---")

    df = load_data()
    model = load_model()

    tab1, tab2, tab3 = st.tabs(["EDA", "Model Results", "Predict"])

    # ── EDA Tab ──────────────────────────────────────────────
    with tab1:
        st.header("Exploratory Data Analysis")

        col1, col2, col3, col4 = st.columns(4)
        total = len(df)
        churned = df.filter(pl.col("Churn") == 1).height
        col1.metric("Total Customers", f"{total:,}")
        col2.metric("Churned", f"{churned:,}")
        col3.metric("Churn Rate", f"{churned / total:.1%}")
        col4.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

        col_left, col_right = st.columns(2)

        with col_left:
            # Churn by Contract type
            contract_churn = (
                df.group_by("Contract")
                .agg(
                    pl.col("Churn").mean().alias("ChurnRate"),
                    pl.len().alias("Count"),
                )
                .sort("ChurnRate", descending=True)
            )
            fig = px.bar(
                contract_churn.to_pandas(),
                x="Contract",
                y="ChurnRate",
                color="ChurnRate",
                title="Churn Rate by Contract Type",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Tenure distribution by churn
            fig = px.histogram(
                df.to_pandas(),
                x="tenure",
                color="Churn",
                nbins=40,
                title="Tenure Distribution by Churn Status",
                barmode="overlay",
                opacity=0.7,
            )
            st.plotly_chart(fig, use_container_width=True)

        col_left2, col_right2 = st.columns(2)

        with col_left2:
            # Monthly charges distribution
            fig = px.box(
                df.to_pandas(),
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges by Churn Status",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right2:
            # Services count vs churn
            services_churn = (
                df.group_by("NumServices")
                .agg(pl.col("Churn").mean().alias("ChurnRate"))
                .sort("NumServices")
            )
            fig = px.line(
                services_churn.to_pandas(),
                x="NumServices",
                y="ChurnRate",
                markers=True,
                title="Churn Rate by Number of Services",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Model Results Tab ────────────────────────────────────
    with tab2:
        st.header("Model Performance")

        results_path = ARTIFACTS_DIR / "results.pkl"
        if results_path.exists():
            with open(results_path, "rb") as f:
                results = pickle.load(f)

            col1, col2, col3 = st.columns(3)
            col1.metric("F1 Score", f"{results['f1_score']:.4f}")
            col2.metric("ROC AUC", f"{results['roc_auc']:.4f}")
            col3.metric("Model", results.get("model_type", "CatBoost"))

            # Feature importance
            if "feature_importances" in results:
                imp = results["feature_importances"]
                imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15])
                fig = px.bar(
                    x=list(imp_sorted.values()),
                    y=list(imp_sorted.keys()),
                    orientation="h",
                    title="Top 15 Feature Importances",
                    labels={"x": "Importance", "y": "Feature"},
                )
                fig.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model results found. Run training first: `make train-churn`")

    # ── Predict Tab ──────────────────────────────────────────
    with tab3:
        st.header("Predict Churn for a Customer")

        if model is None:
            st.warning("Model not loaded. Run training first.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        with col3:
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 10000.0, monthly * tenure)

        if st.button("Predict", type="primary"):
            num_services = sum(
                [
                    security == "Yes",
                    backup == "Yes",
                    protection == "Yes",
                    support == "Yes",
                    tv == "Yes",
                    movies == "Yes",
                ]
            )
            features = [
                gender,
                partner,
                dependents,
                phone,
                multi,
                internet,
                security,
                backup,
                protection,
                support,
                tv,
                movies,
                contract,
                paperless,
                payment,
                senior,
                tenure,
                monthly,
                total,
                total / (tenure + 1),
                monthly * tenure,
                "new" if tenure <= 12 else "mid" if tenure <= 36 else "long",
                num_services,
            ]

            proba = float(model.predict_proba([features])[0][1])

            if proba >= 0.7:
                risk_color, risk_label = "red", "HIGH RISK"
            elif proba >= 0.4:
                risk_color, risk_label = "orange", "MEDIUM RISK"
            else:
                risk_color, risk_label = "green", "LOW RISK"

            st.markdown(f"### Churn Probability: `{proba:.1%}`")
            st.markdown(f"### Risk Level: :{risk_color}[{risk_label}]")

            # Gauge chart
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={"text": "Churn Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, 40], "color": "#d4edda"},
                            {"range": [40, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
