"""Streamlit-дашборд для анализа и оценки недвижимости.

Три вкладки:
1. EDA — распределения цен, зависимость от площади, районы
2. Model — метрики и важность признаков
3. Estimator — форма для оценки конкретной квартиры
"""

from __future__ import annotations

import pickle
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from ..data.load import CONDITION_MAP, NEIGHBORHOODS, load_dataset

st.set_page_config(page_title="Moscow Real Estate", layout="wide")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


@st.cache_data
def get_data() -> pl.DataFrame:
    return load_dataset()


@st.cache_resource
def get_model_and_results() -> tuple:
    model_path = ARTIFACTS_DIR / "model.pkl"
    results_path = ARTIFACTS_DIR / "results.pkl"
    model = None
    results = None
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    if results_path.exists():
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    return model, results


def main() -> None:
    st.title("Moscow Real Estate Analysis & Pricing")
    st.markdown("---")

    df = get_data()
    model, results = get_model_and_results()

    tab1, tab2, tab3 = st.tabs(["EDA", "Model Results", "Price Estimator"])

    # -- EDA Tab -------------------------------------------------------
    with tab1:
        st.header("Exploratory Data Analysis")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Properties", f"{len(df):,}")
        col2.metric("Avg Price", f"{df['price'].mean():,.0f} RUB")
        col3.metric("Avg Sqft", f"{df['sqft'].mean():.0f} m2")
        col4.metric("Neighborhoods", str(df["neighborhood"].n_unique()))

        col_left, col_right = st.columns(2)

        with col_left:
            fig = px.histogram(
                df.to_pandas(),
                x="price",
                nbins=40,
                title="Price Distribution",
                labels={"price": "Price (RUB)"},
                color_discrete_sequence=["#3498db"],
            )
            fig.update_layout(yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig = px.scatter(
                df.to_pandas(),
                x="sqft",
                y="price",
                color="neighborhood",
                title="Price vs Area by Neighborhood",
                labels={"sqft": "Area (m2)", "price": "Price (RUB)"},
                opacity=0.6,
            )
            st.plotly_chart(fig, use_container_width=True)

        col_left2, col_right2 = st.columns(2)

        with col_left2:
            # Median price by neighborhood
            by_hood = (
                df.group_by("neighborhood")
                .agg(pl.col("price").median().alias("median_price"))
                .sort("median_price", descending=True)
            )
            fig = px.bar(
                by_hood.to_pandas(),
                x="median_price",
                y="neighborhood",
                orientation="h",
                title="Median Price by Neighborhood",
                labels={"median_price": "Median Price (RUB)", "neighborhood": ""},
                color="median_price",
                color_continuous_scale="RdYlGn_r",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right2:
            fig = px.box(
                df.to_pandas(),
                x="condition",
                y="price",
                color="condition",
                title="Price by Condition",
                labels={"condition": "Condition", "price": "Price (RUB)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # -- Model Results Tab --------------------------------------------
    with tab2:
        st.header("Model Performance")

        if results is None:
            st.warning("No model results found. Run `train.py` first.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("RMSE", f"{results.get('rmse', 0):,.0f}")
            col2.metric("MAE", f"{results.get('mae', 0):,.0f}")
            col3.metric("MAPE", f"{results.get('mape', 0):.1%}")
            col4.metric("R2", f"{results.get('r2', 0):.4f}")

            st.markdown(f"**Model:** {results.get('model_type', 'N/A')}")

            if "feature_importances" in results:
                imp = results["feature_importances"]
                imp_sorted = dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))
                fig = px.bar(
                    x=list(imp_sorted.values()),
                    y=list(imp_sorted.keys()),
                    orientation="h",
                    title="Feature Importance",
                    labels={"x": "Importance", "y": "Feature"},
                    color=list(imp_sorted.values()),
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(yaxis={"autorange": "reversed"})
                st.plotly_chart(fig, use_container_width=True)

    # -- Price Estimator Tab ------------------------------------------
    with tab3:
        st.header("Estimate Property Price")

        if model is None:
            st.warning("Model not loaded. Run `train.py` first.")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            sqft = st.number_input("Area (m2)", 15, 500, 65)
            bedrooms = st.number_input("Bedrooms", 1, 10, 2)
            bathrooms = st.number_input("Bathrooms", 1, 5, 1)

        with col2:
            year_built = st.number_input("Year Built", 1900, 2026, 2015)
            lot_size = st.number_input("Lot Size (m2)", 0, 2000, 0)
            garage = st.selectbox("Garage", ["no", "yes"])

        with col3:
            neighborhood = st.selectbox("Neighborhood", list(NEIGHBORHOODS.keys()))
            condition = st.selectbox("Condition", list(CONDITION_MAP.keys()))

        if st.button("Estimate Price", type="primary"):
            import numpy as np

            from ..data.load import CURRENT_YEAR

            age = CURRENT_YEAR - year_built
            has_garage = garage

            features = np.array(
                [
                    sqft,
                    bedrooms,
                    bathrooms,
                    year_built,
                    lot_size,
                    age,
                    neighborhood,
                    condition,
                    has_garage,
                ],
                dtype=object,
            )

            prediction = float(model.predict(features.reshape(1, -1))[0])
            estimated = max(int(round(prediction, -4)), 1_000_000)

            mape = results.get("mape", 0.10) if results else 0.10
            margin = max(mape, 0.05)
            low = int(round(estimated * (1 - margin), -4))
            high = int(round(estimated * (1 + margin), -4))

            st.markdown(f"### Estimated Price: **{estimated:,} RUB**")
            st.markdown(f"Confidence interval: {low:,} -- {high:,} RUB")

            # Gauge
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=estimated / 1_000_000,
                    number={"suffix": "M RUB"},
                    title={"text": "Estimated Price"},
                    gauge={
                        "axis": {"range": [3, 30]},
                        "bar": {"color": "#3498db"},
                        "steps": [
                            {"range": [3, 10], "color": "#d4edda"},
                            {"range": [10, 20], "color": "#fff3cd"},
                            {"range": [20, 30], "color": "#f8d7da"},
                        ],
                    },
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top factors
            if results and "feature_importances" in results:
                st.subheader("Top contributing factors")
                sorted_imp = sorted(
                    results["feature_importances"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                for name, importance in sorted_imp[:5]:
                    st.write(f"- **{name}**: {importance:.1f}")


if __name__ == "__main__":
    main()
