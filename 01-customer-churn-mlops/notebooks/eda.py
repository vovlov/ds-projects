# %% [markdown]
# # Customer Churn — Exploratory Data Analysis
#
# Dataset: Telco Customer Churn (IBM, Kaggle) — 7,043 customers, 21 features.
#
# **Goal:** Understand churn patterns, identify key drivers, engineer features.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from churn.data.load import TARGET, prepare_dataset

# %%
df = prepare_dataset(Path.cwd().parent / "data" / "raw.csv")
print(f"Shape: {df.shape}")
print(f"Churn rate: {df[TARGET].mean():.2%}")
df.head()

# %% [markdown]
# ## 1. Target Distribution

# %%
churn_counts = df.group_by(TARGET).len().sort(TARGET)
fig = px.pie(
    churn_counts.to_pandas(),
    values="len",
    names=TARGET,
    title="Churn Distribution",
    color_discrete_sequence=["#2ecc71", "#e74c3c"],
)
fig.show()

# %% [markdown]
# ## 2. Churn by Contract Type

# %%
contract_churn = (
    df.group_by("Contract")
    .agg(
        pl.col(TARGET).mean().alias("ChurnRate"),
        pl.len().alias("Count"),
    )
    .sort("ChurnRate", descending=True)
)

fig = px.bar(
    contract_churn.to_pandas(),
    x="Contract",
    y="ChurnRate",
    color="ChurnRate",
    text="Count",
    title="Churn Rate by Contract Type",
    color_continuous_scale="RdYlGn_r",
)
fig.update_traces(textposition="outside")
fig.show()

# %% [markdown]
# ## 3. Tenure Distribution

# %%
fig = px.histogram(
    df.to_pandas(),
    x="tenure",
    color="Churn",
    nbins=40,
    title="Tenure Distribution by Churn Status",
    barmode="overlay",
    opacity=0.7,
    color_discrete_sequence=["#2ecc71", "#e74c3c"],
)
fig.show()

# %% [markdown]
# ## 4. Monthly Charges vs Total Charges

# %%
fig = px.scatter(
    df.to_pandas(),
    x="tenure",
    y="MonthlyCharges",
    color="Churn",
    size="TotalCharges",
    title="Tenure vs Monthly Charges (size = Total Charges)",
    opacity=0.5,
    color_discrete_sequence=["#2ecc71", "#e74c3c"],
)
fig.show()

# %% [markdown]
# ## 5. Services Impact on Churn

# %%
services_churn = (
    df.group_by("NumServices")
    .agg(pl.col(TARGET).mean().alias("ChurnRate"), pl.len().alias("Count"))
    .sort("NumServices")
)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=services_churn["NumServices"].to_list(), y=services_churn["Count"].to_list(), name="Count"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(
        x=services_churn["NumServices"].to_list(),
        y=services_churn["ChurnRate"].to_list(),
        name="Churn Rate",
        mode="lines+markers",
        line={"color": "red"},
    ),
    secondary_y=True,
)
fig.update_layout(title="Number of Services vs Churn Rate")
fig.show()

# %% [markdown]
# ## 6. Feature Importance (Correlation with Churn)

# %%
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
                "AvgMonthlySpend", "ExpectedTotalCharges", "NumServices"]
correlations = {}
for col in numeric_cols:
    correlations[col] = df.select(pl.corr(col, TARGET)).item()

corr_df = pl.DataFrame({"Feature": list(correlations.keys()), "Correlation": list(correlations.values())})
corr_df = corr_df.sort("Correlation", descending=True)

fig = px.bar(
    corr_df.to_pandas(),
    x="Correlation",
    y="Feature",
    orientation="h",
    title="Feature Correlation with Churn",
    color="Correlation",
    color_continuous_scale="RdBu_r",
)
fig.show()

# %% [markdown]
# ## Key Findings
#
# 1. **Churn rate: ~26.5%** — significant class imbalance
# 2. **Month-to-month contracts** have highest churn (~43%)
# 3. **New customers** (tenure < 12 months) churn most frequently
# 4. **Higher monthly charges** correlate with higher churn
# 5. **More services** ≠ less churn — relationship is non-linear
# 6. **Strongest predictors:** Contract type, tenure, monthly charges
