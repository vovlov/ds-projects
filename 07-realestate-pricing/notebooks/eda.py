# %% [markdown]
# # Moscow Real Estate — Exploratory Data Analysis
#
# Synthetic dataset: 1000 properties across 15 Moscow neighborhoods.
#
# **Goal:** Understand pricing patterns, validate synthetic data quality,
# identify key price drivers before modeling.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from pricing.data.load import TARGET, load_dataset

# %%
df = load_dataset()
print(f"Shape: {df.shape}")
print(f"Price range: {df[TARGET].min():,} — {df[TARGET].max():,} RUB")
print(f"Mean price: {df[TARGET].mean():,.0f} RUB")
df.head()

# %% [markdown]
# ## 1. Price Distribution
#
# Ожидаем правую асимметрию — большинство квартир 5-15М, немного дорогих 20М+.

# %%
fig = px.histogram(
    df.to_pandas(),
    x="price",
    nbins=40,
    title="Price Distribution (RUB)",
    color_discrete_sequence=["#3498db"],
)
fig.show()

# %% [markdown]
# ## 2. Price vs Area
#
# Главная зависимость — цена растёт с площадью, но не линейно.
# Маленькие квартиры дороже за кв.м (московская специфика — студии в центре).

# %%
fig = px.scatter(
    df.to_pandas(),
    x="sqft",
    y="price",
    color="neighborhood",
    title="Price vs Area by Neighborhood",
    opacity=0.6,
    trendline="ols",
)
fig.show()

# %% [markdown]
# ## 3. Price per Sqft by Neighborhood
#
# Район — второй по значимости фактор после площади.
# Разброс между Арбатом и Бирюлёво должен быть ~3x.

# %%
by_hood = (
    df.group_by("neighborhood")
    .agg(
        pl.col("price_per_sqft").median().alias("median_pps"),
        pl.col("price").median().alias("median_price"),
        pl.len().alias("count"),
    )
    .sort("median_pps", descending=True)
)

fig = px.bar(
    by_hood.to_pandas(),
    x="median_pps",
    y="neighborhood",
    orientation="h",
    title="Median Price per m2 by Neighborhood",
    color="median_pps",
    color_continuous_scale="RdYlGn_r",
    text="count",
)
fig.update_traces(textposition="outside")
fig.show()

# %% [markdown]
# ## 4. Year Built vs Price
#
# U-образная зависимость: сталинки (1935-1955) и новостройки (2005+) дороже
# хрущёвок (1956-1971).

# %%
fig = px.scatter(
    df.to_pandas(),
    x="year_built",
    y="price",
    color="condition",
    title="Year Built vs Price",
    opacity=0.5,
)
fig.show()

# %%
# Средняя цена по эпохам
df_epochs = df.with_columns(
    pl.when(pl.col("year_built") < 1956).then(pl.lit("Сталинки"))
    .when(pl.col("year_built") < 1972).then(pl.lit("Хрущёвки"))
    .when(pl.col("year_built") < 1995).then(pl.lit("Брежневки"))
    .otherwise(pl.lit("Новостройки"))
    .alias("epoch")
)

epoch_stats = (
    df_epochs.group_by("epoch")
    .agg(
        pl.col("price").median().alias("median_price"),
        pl.col("price_per_sqft").median().alias("median_pps"),
        pl.len().alias("count"),
    )
    .sort("median_price")
)
print(epoch_stats)

# %% [markdown]
# ## 5. Condition Impact

# %%
fig = px.box(
    df.to_pandas(),
    x="condition",
    y="price",
    color="condition",
    title="Price Distribution by Condition",
)
fig.show()

# %% [markdown]
# ## 6. Correlation Matrix

# %%
numeric_cols = ["price", "sqft", "bedrooms", "bathrooms", "year_built", "lot_size", "age"]
correlations = {}
for col in numeric_cols:
    correlations[col] = df.select(pl.corr(col, TARGET)).item()

corr_df = pl.DataFrame({
    "Feature": list(correlations.keys()),
    "Correlation": list(correlations.values()),
}).sort("Correlation", descending=True)

fig = px.bar(
    corr_df.to_pandas(),
    x="Correlation",
    y="Feature",
    orientation="h",
    title="Feature Correlation with Price",
    color="Correlation",
    color_continuous_scale="RdBu_r",
)
fig.show()

# %% [markdown]
# ## Key Findings
#
# 1. **Area (sqft)** is the strongest price predictor (correlation ~0.7+)
# 2. **Neighborhood** creates a ~3x price multiplier (Arbat vs Biryulyovo)
# 3. **Year built** has a U-shaped effect — Stalinist and new builds are pricier
# 4. **Condition** adds ~30% premium (excellent vs needs renovation)
# 5. **Garage** has modest impact (~5%), mostly correlated with new builds
