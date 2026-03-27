# %% [markdown]
# # Graph Fraud Detection — Exploratory Data Analysis
#
# Analyze synthetic transaction graph structure and fraud patterns.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import networkx as nx
import numpy as np
import plotly.express as px
from src.data.dataset import generate_synthetic_transactions, get_edge_index, get_feature_matrix

# %%
data = generate_synthetic_transactions(n_nodes=500, n_transactions=2000, fraud_rate=0.08)
X, y = get_feature_matrix(data)
edge_index = get_edge_index(data)

print(f"Nodes: {len(X)}")
print(f"Edges: {edge_index.shape[1]}")
print(f"Fraud rate: {y.mean():.2%}")
print(f"Fraud nodes: {y.sum()}")

# %% [markdown]
# ## 1. Node Feature Distributions

# %%
fig = px.histogram(
    x=X[:, 0],
    color=y.astype(str),
    nbins=50,
    title="Average Transaction Amount by Fraud Status",
    labels={"x": "Avg Amount", "color": "Is Fraud"},
    barmode="overlay",
    opacity=0.7,
)
fig.show()

# %%
fig = px.histogram(
    x=X[:, 2],
    color=y.astype(str),
    nbins=50,
    title="Account Age (days) by Fraud Status",
    labels={"x": "Account Age (days)", "color": "Is Fraud"},
    barmode="overlay",
    opacity=0.7,
)
fig.show()

# %% [markdown]
# ## 2. Graph Structure Analysis

# %%
G = nx.DiGraph()
for i, node in enumerate(data["nodes"]):
    G.add_node(i, is_fraud=node["is_fraud"])
for src, dst, amount, _ts in data["edges"]:
    G.add_edge(src, dst, weight=amount)

print(f"Graph density: {nx.density(G):.4f}")
print(f"Avg clustering (undirected): {nx.average_clustering(G.to_undirected()):.4f}")

# Degree distribution
in_degrees = [G.in_degree(n) for n in G.nodes()]
out_degrees = [G.out_degree(n) for n in G.nodes()]

fig = px.histogram(
    x=in_degrees,
    nbins=30,
    title="In-Degree Distribution",
    labels={"x": "In-Degree"},
)
fig.show()

# %% [markdown]
# ## 3. Fraud Node Connectivity

# %%
fraud_nodes = [n for n in G.nodes() if data["nodes"][n]["is_fraud"]]
normal_nodes = [n for n in G.nodes() if not data["nodes"][n]["is_fraud"]]

fraud_in_deg = [G.in_degree(n) for n in fraud_nodes]
normal_in_deg = [G.in_degree(n) for n in normal_nodes]

print(f"Fraud nodes avg in-degree: {np.mean(fraud_in_deg):.2f}")
print(f"Normal nodes avg in-degree: {np.mean(normal_in_deg):.2f}")
print(f"Fraud nodes avg out-degree: {np.mean([G.out_degree(n) for n in fraud_nodes]):.2f}")
print(f"Normal nodes avg out-degree: {np.mean([G.out_degree(n) for n in normal_nodes]):.2f}")

# Fraud-to-fraud edges
fraud_set = set(fraud_nodes)
f2f_edges = sum(1 for u, v in G.edges() if u in fraud_set and v in fraud_set)
total_edges = G.number_of_edges()
print(f"\nFraud-to-fraud edges: {f2f_edges} ({f2f_edges/total_edges:.2%} of total)")

# %% [markdown]
# ## 4. Feature Separability

# %%
fig = px.scatter(
    x=X[:, 0],
    y=X[:, 2],
    color=y.astype(str),
    title="Avg Amount vs Account Age (colored by fraud)",
    labels={"x": "Avg Amount", "y": "Account Age (days)", "color": "Is Fraud"},
    opacity=0.6,
)
fig.show()

# %% [markdown]
# ## Key Findings
#
# 1. **Fraud nodes** have higher average transaction amounts (lognormal mean=8 vs 5)
# 2. **Fraud nodes** have shorter account age (exponential scale=30 vs 365)
# 3. **Fraud-to-fraud connectivity** is elevated due to preferential attachment
# 4. **Graph structure** provides additional signal beyond tabular features
# 5. **Class imbalance** (~8% fraud) requires careful handling
