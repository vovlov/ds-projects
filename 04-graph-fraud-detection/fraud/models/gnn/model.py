"""Graph Neural Network models for fraud detection.

Note: Requires PyTorch and PyTorch Geometric.
On platforms where torch is not available, use the tabular baseline instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def is_available() -> bool:
    """Check if PyTorch Geometric is available."""
    return TORCH_AVAILABLE


if TORCH_AVAILABLE:

    class GCNFraudDetector(torch.nn.Module):
        """Graph Convolutional Network for node classification."""

        def __init__(self, in_channels: int, hidden_channels: int = 32):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(hidden_channels, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.lin(x)
            return x

    class GraphSAGEFraudDetector(torch.nn.Module):
        """GraphSAGE model for node classification."""

        def __init__(self, in_channels: int, hidden_channels: int = 32):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(hidden_channels, 2)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.lin(x)
            return x

    def prepare_graph_data(
        X: np.ndarray, y: np.ndarray, edge_index: np.ndarray, train_ratio: float = 0.8
    ) -> Data:
        """Convert numpy arrays to PyTorch Geometric Data object."""
        n_nodes = len(y)
        n_train = int(n_nodes * train_ratio)

        perm = np.random.permutation(n_nodes)
        train_mask = np.zeros(n_nodes, dtype=bool)
        train_mask[perm[:n_train]] = True
        test_mask = ~train_mask

        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(y, dtype=torch.long),
        )
        data.train_mask = torch.tensor(train_mask)
        data.test_mask = torch.tensor(test_mask)
        return data

    def train_gnn(
        X: np.ndarray,
        y: np.ndarray,
        edge_index: np.ndarray,
        model_type: str = "gcn",
        epochs: int = 200,
        lr: float = 0.01,
    ) -> dict[str, Any]:
        """Train GNN model and return results."""
        from sklearn.metrics import classification_report, f1_score, roc_auc_score

        data = prepare_graph_data(X, y, edge_index)

        if model_type == "sage":
            model = GraphSAGEFraudDetector(X.shape[1])
        else:
            model = GCNFraudDetector(X.shape[1])

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # Class weights for imbalanced data
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=weight)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            proba = F.softmax(out, dim=1)[:, 1].numpy()
            pred = out.argmax(dim=1).numpy()

        test_mask = data.test_mask.numpy()
        y_test = y[test_mask]
        y_pred = pred[test_mask]
        y_proba = proba[test_mask]

        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        return {
            "model": model,
            "f1_score": f1,
            "roc_auc": auc,
            "report": report,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
