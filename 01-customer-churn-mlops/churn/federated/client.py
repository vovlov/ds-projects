"""Federated Learning client — local model update without sharing raw data.

Клиент федеративного обучения: телеком-оператор обучает локальную модель
на своих данных и отправляет только веса на сервер (данные клиентов не покидают оператора).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClientConfig:
    """Конфигурация локального обучения клиента.

    Configuration for local training on a federation client node.
    """

    client_id: str
    n_local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32


@dataclass
class ClientUpdate:
    """Обновление весов от клиента — передаётся на сервер агрегации.

    Weight update from a client node, sent to the aggregation server.
    """

    client_id: str
    weights: dict[str, np.ndarray]
    n_samples: int
    local_loss: float
    n_epochs: int


class FederatedClient:
    """Локальный участник федеративного обучения.

    Federated Learning participant — trains a logistic regression model
    on local data via mini-batch SGD. Only sends weight updates to server;
    raw customer data never leaves the client node (McMahan et al. 2017 §3).
    """

    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self._weights: dict[str, np.ndarray] = {}

    def receive_global_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Принять глобальные веса от сервера (broadcast step).

        Accept global model weights from the aggregation server.
        """
        self._weights = {k: v.copy() for k, v in weights.items()}

    def local_update(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
    ) -> ClientUpdate:
        """Локальное обновление: E эпох mini-batch SGD на локальных данных.

        Perform E epochs of mini-batch SGD on local data.
        Returns updated weights (not gradients) — FedAvg protocol.
        """
        rng = np.random.default_rng(abs(hash(self.config.client_id)) % (2**31))
        n, d = X.shape

        # Инициализация нулями при первом вызове (холодный старт)
        if "w" not in self._weights:
            self._weights = {
                "w": np.zeros(d),
                "b": np.array(0.0),
            }

        w = self._weights["w"].copy()
        b = float(self._weights["b"])

        for _ in range(self.config.n_local_epochs):
            indices = rng.permutation(n)
            for start in range(0, n, self.config.batch_size):
                batch_idx = indices[start : start + self.config.batch_size]
                X_b = X[batch_idx]
                y_b = y[batch_idx]

                logits = X_b @ w + b
                preds = _sigmoid(logits)

                err = preds - y_b
                grad_w = X_b.T @ err / len(batch_idx)
                grad_b = err.mean()

                w -= self.config.learning_rate * grad_w
                b -= self.config.learning_rate * grad_b

        final_loss = _binary_cross_entropy(X, y, w, b)

        return ClientUpdate(
            client_id=self.config.client_id,
            weights={"w": w, "b": np.array(b)},
            n_samples=n,
            local_loss=final_loss,
            n_epochs=self.config.n_local_epochs,
        )

    @property
    def weights(self) -> dict[str, np.ndarray]:
        """Текущие локальные веса."""
        return {k: v.copy() for k, v in self._weights.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _binary_cross_entropy(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    w: np.ndarray,
    b: float,
) -> float:
    logits = X @ w + b
    p = np.clip(_sigmoid(logits), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
