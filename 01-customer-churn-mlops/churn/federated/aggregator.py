"""FedAvg aggregation server — объединяет обновления клиентов без доступа к их данным.

FedAvg aggregation server (McMahan et al. 2017, "Communication-Efficient Learning
of Deep Networks from Decentralized Data", arxiv:1602.05629).

Алгоритм:
  1. Сервер broadcast глобальных весов выбранным клиентам.
  2. Каждый клиент обучает E эпох на локальных данных, возвращает обновлённые веса.
  3. Сервер агрегирует: w_global = Σ (n_k / N) · w_k, где n_k — размер датасета клиента k.
  4. Опционально: шум Гаусса для (ε, δ)-DP (Dwork 2006).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .client import ClientConfig, ClientUpdate, FederatedClient, _sigmoid


@dataclass
class FederatedConfig:
    """Конфигурация федеративного обучения.

    Federated training configuration.
    """

    n_rounds: int = 10
    # Доля клиентов, участвующих в каждом раунде (параметр C из FedAvg)
    fraction_clients: float = 1.0
    min_clients: int = 2
    # Масштаб шума Гаусса для дифференциальной приватности (0 = без DP)
    dp_noise_scale: float = 0.0
    seed: int = 42


@dataclass
class RoundResult:
    """Результат одного раунда федеративного обучения.

    Result of a single federation round.
    """

    round_num: int
    n_clients: int
    avg_loss: float
    global_weights: dict[str, np.ndarray]
    client_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "round_num": self.round_num,
            "n_clients": self.n_clients,
            "avg_loss": round(self.avg_loss, 6),
            "client_ids": self.client_ids,
        }


@dataclass
class FederationResult:
    """Итоговый результат всей федерации.

    Final result after all federation rounds.
    """

    n_rounds_completed: int
    final_weights: dict[str, np.ndarray]
    round_history: list[RoundResult]
    dp_noise_applied: bool

    @property
    def converged(self) -> bool:
        """Проверка сходимости: |loss_{t} - loss_{t-1}| < 1e-4.

        Check convergence by comparing loss across last two rounds.
        """
        if len(self.round_history) < 2:
            return False
        delta = abs(self.round_history[-1].avg_loss - self.round_history[-2].avg_loss)
        return delta < 1e-4

    def to_dict(self) -> dict:
        return {
            "n_rounds_completed": self.n_rounds_completed,
            "converged": self.converged,
            "dp_noise_applied": self.dp_noise_applied,
            "final_loss": round(self.round_history[-1].avg_loss, 6) if self.round_history else None,
            "round_history": [r.to_dict() for r in self.round_history],
        }


class FedAvgAggregator:
    """Сервер агрегации FedAvg.

    Aggregates local model updates from federated clients using
    weighted averaging by dataset size (FedAvg, McMahan et al. 2017).
    Supports optional Gaussian noise for (ε, δ)-differential privacy.
    """

    def __init__(self, config: FederatedConfig) -> None:
        self.config = config
        self._global_weights: dict[str, np.ndarray] = {}
        self._round_history: list[RoundResult] = []
        self._rng = np.random.default_rng(config.seed)

    def initialize(self, feature_dim: int) -> None:
        """Инициализация глобальной модели нулями (единый старт для всех клиентов).

        Initialize global model weights to zero (consensus initialization).
        """
        self._global_weights = {
            "w": np.zeros(feature_dim),
            "b": np.array(0.0),
        }
        self._round_history = []

    def aggregate(self, updates: list[ClientUpdate]) -> dict[str, np.ndarray]:
        """Взвешенное среднее весов клиентов (вес ∝ числу образцов).

        Compute weighted average of client weights:
            w_global = Σ_k (n_k / N) · w_k

        Опционально: шум Гаусса для DP (Dwork et al. 2006, TCC).
        """
        if not updates:
            raise ValueError("Нет обновлений для агрегации / No updates to aggregate")

        total_samples = sum(u.n_samples for u in updates)

        aggregated: dict[str, np.ndarray] = {}
        for key in updates[0].weights:
            aggregated[key] = sum(u.weights[key] * (u.n_samples / total_samples) for u in updates)

        # (ε, δ)-DP: шум Гаусса масштабируется на dp_noise_scale
        if self.config.dp_noise_scale > 0:
            for key in aggregated:
                noise = self._rng.normal(0.0, self.config.dp_noise_scale, aggregated[key].shape)
                aggregated[key] = aggregated[key] + noise

        return aggregated

    def run_round(
        self,
        clients: list[FederatedClient],
        datasets: list[tuple[np.ndarray, np.ndarray]],
        round_num: int,
    ) -> RoundResult:
        """Один раунд: выбор клиентов → broadcast → local update → aggregate.

        Execute one federation round:
        1. Sample C · K clients (C = fraction_clients, K = total clients).
        2. Broadcast global weights.
        3. Collect local updates.
        4. FedAvg aggregate.
        """
        if len(clients) != len(datasets):
            raise ValueError("Число клиентов и датасетов должно совпадать")

        n_select = max(
            self.config.min_clients,
            int(len(clients) * self.config.fraction_clients),
        )
        n_select = min(n_select, len(clients))
        selected = self._rng.choice(len(clients), n_select, replace=False)

        updates: list[ClientUpdate] = []
        for idx in selected:
            client = clients[idx]
            X, y = datasets[idx]
            client.receive_global_weights(self._global_weights)
            update = client.local_update(X, y)
            updates.append(update)

        new_weights = self.aggregate(updates)
        self._global_weights = new_weights

        avg_loss = float(np.mean([u.local_loss for u in updates]))

        result = RoundResult(
            round_num=round_num,
            n_clients=len(updates),
            avg_loss=avg_loss,
            global_weights={k: v.copy() for k, v in new_weights.items()},
            client_ids=[u.client_id for u in updates],
        )
        self._round_history.append(result)
        return result

    def train(
        self,
        clients: list[FederatedClient],
        datasets: list[tuple[np.ndarray, np.ndarray]],
    ) -> FederationResult:
        """Запустить полный цикл федеративного обучения.

        Run the full federation training loop for n_rounds rounds.
        """
        if not clients:
            raise ValueError("Нужен хотя бы один клиент / At least one client required")
        if len(clients) < self.config.min_clients:
            raise ValueError(
                f"Нужно минимум {self.config.min_clients} клиентов, получено {len(clients)}"
            )
        if len(clients) != len(datasets):
            raise ValueError("Число клиентов и датасетов должно совпадать")

        feature_dim = datasets[0][0].shape[1]
        if not self._global_weights:
            self.initialize(feature_dim)

        for r in range(self.config.n_rounds):
            self.run_round(clients, datasets, round_num=r + 1)

        return FederationResult(
            n_rounds_completed=self.config.n_rounds,
            final_weights={k: v.copy() for k, v in self._global_weights.items()},
            round_history=list(self._round_history),
            dp_noise_applied=self.config.dp_noise_scale > 0,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Предсказание вероятности оттока глобальной моделью.

        Predict churn probability using the aggregated global model.
        """
        if not self._global_weights:
            raise RuntimeError("Модель не обучена — вызовите train() или run_round() сначала")
        w = self._global_weights["w"]
        b = float(self._global_weights["b"])
        return _sigmoid(X @ w + b)

    @property
    def global_weights(self) -> dict[str, np.ndarray]:
        """Копия текущих глобальных весов."""
        return {k: v.copy() for k, v in self._global_weights.items()}

    @property
    def round_history(self) -> list[RoundResult]:
        return list(self._round_history)

    def reset(self) -> None:
        """Сброс состояния сервера (для тестовой изоляции / новой федерации)."""
        self._global_weights = {}
        self._round_history = []


def make_clients(
    n_clients: int,
    n_local_epochs: int = 5,
    learning_rate: float = 0.01,
) -> list[FederatedClient]:
    """Фабрика клиентов для быстрого создания тестовой федерации.

    Factory to quickly create a list of FederatedClient instances.
    """
    return [
        FederatedClient(
            ClientConfig(
                client_id=f"operator_{i + 1}",
                n_local_epochs=n_local_epochs,
                learning_rate=learning_rate,
            )
        )
        for i in range(n_clients)
    ]
