"""
Two-tower retrieval model for recommendation systems.
Двухбашенная модель поиска кандидатов для рекомендательных систем.

Архитектура: две независимые башни (user tower и item tower) обучаются совместно
методом in-batch negative sampling. Во время инференса косинусное сходство
эмбеддингов используется как ANN (approximate nearest neighbor) поиск.

Architecture:
    User Tower: user_features → W_user → user_embedding (embedding_dim,)
    Item Tower: item_features → W_item → item_embedding (embedding_dim,)
    Score: cosine_similarity(user_embedding, item_embedding)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TowerConfig:
    """Гиперпараметры двухбашенной модели. / Two-tower hyperparameters."""

    embedding_dim: int = 32
    learning_rate: float = 0.01
    # Меньше эпох в конфиге по умолчанию, чтобы CI не тормозил
    n_epochs: int = 20
    batch_size: int = 512
    random_state: int = 42


class TwoTowerModel:
    """
    Two-tower retrieval model using gradient descent (no PyTorch required).

    Двухбашенная модель без PyTorch — обучается через numpy mini-batch GD.
    Подходит для macOS x86_64 и CI-окружений без GPU.

    Обучение:
        - Строим матрицы признаков пользователей и товаров (OHE + StandardScaler)
        - Для каждого мини-батча позитивных пар берём in-batch негативы
        - Оптимизируем softmax cross-entropy по косинусным скорам
        - После обучения L2-нормируем товарные эмбеддинги для быстрого ANN

    Инференс:
        - user_embedding = user_features @ W_user (L2-норм)
        - scores = user_embedding @ item_embeddings.T
        - Сортируем по убыванию, исключаем уже просмотренные товары
    """

    def __init__(self, config: TowerConfig | None = None) -> None:
        self.config = config or TowerConfig()

        # Tower weight matrices (инициализируются при fit)
        self.W_user: np.ndarray | None = None
        self.W_item: np.ndarray | None = None

        # Encoders fitted on training data
        self._user_cat_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._item_cat_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self._user_num_scaler = StandardScaler()

        # State preserved for inference
        self._user_ids: np.ndarray | None = None
        self._item_ids: np.ndarray | None = None
        self._user_features: np.ndarray | None = None
        self._item_features: np.ndarray | None = None
        # L2-нормированные эмбеддинги товаров для быстрого dot-product ANN
        self._item_embeddings: np.ndarray | None = None

        self.is_fitted = False

    # ------------------------------------------------------------------
    # Feature encoding helpers
    # ------------------------------------------------------------------

    def _encode_users(self, users: pl.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Build user feature matrix from a polars DataFrame.

        Строим матрицу признаков пользователей: OHE для категориальных
        (age_group) + StandardScaler для числовых (signup_days_ago).
        """
        cat_cols = [c for c in ["age_group"] if c in users.columns]
        num_cols = [c for c in ["signup_days_ago"] if c in users.columns]

        parts: list[np.ndarray] = []

        if cat_cols:
            cat_data = users.select(cat_cols).to_numpy()
            if fit:
                self._user_cat_ohe.fit(cat_data)
            parts.append(self._user_cat_ohe.transform(cat_data))

        if num_cols:
            num_data = users.select(num_cols).cast(pl.Float64).to_numpy()
            if fit:
                self._user_num_scaler.fit(num_data)
            parts.append(self._user_num_scaler.transform(num_data))

        if not parts:
            return np.ones((len(users), 1), dtype=np.float64)
        return np.hstack(parts).astype(np.float64)

    def _encode_items(self, products: pl.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Build item feature matrix from a polars DataFrame.

        OHE для category и price_tier — основные дискриминирующие признаки.
        """
        cat_cols = [c for c in ["category", "price_tier"] if c in products.columns]

        if not cat_cols:
            return np.ones((len(products), 1), dtype=np.float64)

        cat_data = products.select(cat_cols).to_numpy()
        if fit:
            self._item_cat_ohe.fit(cat_data)
        return self._item_cat_ohe.transform(cat_data).astype(np.float64)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        users: pl.DataFrame,
        products: pl.DataFrame,
        interactions: pl.DataFrame,
    ) -> TwoTowerModel:
        """
        Train both towers with in-batch negative sampling.

        Обучение обеих башен методом in-batch negative sampling:
        - Для мини-батча из B позитивных пар строим матрицу скоров (B×B)
        - Диагональ — позитивные пары, остальные — implicit negatives
        - Softmax cross-entropy заставляет модель разделять позитивы и негативы

        Args:
            users: DataFrame с колонками user_id, age_group, signup_days_ago
            products: DataFrame с колонками product_id, category, price_tier
            interactions: DataFrame с колонками user_id, product_id (или item_id)

        Returns:
            self (для цепочки вызовов / fluent interface)
        """
        rng = np.random.RandomState(self.config.random_state)

        # Сохраняем ID для маппинга индекс → ID при инференсе
        self._user_ids = users["user_id"].to_numpy()
        self._item_ids = products["product_id"].to_numpy()
        user_id_to_idx = {int(uid): i for i, uid in enumerate(self._user_ids)}
        item_id_to_idx = {int(iid): i for i, iid in enumerate(self._item_ids)}

        # Строим и кешируем матрицы признаков
        self._user_features = self._encode_users(users, fit=True)
        self._item_features = self._encode_items(products, fit=True)

        u_dim = self._user_features.shape[1]
        i_dim = self._item_features.shape[1]
        emb_dim = self.config.embedding_dim

        # Xavier-like init: масштаб 0.1 предотвращает взрыв градиентов
        self.W_user = rng.randn(u_dim, emb_dim) * 0.1
        self.W_item = rng.randn(i_dim, emb_dim) * 0.1

        # Определяем колонку item_id (разные схемы для synthetic vs movielens)
        item_col = "product_id" if "product_id" in interactions.columns else "item_id"

        # Строим массив позитивных пар (user_idx, item_idx)
        pairs: list[tuple[int, int]] = []
        for row in interactions.iter_rows(named=True):
            uid = int(row["user_id"])
            iid = int(row[item_col])
            if uid in user_id_to_idx and iid in item_id_to_idx:
                pairs.append((user_id_to_idx[uid], item_id_to_idx[iid]))

        if not pairs:
            self.is_fitted = True
            return self

        pairs_arr = np.array(pairs, dtype=np.int32)
        lr = self.config.learning_rate
        bs = self.config.batch_size

        for _epoch in range(self.config.n_epochs):
            rng.shuffle(pairs_arr)

            for start in range(0, len(pairs_arr), bs):
                batch = pairs_arr[start : start + bs]
                u_idx, i_idx = batch[:, 0], batch[:, 1]

                # Forward: compute embeddings for batch
                U = self._user_features[u_idx] @ self.W_user  # (B, emb_dim)
                V = self._item_features[i_idx] @ self.W_item  # (B, emb_dim)

                # In-batch similarity matrix; diagonal = positives
                scores = U @ V.T  # (B, B)

                # Numerically stable softmax
                exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
                softmax = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-9)

                # Gradient wrt scores: softmax - one_hot(target)
                grad = softmax.copy()
                grad[np.arange(len(batch)), np.arange(len(batch))] -= 1.0
                grad /= len(batch)

                # Backprop to tower weights
                X_u = self._user_features[u_idx]
                X_i = self._item_features[i_idx]
                self.W_user -= lr * (X_u.T @ (grad @ V))
                self.W_item -= lr * (X_i.T @ (grad.T @ U))

        # Предвычисляем и L2-нормируем товарные эмбеддинги один раз —
        # это позволяет использовать dot product вместо полного cosine similarity
        self._item_embeddings = self._item_features @ self.W_item
        norms = np.linalg.norm(self._item_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self._item_embeddings = self._item_embeddings / norms

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_user_embedding(self, user_id: int, users: pl.DataFrame) -> np.ndarray | None:
        """
        Compute L2-normalized user embedding for a given user_id.

        Возвращает L2-нормированный эмбеддинг пользователя или None, если
        пользователь не найден.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling get_user_embedding()")

        row = users.filter(pl.col("user_id") == user_id)
        if len(row) == 0:
            return None

        feats = self._encode_users(row, fit=False)
        emb = feats @ self.W_user  # (1, emb_dim)
        norm = np.linalg.norm(emb)
        return (emb / norm).flatten() if norm > 1e-9 else emb.flatten()

    def recommend(
        self,
        user_id: int,
        users: pl.DataFrame,
        interactions: pl.DataFrame,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Retrieve top-K items via cosine ANN search.

        Возвращает топ-K товаров с наибольшим косинусным сходством к
        пользовательскому эмбеддингу. Уже просмотренные товары исключаются.

        Args:
            user_id: идентификатор пользователя
            users: таблица пользователей (для построения эмбеддинга)
            interactions: история взаимодействий (для фильтрации уже виденных)
            top_k: число рекомендаций

        Returns:
            Список (item_id, cosine_score) отсортированный по убыванию скора
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        u_emb = self.get_user_embedding(user_id, users)
        if u_emb is None:
            return []

        # Dot product с L2-нормированными item_embeddings = cosine similarity
        assert self._item_embeddings is not None
        scores = (u_emb @ self._item_embeddings.T).flatten()

        # Исключаем товары, с которыми пользователь уже взаимодействовал
        item_col = "product_id" if "product_id" in interactions.columns else "item_id"
        rated = set(interactions.filter(pl.col("user_id") == user_id)[item_col].to_list())

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[tuple[int, float]] = []
        for idx, score in ranked:
            item_id = int(self._item_ids[idx])  # type: ignore[index]
            if item_id not in rated:
                results.append((item_id, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Сохраняем модель на диск / Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> TwoTowerModel:
        """Загружаем модель с диска / Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"TwoTowerModel(embedding_dim={self.config.embedding_dim}, "
            f"n_epochs={self.config.n_epochs}, status={status})"
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        users: pl.DataFrame,
        products: pl.DataFrame,
        test_interactions: pl.DataFrame,
        train_interactions: pl.DataFrame,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Compute Precision@K, Recall@K, NDCG@K on a test split.

        Оцениваем качество ранжирования: для каждого пользователя в тесте
        предсказываем top-K (по train истории) и сравниваем с test ground truth.
        """
        item_col = "product_id" if "product_id" in test_interactions.columns else "item_id"

        precisions, recalls, ndcgs = [], [], []

        # Берём пользователей, у которых есть и train, и test взаимодействия
        test_users = test_interactions["user_id"].unique().to_list()

        for uid in test_users:
            recs = self.recommend(uid, users, train_interactions, top_k=top_k)
            rec_ids = [iid for iid, _ in recs]

            relevant = set(test_interactions.filter(pl.col("user_id") == uid)[item_col].to_list())
            if not relevant:
                continue

            hits = [1 if iid in relevant else 0 for iid in rec_ids]
            precision = sum(hits) / top_k
            recall = sum(hits) / len(relevant)

            # NDCG: взвешиваем по позиции
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), top_k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        return {
            f"precision@{top_k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
            f"ndcg@{top_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "n_users_evaluated": len(precisions),
        }
