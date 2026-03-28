"""
Коллаборативная фильтрация на основе SVD.
SVD-based collaborative filtering recommender.

Используем TruncatedSVD из sklearn — работает на любой платформе,
не требует GPU или PyTorch.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.decomposition import TruncatedSVD


class CollaborativeRecommender:
    """Рекомендации через матричную факторизацию (SVD).
    Matrix factorization recommender using truncated SVD."""

    def __init__(self, n_components: int = 50, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(
            n_components=n_components, random_state=random_state
        )
        # Маппинги id <-> индекс / ID to index mappings
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.product_to_idx: dict[int, int] = {}
        self.idx_to_product: dict[int, int] = {}
        # Матрица предсказаний после SVD / Predicted ratings matrix
        self.predicted_ratings: np.ndarray | None = None
        self.user_item_matrix: np.ndarray | None = None

    def fit(self, interactions_df: pl.DataFrame) -> CollaborativeRecommender:
        """Строим user-item матрицу и обучаем SVD.
        Build user-item matrix and fit SVD decomposition."""
        # Уникальные пользователи и товары
        unique_users = sorted(interactions_df["user_id"].unique().to_list())
        unique_products = sorted(interactions_df["product_id"].unique().to_list())

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        self.product_to_idx = {pid: i for i, pid in enumerate(unique_products)}
        self.idx_to_product = {i: pid for pid, i in self.product_to_idx.items()}

        n_users = len(unique_users)
        n_products = len(unique_products)

        # Заполняем матрицу средним рейтингом по пользователю (implicit centering)
        self.user_item_matrix = np.zeros((n_users, n_products), dtype=np.float32)

        for row in interactions_df.iter_rows(named=True):
            u_idx = self.user_to_idx[row["user_id"]]
            p_idx = self.product_to_idx[row["product_id"]]
            self.user_item_matrix[u_idx, p_idx] = row["rating"]

        # SVD: понижаем размерность и восстанавливаем
        # Количество компонент не может превышать min(rows, cols)
        actual_components = min(
            self.n_components,
            n_users - 1,
            n_products - 1,
        )
        self.svd = TruncatedSVD(
            n_components=actual_components, random_state=self.random_state
        )

        user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.predicted_ratings = user_factors @ self.svd.components_

        return self

    def recommend(
        self, user_id: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Рекомендации для конкретного пользователя.
        Get top-K recommendations for a user. Returns (product_id, score) pairs."""
        if self.predicted_ratings is None:
            raise RuntimeError("Model not fitted. Call fit() first / Модель не обучена")

        if user_id not in self.user_to_idx:
            return []  # Неизвестный пользователь — пустой список

        u_idx = self.user_to_idx[user_id]
        scores = self.predicted_ratings[u_idx]

        # Исключаем товары, которые пользователь уже оценил
        already_rated = np.where(self.user_item_matrix[u_idx] > 0)[0]
        scores_copy = scores.copy()
        scores_copy[already_rated] = -np.inf

        # Берём top-K по предсказанному скору
        top_indices = np.argsort(scores_copy)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores_copy[idx] == -np.inf:
                break
            product_id = self.idx_to_product[idx]
            results.append((product_id, float(scores_copy[idx])))

        return results

    def evaluate(
        self, test_df: pl.DataFrame, top_k: int = 10, threshold: float = 4.0
    ) -> dict[str, float]:
        """Оценка качества: Precision@K, Recall@K, NDCG@K.
        Evaluate model with ranking metrics on test data."""
        if self.predicted_ratings is None:
            raise RuntimeError("Model not fitted / Модель не обучена")

        precisions = []
        recalls = []
        ndcgs = []

        # Группируем тест по пользователям
        test_grouped = test_df.group_by("user_id").agg([
            pl.col("product_id"),
            pl.col("rating"),
        ])

        for row in test_grouped.iter_rows(named=True):
            user_id = row["user_id"]
            if user_id not in self.user_to_idx:
                continue

            true_products = row["product_id"]
            true_ratings = row["rating"]

            # Релевантные — с рейтингом >= threshold
            relevant = {
                pid for pid, r in zip(true_products, true_ratings) if r >= threshold
            }
            if not relevant:
                continue

            # Получаем рекомендации
            recs = self.recommend(user_id, top_k=top_k)
            rec_ids = [pid for pid, _ in recs]

            # Precision@K
            hits = len(set(rec_ids) & relevant)
            precisions.append(hits / top_k)

            # Recall@K
            recalls.append(hits / len(relevant))

            # NDCG@K
            dcg = 0.0
            for i, pid in enumerate(rec_ids):
                if pid in relevant:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 потому что позиция с 1
            # Идеальный DCG
            ideal_dcg = sum(
                1.0 / np.log2(i + 2) for i in range(min(len(relevant), top_k))
            )
            ndcgs.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

        return {
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        }
