"""
Session-based рекомендации — декайное взвешивание истории сессии.
Session-based recommendations via decay-weighted session embedding.

Алгоритм (GRU4Rec-инспированный, Hidasi et al. 2016 ICLR):
    emb[item] — случайный вектор L2-нормированный (инициализация из seed)
    session_vec = Σ decay^(T-t) · emb[i_t] / Σ decay^(T-t)  — взвешенное среднее
    score(item) = cosine(session_vec, emb[item])               — следующий товар

Decay даёт свежим взаимодействиям больший вес (как GRU hidden state,
но без градиентов — полностью numpy, macOS x86_64 совместимо).

Источники:
    Hidasi et al. 2016 ICLR "Session-based Recommendations with RNNs"
    Ludewig & Jannach 2018 RecSys "Evaluation of Session-based Rec Algorithms"
    Koren et al. 2009 IEEE Computer (temporal decay in CF)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np


@dataclass
class SessionConfig:
    """Гиперпараметры session-based рекомендера.
    Hyperparameters for the session-based recommender.

    max_session_length: sliding window — только последние N взаимодействий.
    embedding_dim: размерность item-эмбеддингов (случайные, L2-норма).
    decay_factor: вес свежих взаимодействий ∈ (0, 1].
        0.5 — быстрое забывание; 1.0 — равные веса (mean-pooling).
    n_items: верхняя граница item_id для инициализации embedding matrix.
    seed: reprodicibility для эмбеддингов (иначе рестарт меняет ранжирование).
    """

    max_session_length: int = 20
    embedding_dim: int = 32
    decay_factor: float = 0.8
    n_items: int = 500
    seed: int = 42


@dataclass
class InteractionEvent:
    """Одно взаимодействие пользователя с товаром.
    Single user-item interaction event."""

    user_id: int
    item_id: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionState:
    """Состояние сессии одного пользователя.
    Per-user session state — sliding window of recent interactions."""

    user_id: int
    item_history: list[int] = field(default_factory=list)
    """Ordered oldest → newest (index 0 = oldest)."""
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_item(self, item_id: int, max_length: int) -> None:
        """Добавить взаимодействие, обрезать окно если нужно.
        Append interaction and trim to sliding window."""
        self.item_history.append(item_id)
        if len(self.item_history) > max_length:
            self.item_history = self.item_history[-max_length:]
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_length": len(self.item_history),
            "item_history": list(self.item_history),
            "last_updated": self.last_updated,
        }


@dataclass
class SessionRecommendation:
    """Одна рекомендация с оценкой сходства.
    Single session-based recommendation with similarity score."""

    item_id: int
    score: float
    rank: int


@dataclass
class SessionResult:
    """Результат session-based рекомендации.
    Session-based recommendation result."""

    user_id: int
    session_length: int
    recommendations: list[SessionRecommendation]
    method: str  # "session" | "popular_fallback"
    session_vector_norm: float  # L2-норма вектора сессии (0.0 при пустой сессии)


class SessionRecommender:
    """
    Session-based рекомендер — decay-weighted mean pooling item-эмбеддингов.
    Session-based recommender using decay-weighted item embedding aggregation.

    Каждый item_id получает случайный L2-нормированный вектор (фиксированный seed).
    Вектор сессии = взвешенное среднее эмбеддингов, где вес ∝ decay^(T-t).
    Ранжирование кандидатов: cosine similarity к вектору сессии.

    Cold start (пустая сессия): возвращает popular_item_ids как fallback.

    Item without registered embedding: автоматически инициализируется из seed.
    """

    def __init__(self, config: SessionConfig | None = None) -> None:
        self.config = config or SessionConfig()
        self._rng = np.random.default_rng(self.config.seed)
        # Embedding matrix: row i = embedding for item_id i (lazy for unknown ids)
        self._embedding_matrix: np.ndarray = self._init_embeddings()
        self._sessions: dict[int, SessionState] = {}
        # Popular item fallback — обновляется через record_interaction
        self._item_counts: dict[int, int] = {}

    def _init_embeddings(self) -> np.ndarray:
        """Инициализировать embedding matrix из фиксированного seed.
        Initialize L2-normalised random embeddings — reproducible."""
        rng = np.random.default_rng(self.config.seed)
        emb = rng.standard_normal((self.config.n_items, self.config.embedding_dim)).astype(
            np.float32
        )
        # L2-нормализация по строкам: cosine = dot-product для нормированных векторов
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return emb / norms

    def _get_embedding(self, item_id: int) -> np.ndarray:
        """Получить эмбеддинг item (с авто-инициализацией для новых item_id).
        Get embedding — auto-extends matrix for unseen item_ids."""
        if item_id < self.config.n_items:
            return self._embedding_matrix[item_id]
        # Детерминированный хэш для item вне матрицы
        # Deterministic hash for out-of-range items
        local_rng = np.random.default_rng(self.config.seed + item_id)
        vec = local_rng.standard_normal(self.config.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-9)

    def _compute_session_vector(self, item_history: list[int]) -> np.ndarray | None:
        """Вычислить вектор сессии = decay-weighted mean embedding.
        Compute session vector as exponential decay-weighted average.

        Позиция 0 (самая старая) получает вес decay^(T-1),
        позиция T-1 (самая новая) получает вес 1.0.
        """
        if not item_history:
            return None

        T = len(item_history)
        # decay^(T-1), decay^(T-2), ..., decay^0  для позиций 0..T-1
        exponents = np.arange(T - 1, -1, -1, dtype=np.float32)
        weights = self.config.decay_factor ** exponents  # (T,)
        weights /= weights.sum()

        embs = np.stack([self._get_embedding(iid) for iid in item_history])  # (T, d)
        session_vec = (weights[:, None] * embs).sum(axis=0)  # (d,)

        norm = float(np.linalg.norm(session_vec))
        if norm < 1e-9:
            return None
        return session_vec / norm

    def record_interaction(self, user_id: int, item_id: int) -> SessionState:
        """Записать взаимодействие пользователя с товаром.
        Record a user-item interaction and update session state.

        Если сессия не существует — создаётся автоматически (холодный старт).
        Session is created automatically on first interaction.
        """
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionState(user_id=user_id)
        self._sessions[user_id].add_item(item_id, self.config.max_session_length)
        self._item_counts[item_id] = self._item_counts.get(item_id, 0) + 1
        return self._sessions[user_id]

    def get_session(self, user_id: int) -> SessionState | None:
        """Получить текущую сессию пользователя.
        Get current session for user (None if no interactions recorded)."""
        return self._sessions.get(user_id)

    def recommend(
        self,
        user_id: int,
        candidate_ids: list[int] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> SessionResult:
        """
        Рекомендовать следующие товары на основе сессии.
        Recommend next items based on session history.

        Args:
            user_id: пользователь, для которого строим рекомендации.
            candidate_ids: список кандидатов для ранжирования.
                None → все item_ids в диапазоне [0, n_items).
            top_k: сколько рекомендаций вернуть.
            exclude_seen: исключить товары из текущей сессии.

        Returns:
            SessionResult с рекомендациями и диагностикой.
        """
        session = self._sessions.get(user_id)
        history = session.item_history if session else []

        if candidate_ids is None:
            candidate_ids = list(range(self.config.n_items))

        if exclude_seen and history:
            seen = set(history)
            candidate_ids = [iid for iid in candidate_ids if iid not in seen]

        if not candidate_ids:
            candidate_ids = list(range(min(top_k, self.config.n_items)))

        session_vec = self._compute_session_vector(history)

        if session_vec is None:
            # Cold start — ранжируем по популярности
            # Cold start — rank by popularity count
            recs = self._popular_fallback(candidate_ids, top_k)
            method = "popular_fallback"
            vec_norm = 0.0
        else:
            recs = self._cosine_rank(session_vec, candidate_ids, top_k)
            method = "session"
            vec_norm = 1.0  # нормированный вектор → норма = 1

        return SessionResult(
            user_id=user_id,
            session_length=len(history),
            recommendations=recs,
            method=method,
            session_vector_norm=vec_norm,
        )

    def _cosine_rank(
        self,
        session_vec: np.ndarray,
        candidate_ids: list[int],
        top_k: int,
    ) -> list[SessionRecommendation]:
        """Ранжировать кандидатов по cosine similarity к вектору сессии.
        Rank candidates by cosine similarity to session vector."""
        embs = np.stack([self._get_embedding(iid) for iid in candidate_ids])  # (C, d)
        # Оба вектора L2-нормированы → cosine = dot
        scores = embs @ session_vec  # (C,)
        top_indices = np.argsort(-scores)[:top_k]

        return [
            SessionRecommendation(
                item_id=candidate_ids[idx],
                score=round(float(scores[idx]), 6),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]

    def _popular_fallback(
        self,
        candidate_ids: list[int],
        top_k: int,
    ) -> list[SessionRecommendation]:
        """Cold start: ранжировать кандидатов по числу взаимодействий.
        Cold start fallback — rank by interaction count (popularity)."""
        scored = [(iid, self._item_counts.get(iid, 0)) for iid in candidate_ids]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SessionRecommendation(
                item_id=iid,
                score=round(float(cnt), 6),
                rank=rank + 1,
            )
            for rank, (iid, cnt) in enumerate(scored[:top_k])
        ]

    def reset_session(self, user_id: int) -> bool:
        """Сбросить сессию пользователя (например, после логаута).
        Reset user session — call on logout or session timeout."""
        if user_id in self._sessions:
            del self._sessions[user_id]
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Статистика рекомендера для мониторинга.
        Recommender statistics for monitoring."""
        return {
            "n_sessions": len(self._sessions),
            "n_known_items": len(self._item_counts),
            "avg_session_length": (
                sum(len(s.item_history) for s in self._sessions.values()) / len(self._sessions)
                if self._sessions
                else 0.0
            ),
            "embedding_dim": self.config.embedding_dim,
            "decay_factor": self.config.decay_factor,
        }
