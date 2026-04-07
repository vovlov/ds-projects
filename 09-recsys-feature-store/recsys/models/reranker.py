"""
LLM-based re-ranker for two-tower retrieval candidates.
Переранжирование кандидатов двухбашенной модели с помощью LLM.

Паттерн: двухэтапный поиск (retrieve → re-rank).
1. Двухбашенная модель даёт топ-2K кандидатов (быстрый ANN поиск)
2. LLM переранжирует их, учитывая контекст пользователя и описание товаров
   (медленнее, но точнее для финального топ-K)

В CI-окружении без ANTHROPIC_API_KEY автоматически используется mock-режим,
сохраняющий кандидатов в исходном порядке — тесты не требуют LLM-доступа.
"""

from __future__ import annotations

import json
import os
from typing import Any

import polars as pl


class LLMReranker:
    """
    Re-ranks top-K retrieval candidates using Claude LLM.

    Переранжирует кандидатов, полученных от двухбашенной модели:
    - Формирует промпт с профилем пользователя + описанием товаров
    - Просит LLM выставить rerank_score ∈ [0,1] каждому кандидату
    - Возвращает отсортированный список с метаданными

    Graceful degradation:
    - Нет ANTHROPIC_API_KEY → mock-режим (порядок retrieval сохраняется)
    - Нет пакета anthropic → mock-режим
    - Любая ошибка LLM → fallback на mock

    Это критично для CI: тесты зелёные без LLM-доступа.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        mock: bool = False,
    ) -> None:
        self.model = model
        # Mock когда явно запрошен или нет API-ключа — CI должен быть зелёным
        self._mock = mock or not os.getenv("ANTHROPIC_API_KEY")

    def rerank(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        users: pl.DataFrame,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Re-rank candidates with LLM context understanding.

        Принимает кандидатов от двухбашенной модели и переранжирует их
        с учётом профиля пользователя. Передаём 2×top_k кандидатов LLM,
        чтобы дать пространство для манёвра.

        Args:
            user_id: идентификатор пользователя
            candidates: список (item_id, retrieval_score) от two-tower модели
            products: каталог товаров (категория, ценовой сегмент)
            users: таблица пользователей (возрастная группа, дней с регистрации)
            top_k: итоговое число рекомендаций

        Returns:
            Список dict с ключами: item_id, retrieval_score, rerank_score,
            category, price_tier, explanation
        """
        # Передаём LLM двойное количество кандидатов для лучшего выбора
        pool = candidates[: top_k * 2]

        if self._mock:
            return self._mock_rerank(pool, products, top_k)

        return self._llm_rerank(user_id, pool, products, users, top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        item_id: int,
        retrieval_score: float,
        rerank_score: float,
        products: pl.DataFrame,
        explanation: str = "",
    ) -> dict[str, Any]:
        """Собираем словарь результата с метаданными товара."""
        row = products.filter(pl.col("product_id") == item_id)
        return {
            "item_id": item_id,
            "retrieval_score": round(retrieval_score, 4),
            "rerank_score": round(rerank_score, 4),
            "category": row["category"][0] if len(row) > 0 else "unknown",
            "price_tier": row["price_tier"][0] if len(row) > 0 else "unknown",
            "explanation": explanation,
        }

    def _mock_rerank(
        self,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Fallback: preserve retrieval order when LLM is unavailable.

        Mock-режим: сохраняем порядок retrieval, не вызывая LLM.
        Нужен для тестов и для окружений без API-ключа.
        """
        return [
            self._build_result(iid, score, score, products, "mock re-ranking")
            for iid, score in candidates[:top_k]
        ]

    def _llm_rerank(
        self,
        user_id: int,
        candidates: list[tuple[int, float]],
        products: pl.DataFrame,
        users: pl.DataFrame,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Call Claude to re-rank based on user-item context.

        Формируем промпт: контекст пользователя + список кандидатов.
        LLM возвращает JSON-массив с rerank_score и кратким объяснением.
        При любой ошибке деградируем в mock.
        """
        try:
            import anthropic
        except ImportError:
            return self._mock_rerank(candidates, products, top_k)

        client = anthropic.Anthropic()

        # Контекст пользователя
        user_row = users.filter(pl.col("user_id") == user_id)
        if len(user_row) > 0:
            r = {k: user_row[k][0] for k in user_row.columns}
            user_ctx = (
                f"Age group: {r.get('age_group', '?')}, "
                f"Member for: {r.get('signup_days_ago', '?')} days"
            )
        else:
            user_ctx = "Unknown user"

        # Описание кандидатов
        item_lines = []
        for rank, (iid, score) in enumerate(candidates, 1):
            row = products.filter(pl.col("product_id") == iid)
            cat = row["category"][0] if len(row) > 0 else "?"
            tier = row["price_tier"][0] if len(row) > 0 else "?"
            item_lines.append(
                f"{rank}. id={iid} category={cat} price_tier={tier} retrieval_score={score:.3f}"
            )

        prompt = (
            f"User profile: {user_ctx}\n\n"
            "Candidate items (ordered by retrieval score):\n"
            + "\n".join(item_lines)
            + f"\n\nRe-rank these items for this user. "
            f"Return JSON array of exactly {top_k} objects:\n"
            '{"item_id": <int>, "rerank_score": <float 0-1>, '
            '"explanation": "<max 10 words>"}\n'
            "Return only valid JSON array, no other text."
        )

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            data: list[dict[str, Any]] = json.loads(response.content[0].text.strip())

            score_map = {iid: s for iid, s in candidates}
            results = []
            for item in sorted(data, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[
                :top_k
            ]:
                iid = int(item["item_id"])
                results.append(
                    self._build_result(
                        iid,
                        score_map.get(iid, 0.0),
                        float(item.get("rerank_score", 0.0)),
                        products,
                        str(item.get("explanation", "")),
                    )
                )
            return results

        except Exception:
            # Деградируем gracefully при любой LLM-ошибке (timeout, parse error)
            return self._mock_rerank(candidates, products, top_k)

    def __repr__(self) -> str:
        mode = "mock" if self._mock else f"llm({self.model})"
        return f"LLMReranker(mode={mode})"
