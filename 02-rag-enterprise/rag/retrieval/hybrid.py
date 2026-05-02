"""
Гибридный поиск: BM25 (лексический) + ChromaDB (семантический) + Reciprocal Rank Fusion.

Зачем hybrid? Семантический поиск хорошо работает для смысловых запросов, но
пропускает точные совпадения: коды ошибок, имена, артикулы, аббревиатуры.
BM25 ловит эти ключевые слова, но не понимает смысл. RRF объединяет ранги без
нормализации скоров — устойчив к разным масштабам оценок.

Recall@10: semantic ~65-78% → hybrid ~91% (Ashutosh Kumar Singh, Medium 2026).

BM25 graceful fallback: если rank_bm25 не установлен (CI без extras),
hybrid_search возвращает только семантические результаты.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import chromadb

from .store import search as semantic_search

if TYPE_CHECKING:
    pass


def _is_available() -> bool:
    """Проверить наличие rank_bm25 в окружении."""
    try:
        import rank_bm25  # noqa: F401

        return True
    except ImportError:
        return False


def _tokenize(text: str) -> list[str]:
    """Базовая токенизация: lowercase + split по не-буквенным символам."""
    return re.findall(r"\w+", text.lower())


@dataclass
class HybridIndex:
    """BM25-индекс поверх проиндексированных чанков.

    Хранит токенизированный корпус и оригинальные чанки для восстановления
    полного контекста (text + metadata) после ранжирования.
    """

    chunks: list[dict]
    tokenized_corpus: list[list[str]]
    _bm25: Any = field(default=None, repr=False)

    @classmethod
    def build(cls, chunks: list[dict]) -> HybridIndex:
        """Построить BM25-индекс из списка чанков."""
        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25_obj = None
        if _is_available():
            from rank_bm25 import BM25Okapi

            bm25_obj = BM25Okapi(tokenized)
        return cls(chunks=chunks, tokenized_corpus=tokenized, _bm25=bm25_obj)

    def bm25_search(self, query: str, k: int) -> list[dict]:
        """Ранжировать чанки по BM25-скору, вернуть top-k.

        Возвращает пустой список, если rank_bm25 не установлен.
        """
        if self._bm25 is None or not self.chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                **self.chunks[i],
                "bm25_score": float(scores[i]),
            }
            for i in ranked_indices
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    n_results: int = 5,
) -> list[dict]:
    """Объединить ранжированные списки через Reciprocal Rank Fusion.

    RRF(d) = Σ 1 / (k + rank(d)) для каждого ранкера.
    k=60: стандарт Cormack et al. 2009, снижает влияние топовых позиций.

    Ключ идентичности — text чанка: одинаковые тексты из разных ранкеров
    получают суммарный скор, что усиливает консенсусные результаты.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            key = item["text"]
            # rank + 1 чтобы первый ранг = 1, не 0
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in items:
                items[key] = {k: v for k, v in item.items() if k not in ("bm25_score", "distance")}

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:n_results]
    return [{**items[key], "rrf_score": round(scores[key], 6)} for key in sorted_keys]


def hybrid_search(
    query: str,
    collection: chromadb.Collection,
    hybrid_index: HybridIndex | None,
    n_results: int = 5,
) -> list[dict]:
    """Гибридный поиск: semantic + BM25 → RRF fusion.

    Если hybrid_index is None или rank_bm25 не установлен — graceful degradation
    к pure semantic search.
    """
    # Семантика всегда доступна
    semantic_k = n_results * 2  # берём больше кандидатов для fusion
    semantic_results = semantic_search(query, collection, n_results=semantic_k)

    # BM25 только если индекс построен
    if hybrid_index is not None and hybrid_index._bm25 is not None:
        bm25_results = hybrid_index.bm25_search(query, k=semantic_k)
        if bm25_results:
            return reciprocal_rank_fusion(
                [semantic_results, bm25_results],
                k=60,
                n_results=n_results,
            )

    # Fallback: semantic only, trim to n_results
    return semantic_results[:n_results]
