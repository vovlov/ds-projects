"""Multi-Query Retrieval с RRF-агрегацией вариантов запроса.

Паттерн 2025-2026: вместо одного запроса генерируем N переформулировок,
получаем результаты для каждой и объединяем через Reciprocal Rank Fusion.

Источники:
- LangChain MultiQueryRetriever docs (2024)
- RAG Fusion: arxiv:2402.03367 (Rackauckas 2024)
- Anthropic Contextual Retrieval blog (2024)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


@dataclass
class MultiQueryConfig:
    """Конфигурация Multi-Query Retrieval."""

    n_variants: int = 3
    rrf_k: int = 60  # RRF параметр (тот же что в hybrid.py для согласованности)


@dataclass
class MultiQueryResult:
    """Результат Multi-Query Retrieval."""

    chunks: list[dict]
    query_variants: list[str]
    consistency_score: float
    retrieval_method: str = "multi_query"

    def to_dict(self) -> dict:
        return {
            "n_chunks": len(self.chunks),
            "query_variants": self.query_variants,
            "consistency_score": self.consistency_score,
            "retrieval_method": self.retrieval_method,
        }


# Стоп-слова для keyword extraction — пропускаем при сборке ключевых слов
_STOP_WORDS = frozenset(
    {
        "what",
        "how",
        "why",
        "when",
        "where",
        "which",
        "who",
        "does",
        "can",
        "will",
        "the",
        "for",
        "and",
        "with",
        "this",
        "that",
        "from",
        "are",
        "you",
        "have",
        "has",
        "been",
        "there",
        "they",
        "their",
        "was",
        "were",
        "not",
        "but",
        "about",
        "all",
        "any",
        "just",
        "also",
        "into",
        "then",
        "than",
        "more",
        "some",
        "its",
        "our",
        "your",
    }
)


def _rule_based_variants(question: str, n: int) -> list[str]:
    """Генерирует варианты запроса без LLM — детерминированно, CI-safe.

    Три стратегии:
    1. Оригинал — всегда первый
    2. Ключевые слова — убираем стоп-слова, берём существительные/термины
    3. Переформулировка — меняем синтаксис без потери смысла
    """
    variants: list[str] = [question]

    # Стратегия 2: keyword extraction
    tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_\-]*[A-Za-z0-9]\b", question)
    keywords = [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) > 2]
    if keywords and len(variants) < n:
        kw_query = " ".join(keywords[:6])
        if kw_query != question:
            variants.append(kw_query)

    # Стратегия 3: переформулировка
    if len(variants) < n:
        q_stripped = question.strip().rstrip("?").rstrip(".")
        lower = q_stripped.lower()
        if lower.startswith(("what is", "what are")):
            reformulated = f"Explain {q_stripped[len('what is') :].strip()}"
        elif lower.startswith(("how does", "how do", "how can")):
            core = re.sub(r"^how\s+\w+\s+", "", lower, count=1)
            reformulated = f"Describe the process of {core}"
        elif lower.startswith("explain"):
            reformulated = f"What is meant by {q_stripped[len('explain') :].strip()}"
        elif lower.startswith("describe"):
            reformulated = f"What is {q_stripped[len('describe') :].strip()}"
        else:
            # Общий случай: добавляем контекстный суффикс
            reformulated = f"{q_stripped} definition and examples"

        if reformulated and reformulated.lower() != question.lower():
            variants.append(reformulated)

    return variants[:n]


def _llm_variants(question: str, n: int) -> list[str]:
    """Генерирует варианты через Claude Haiku (production).

    Graceful fallback на rule-based при отсутствии ключа или ошибке.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _rule_based_variants(question, n)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"Generate {n - 1} alternative phrasings of the following question "
            f"for document retrieval. Each phrasing should emphasize different aspects. "
            f"Return only the questions, one per line, no numbering or bullets.\n\n"
            f"Original question: {question}"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_lines = response.content[0].text.strip().split("\n")
        llm_variants = [line.strip() for line in raw_lines if line.strip()]
        return ([question] + llm_variants)[:n]
    except Exception:
        return _rule_based_variants(question, n)


def generate_query_variants(
    question: str,
    n: int = 3,
    use_llm: bool = False,
) -> list[str]:
    """Генерирует N вариантов запроса для Multi-Query Retrieval.

    Args:
        question: Исходный вопрос пользователя.
        n: Количество вариантов (включая оригинал).
        use_llm: Использовать Claude Haiku для генерации. Fallback на rule-based.

    Returns:
        Список из N вариантов запроса, первый = оригинал.
    """
    if n <= 1:
        return [question]
    if use_llm:
        return _llm_variants(question, n)
    return _rule_based_variants(question, n)


def compute_consistency_score(results_per_variant: list[list[dict]]) -> float:
    """Измеряет согласованность результатов разных вариантов запроса.

    Высокий score (→1.0) = одни и те же чанки находятся при любой формулировке —
    модель уверенно знает ответ. Низкий (→0.0) = результаты нестабильны.

    Метрика: средний pairwise Jaccard overlap первых 5 чанков каждого варианта.
    """
    if len(results_per_variant) < 2:
        return 1.0

    def _chunk_ids(chunks: list[dict]) -> frozenset[str]:
        # Используем первые 100 символов текста как идентификатор чанка
        return frozenset(c.get("text", "")[:100] for c in chunks[:5] if c.get("text"))

    id_sets = [_chunk_ids(r) for r in results_per_variant]

    pairwise_scores: list[float] = []
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            a, b = id_sets[i], id_sets[j]
            if not a and not b:
                pairwise_scores.append(1.0)
            elif not a or not b:
                pairwise_scores.append(0.0)
            else:
                pairwise_scores.append(len(a & b) / len(a | b))

    return round(sum(pairwise_scores) / len(pairwise_scores), 3) if pairwise_scores else 1.0


def multi_query_retrieve(
    question: str,
    collection,
    hybrid_index,
    n_results: int = 5,
    config: MultiQueryConfig | None = None,
    use_llm: bool = False,
) -> MultiQueryResult:
    """Multi-Query Retrieval: N вариантов запроса → RRF агрегация.

    Алгоритм (RAG Fusion, Rackauckas 2024):
    1. Генерируем N переформулировок исходного запроса
    2. Для каждой выполняем hybrid search (BM25+vector+RRF)
    3. Объединяем все результаты через RRF — одинаковые чанки
       из разных вариантов поднимаются вверх
    4. Вычисляем consistency_score как меру надёжности ответа

    Args:
        question: Исходный запрос пользователя.
        collection: ChromaDB коллекция.
        hybrid_index: BM25-индекс (или None → только semantic).
        n_results: Сколько чанков вернуть итого.
        config: Параметры Multi-Query (n_variants, rrf_k).
        use_llm: Генерировать варианты через LLM (production mode).

    Returns:
        MultiQueryResult с объединёнными чанками и метаданными.
    """
    from .hybrid import hybrid_search, reciprocal_rank_fusion

    if config is None:
        config = MultiQueryConfig()

    variants = generate_query_variants(question, n=config.n_variants, use_llm=use_llm)

    results_per_variant: list[list[dict]] = []
    for variant in variants:
        # Запрашиваем 2×n_results для каждого варианта — после RRF обрежем до n_results
        chunks = hybrid_search(
            variant,
            collection,
            hybrid_index,
            n_results=n_results * 2,
        )
        results_per_variant.append(chunks)

    merged = reciprocal_rank_fusion(
        results_per_variant,
        k=config.rrf_k,
        n_results=n_results,
    )

    consistency = compute_consistency_score(results_per_variant)

    return MultiQueryResult(
        chunks=merged,
        query_variants=variants,
        consistency_score=consistency,
    )
